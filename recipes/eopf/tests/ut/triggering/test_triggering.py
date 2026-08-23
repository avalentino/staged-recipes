#
# Copyright 2026 CS - SOPRA STERIA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import json
import logging
import os
import sys
import traceback
import types
import uuid
from pathlib import Path
from pprint import pprint
from typing import Any
from unittest import mock
from warnings import warn

import pytest
import xarray as xr
import yaml
from click.testing import CliRunner
from xarray import DataTree

from eopf import open_datatree
from eopf.cli.cli_triggering_triggers import trigger_cli
from eopf.common import file_utils
from eopf.common.constants import EOPF_CPM_DEFAULT_CONFIG_FILE
from eopf.common.env_utils import resolve_env_vars
from eopf.common.file_utils import AnyPath
from eopf.common.temp_utils import EOTemporaryFolder
from eopf.common.types import ImmediateJob
from eopf.computing import MappingAuxiliary, MappingDataType
from eopf.config.config import EOConfiguration
from eopf.exceptions.errors import TriggeringConfigurationError, TriggerInvalidWorkflow
from eopf.common import (
    reset_execution_status,
    set_execution_exit_code,
)
from eopf.logging import PROGRESS_LOG_LEVEL, log_progress, reset_logging
from eopf.logging.log import DEFAULT_LOG_FORMAT
from eopf.product.datatree_validation import ValidationMode
from eopf.store.reader_registry import EOReaderRegistry
from eopf.store.writer_registry import EOWriterRegistry
from eopf.triggering import EORunner
from eopf.triggering.general_utils import resolve_storage_options
from eopf.triggering.interfaces import EOIOParserResult, EOOutputProductParserResult, PathType
from eopf.triggering.parsers import (
    BreakPointSectionRaw,
    ConfigFilesSection,
    ContextManagersSectionRaw,
    DotEnvSection,
    EOIOSectionRaw,
    EOOutputRaw,
    SecretSection,
    WorkflowCallbacksSectionRaw,
    WorkflowRaw,
    WorkflowSectionRaw,
    build_context_managers,
    build_io,
    build_output_product,
    build_workflow_callbacks,
    build_workflow_unit,
    load_section,
    normalize_legacy_dask_context_section,
    validate_triggering_payload_structure,
)
from eopf.triggering.payload_loader import load_triggering_payload_file, prepare_triggering_payload
from eopf.triggering.runner import ParsersResults
from eopf.triggering.workflow import EOProcessorWorkFlow, WorkFlowUnitDescription
from eopf.triggering.workflow_callbacks import (
    WORKFLOW_EVENT_AFTER_OUTPUT_WRITE_FLUSH,
    WORKFLOW_EVENT_AFTER_UNIT,
    WORKFLOW_EVENT_BEFORE_OUTPUT_WRITE_FLUSH,
    WORKFLOW_EVENT_BEFORE_UNIT,
    WORKFLOW_EVENT_ON_UNIT_ERROR,
    WORKFLOW_EVENT_SETUP,
    WORKFLOW_EVENT_TEARDOWN,
    WorkflowCallbackContext,
)
from eopf.triggering.workflow_input import WorkflowInputManager
from eopf.triggering.workflow_output import WorkflowOutputManager
from eopf.triggering.workflow_progress import WorkflowProgressLogger
from eopf.triggering.workflow_types import (
    OutputWriteStatus,
    WorkflowExecutionReport,
    WorkflowUnitReport,
    WrittenProductReport,
)
from eopf.triggering.workflow_writer import ProductWriter

WORKFLOW_CALLBACK_INVOCATIONS: list[tuple[str, str | None, str | None]] = []
CONTEXT_MANAGER_EVENTS: list[str] = []


class TriggeringTestProductConvention:
    pass


SAMPLE_OL_1_EFR_ZIP = (
    "S3A_OL_1_EFR____20231011T123858_20231011T124158_20231011T143747_0179_104_209_2160_PS1_O_NR_003.SEN3"
)


@pytest.mark.unit
def test_resolve_storage_options_formats_secret_alias_for_s3_path():
    with mock.patch("eopf.triggering.general_utils.SecretsManager") as secrets_manager:
        secrets_manager.return_value.secrets.return_value = {
            "key": "my-key",
            "secret": "my-secret",
            "client_kwargs": {
                "endpoint_url": "https://s3.example.com",
                "region_name": "eu-west-1",
            },
        }

        result = resolve_storage_options({"secret_alias": "common"}, "s3://bucket/data")

    assert result == {
        "storage_options": {
            "key": "my-key",
            "secret": "my-secret",
            "client_kwargs": {
                "endpoint_url": "https://s3.example.com",
                "region_name": "eu-west-1",
            },
        },
    }


@pytest.mark.unit
def test_resolve_storage_options_keeps_non_s3_secret_alias_raw():
    with mock.patch("eopf.triggering.general_utils.SecretsManager") as secrets_manager:
        secrets_manager.return_value.secrets.return_value = {
            "token": "abc123",
            "username": "user",
        }

        result = resolve_storage_options({"secret_alias": "common"}, "/tmp/data")

    assert result == {
        "storage_options": {
            "token": "abc123",
            "username": "user",
        },
    }


def _mock_processing_unit(
    identifier: str,
    *,
    mandatory_inputs: list[str] | None = None,
) -> mock.Mock:
    processing_unit = mock.Mock()
    processing_unit.identifier = identifier
    processing_unit.processor_contract.return_value = None
    processing_unit.get_mandatory_input_list.return_value = mandatory_inputs or []
    processing_unit.get_mandatory_adf_list.return_value = []
    processing_unit.get_processing_complexity.return_value = 1.0
    return processing_unit


class _NoQCProcessor:
    """Minimal stand-in for ``EOQCProcessor`` used when ``apply_eoqc=False``."""

    def __init__(self) -> None:
        self.called = False

    def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - defensive
        self.called = True
        raise AssertionError("EOQC processor should not be called when apply_eoqc=False")


def _workflow_unit(
    processing_unit: mock.Mock,
    *,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
    step: int = 0,
    allowed_to_fail: bool = False,
) -> WorkFlowUnitDescription:
    return WorkFlowUnitDescription(
        active=True,
        mode="",
        processing_unit=processing_unit,
        inputs=inputs or {},
        adfs={},
        outputs=outputs or {},
        parameters={},
        step=step,
        validate=True,
        allowed_to_fail=allowed_to_fail,
    )


class _RecordingProductWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def write(
        self,
        eo_data: DataTree,
        *,
        output_id: str,
        output_path: str,
        engine: str | None = None,
        writer_params: dict[str, Any] | None = None,
        reopen_params: dict[str, Any] | None = None,
        **_: Any,
    ) -> OutputWriteStatus:
        self.calls.append((output_id, eo_data.name))
        return OutputWriteStatus(
            writer_params=writer_params or {},
            reopen_params=reopen_params,
            product_path=output_path,
            product_name=eo_data.name,
            product_id=output_id,
            engine=engine,
            job=None,
            write_started_at=0.0,
            write_is_delayed=False,
        )


def _output_io(output_id: str) -> EOOutputProductParserResult:
    return EOOutputProductParserResult(
        id=output_id,
        path=f"memory://{output_id}",
        path_type=PathType.File,
        apply_eoqc=False,
    )


def _output_manager(product_writer: _RecordingProductWriter) -> WorkflowOutputManager:
    plan = mock.Mock()
    plan.punits_indices = {"unit": 0}
    plan.processing_units_weak_graph.is_leaf.return_value = True
    return WorkflowOutputManager(
        plan=plan,
        product_writer=product_writer,
        product_opener=mock.Mock(),
        initial_products={},
    )


def _named_products(*names: str):
    for name in names:
        yield DataTree(name=name)


def _set_output_generator_consumption(mode: str) -> str:
    previous = EOConfiguration().get("triggering__output_generator_consumption", "round_robin")
    EOConfiguration()["triggering__output_generator_consumption"] = mode
    return previous


@pytest.mark.unit
def test_multiple_output_generators_are_consumed_in_output_order():
    previous_mode = _set_output_generator_consumption("output_order")
    product_writer = _RecordingProductWriter()
    manager = _output_manager(product_writer)

    try:
        manager.handle_unit_outputs(
            _NoQCProcessor(),
            {"out_a": _output_io("out_a"), "out_b": _output_io("out_b"), "out_c": _output_io("out_c")},
            _workflow_unit(_mock_processing_unit("unit"), outputs={"a": "out_a", "b": "out_b", "c": "out_c"}),
            {
                "a": _named_products("a1", "a2", "a3"),
                "b": _named_products("b1", "b2"),
                "c": _named_products("c1"),
            },
        )
    finally:
        EOConfiguration()["triggering__output_generator_consumption"] = previous_mode

    assert product_writer.calls == [
        ("out_a", "a1"),
        ("out_b", "b1"),
        ("out_c", "c1"),
        ("out_a", "a2"),
        ("out_b", "b2"),
        ("out_a", "a3"),
    ]


@pytest.mark.unit
def test_exhausted_output_generator_is_removed_from_output_order():
    previous_mode = _set_output_generator_consumption("output_order")
    product_writer = _RecordingProductWriter()
    manager = _output_manager(product_writer)

    try:
        manager.handle_unit_outputs(
            _NoQCProcessor(),
            {"out_a": _output_io("out_a"), "out_b": _output_io("out_b")},
            _workflow_unit(_mock_processing_unit("unit"), outputs={"a": "out_a", "b": "out_b"}),
            {
                "a": _named_products("a1", "a2", "a3"),
                "b": _named_products("b1"),
            },
        )
    finally:
        EOConfiguration()["triggering__output_generator_consumption"] = previous_mode

    assert product_writer.calls == [("out_a", "a1"), ("out_b", "b1"), ("out_a", "a2"), ("out_a", "a3")]


@pytest.mark.unit
def test_output_generators_can_be_consumed_exhaustively():
    previous_mode = _set_output_generator_consumption("exhaustive")
    product_writer = _RecordingProductWriter()
    manager = _output_manager(product_writer)

    try:
        manager.handle_unit_outputs(
            _NoQCProcessor(),
            {"out_a": _output_io("out_a"), "out_b": _output_io("out_b")},
            _workflow_unit(_mock_processing_unit("unit"), outputs={"a": "out_a", "b": "out_b"}),
            {
                "a": _named_products("a1", "a2"),
                "b": _named_products("b1", "b2"),
            },
        )
    finally:
        EOConfiguration()["triggering__output_generator_consumption"] = previous_mode

    assert product_writer.calls == [("out_a", "a1"), ("out_a", "a2"), ("out_b", "b1"), ("out_b", "b2")]


@pytest.mark.unit
def test_regular_outputs_are_written_before_generator_output_order():
    previous_mode = _set_output_generator_consumption("output_order")
    product_writer = _RecordingProductWriter()
    manager = _output_manager(product_writer)

    try:
        manager.handle_unit_outputs(
            _NoQCProcessor(),
            {"regular": _output_io("regular"), "stream": _output_io("stream")},
            _workflow_unit(_mock_processing_unit("unit"), outputs={"regular": "regular", "stream": "stream"}),
            {
                "regular": DataTree(name="regular_product"),
                "stream": _named_products("stream1", "stream2"),
            },
        )
    finally:
        EOConfiguration()["triggering__output_generator_consumption"] = previous_mode

    assert product_writer.calls == [
        ("regular", "regular_product"),
        ("stream", "stream1"),
        ("stream", "stream2"),
    ]


@pytest.mark.unit
def test_output_generators_are_not_materialized_before_output_order_progress():
    previous_mode = _set_output_generator_consumption("output_order")
    product_writer = _RecordingProductWriter()
    manager = _output_manager(product_writer)
    second_stream_consumed = False

    def first_stream():
        yield DataTree(name="a1")
        if not second_stream_consumed:
            raise AssertionError("first stream consumed twice before second stream progressed")
        yield DataTree(name="a2")

    def second_stream():
        nonlocal second_stream_consumed
        second_stream_consumed = True
        yield DataTree(name="b1")

    try:
        manager.handle_unit_outputs(
            _NoQCProcessor(),
            {"out_a": _output_io("out_a"), "out_b": _output_io("out_b")},
            _workflow_unit(_mock_processing_unit("unit"), outputs={"a": "out_a", "b": "out_b"}),
            {"a": first_stream(), "b": second_stream()},
        )
    finally:
        EOConfiguration()["triggering__output_generator_consumption"] = previous_mode

    assert product_writer.calls == [("out_a", "a1"), ("out_b", "b1"), ("out_a", "a2")]


@pytest.mark.unit
def test_single_output_generator_is_still_consumed_fully():
    product_writer = _RecordingProductWriter()
    manager = _output_manager(product_writer)

    manager.handle_unit_outputs(
        _NoQCProcessor(),
        {"stream": _output_io("stream")},
        _workflow_unit(_mock_processing_unit("unit"), outputs={"stream": "stream"}),
        {"stream": _named_products("stream1", "stream2")},
    )

    assert product_writer.calls == [("stream", "stream1"), ("stream", "stream2")]


@pytest.mark.unit
def test_output_generator_error_includes_output_name():
    product_writer = _RecordingProductWriter()
    manager = _output_manager(product_writer)

    def broken_generator():
        yield DataTree(name="stream1")
        raise RuntimeError("boom")

    with pytest.raises(TriggerInvalidWorkflow, match="stream"):
        manager.handle_unit_outputs(
            _NoQCProcessor(),
            {"stream": _output_io("stream")},
            _workflow_unit(_mock_processing_unit("unit"), outputs={"stream": "stream"}),
            {"stream": broken_generator()},
        )

    assert product_writer.calls == [("stream", "stream1")]


@pytest.mark.unit
def test_cache_output_uses_configured_write_and_read_params(monkeypatch: pytest.MonkeyPatch):
    class FakeTempFolder:
        def get_uuid_subfolder(self):
            return self

        def get_url_and_params(self):
            return "memory://cache-product", {"storage_options": {"token": "cache"}}

    monkeypatch.setattr("eopf.triggering.workflow_output.EOTemporaryFolder", FakeTempFolder)
    EOConfiguration()["triggering__create_temporary"] = True
    EOConfiguration()["triggering__cache_writer_params"] = {"mode": "w", "compute": False}
    EOConfiguration()["triggering__cache_reader_params"] = {"consolidated": False}

    try:
        output_io_param, output_id = WorkflowOutputManager._cache_output_io_param(
            "prepared",
            _workflow_unit(_mock_processing_unit("prepare")),
        )
    finally:
        EOConfiguration()["triggering__cache_writer_params"] = {}
        EOConfiguration()["triggering__cache_reader_params"] = {}

    assert output_id == "prepare.prepared"
    assert output_io_param.engine == "cpm_zarr"
    assert output_io_param.writer_params == {
        "mode": "w",
        "compute": False,
        "storage_options": {"token": "cache"},
    }
    assert output_io_param.reopen_params == {
        "consolidated": False,
        "storage_options": {"token": "cache"},
    }


@pytest.mark.unit
def test_stored_output_reopens_with_reader_auto_discovery():
    product_opener = mock.Mock(return_value=DataTree())
    manager = WorkflowOutputManager(
        plan=mock.Mock(),
        product_writer=mock.Mock(),
        product_opener=product_opener,
        initial_products={},
    )
    manager.available_store_instances["outputs.output"] = [
        OutputWriteStatus(
            writer_params={},
            reopen_params=None,
            product_path="output.zarr",
            product_name="output",
            product_id="output",
            engine="cpm_zarr",
            job=None,
            write_started_at=0.0,
            write_is_delayed=False,
        ),
    ]

    manager.resolve_product_input("output")

    product_opener.assert_called_once_with(
        reader_params={},
        engine=None,
        path="output.zarr",
        product_id="output",
    )


@pytest.mark.unit
def test_product_writer_stages_s3_zarr_output_locally(monkeypatch: pytest.MonkeyPatch, caplog):
    caplog.set_level(logging.INFO, logger="eopf.triggering.workflow_writer")
    EOConfiguration()["triggering__stage_s3_outputs"] = True
    product = DataTree(name="product")
    product["measurements"] = DataTree(
        name="measurements",
        dataset=xr.Dataset({"measurement": ("x", [1])}, attrs={"group": "measurements"}),
    )
    write_calls: list[dict[str, Any]] = []
    uploads: list[dict[str, Any]] = []

    def recording_write_datatree(dtree, filename_or_obj, engine=None, **kwargs):
        write_calls.append(
            {
                "dtree": dtree,
                "filename_or_obj": filename_or_obj,
                "engine": engine,
                "kwargs": kwargs,
            },
        )
        Path(filename_or_obj).mkdir(parents=True, exist_ok=True)
        return None

    def fake_upload_staged_product(*, local_product_path, product_path, writer_params):
        assert Path(local_product_path).exists()
        uploads.append(
            {
                "local_product_path": local_product_path,
                "product_path": product_path,
                "writer_params": writer_params,
            },
        )

    monkeypatch.setattr("eopf.write_datatree", recording_write_datatree)
    monkeypatch.setattr(ProductWriter, "_upload_staged_product", staticmethod(fake_upload_staged_product))
    monkeypatch.setattr(ProductWriter, "_has_remote_distributed_client", staticmethod(lambda: True))
    monkeypatch.setattr(
        "eopf.store.staging.psutil.virtual_memory",
        lambda: types.SimpleNamespace(available=1),
    )

    try:
        status = ProductWriter().write(
            product,
            eoqc_processor=_NoQCProcessor(),
            engine="cpm_zarr",
            writer_params={"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
            reopen_params=None,
            output_id="output",
            output_type=PathType.File,
            output_path="s3://bucket/output.zarr",
            apply_eoqc=False,
        )
    finally:
        EOConfiguration()["triggering__stage_s3_outputs"] = False

    assert status.product_path == "s3://bucket/output.zarr"
    assert status.write_is_delayed is False
    assert len(write_calls) == 1
    assert write_calls[0]["filename_or_obj"] != status.product_path
    assert write_calls[0]["engine"] == "cpm_zarr_subdatatree"
    assert write_calls[0]["kwargs"]["compute"] is True
    assert "write_by_subdatatree" not in write_calls[0]["kwargs"]
    assert "storage_options" not in write_calls[0]["kwargs"]
    assert uploads == [
        {
            "local_product_path": write_calls[0]["filename_or_obj"],
            "product_path": "s3://bucket/output.zarr",
            "writer_params": {"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
        },
    ]
    assert "Wrote staged S3 output s3://bucket/output.zarr locally at " in caplog.text


@pytest.mark.unit
def test_product_writer_logs_experimental_s3_staging_warning(monkeypatch: pytest.MonkeyPatch, caplog):
    caplog.set_level(logging.WARNING, logger="eopf.triggering.workflow_writer")

    monkeypatch.setattr("eopf.write_datatree", lambda **_: None)
    monkeypatch.setattr(ProductWriter, "_upload_staged_product", staticmethod(lambda **_: None))
    monkeypatch.setattr(ProductWriter, "_has_remote_distributed_client", staticmethod(lambda: False))

    ProductWriter()._write_staged_s3_product(
        eo_data=DataTree(name="product"),
        engine="cpm_zarr",
        writer_params={"mode": "w", "storage_options": {"token": "secret"}},
        product_path="s3://bucket/output.zarr",
    )

    assert "Experimental feature: S3 output staging writes s3://bucket/output.zarr" in caplog.text
    assert "Behavior and failure recovery may change" in caplog.text


@pytest.mark.unit
def test_product_writer_staged_s3_output_chains_delayed_upload(monkeypatch: pytest.MonkeyPatch):
    EOConfiguration()["triggering__stage_s3_outputs"] = True
    product = DataTree(name="product")
    product["measurements"] = DataTree(
        name="measurements",
        dataset=xr.Dataset({"measurement": ("x", [1])}),
    )
    write_calls: list[dict[str, Any]] = []
    uploads: list[dict[str, Any]] = []

    from eopf import write_datatree as real_write_datatree

    def recording_write_datatree(dtree, filename_or_obj, engine=None, **kwargs):
        write_calls.append(
            {
                "filename_or_obj": filename_or_obj,
                "kwargs": kwargs,
            },
        )
        return real_write_datatree(dtree, filename_or_obj, engine=engine, **kwargs)

    def fake_upload_staged_product(*, local_product_path, product_path, writer_params):
        assert Path(local_product_path).exists()
        uploads.append(
            {
                "local_product_path": local_product_path,
                "product_path": product_path,
                "writer_params": writer_params,
            },
        )

    monkeypatch.setattr("eopf.write_datatree", recording_write_datatree)
    monkeypatch.setattr(ProductWriter, "_upload_staged_product", staticmethod(fake_upload_staged_product))
    monkeypatch.setattr(ProductWriter, "_has_remote_distributed_client", staticmethod(lambda: False))

    try:
        status = ProductWriter().write(
            product,
            eoqc_processor=_NoQCProcessor(),
            engine="cpm_zarr",
            writer_params={"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
            reopen_params=None,
            output_id="output",
            output_type=PathType.File,
            output_path="s3://bucket/output.zarr",
            apply_eoqc=False,
        )
    finally:
        EOConfiguration()["triggering__stage_s3_outputs"] = False

    assert status.write_is_delayed is True
    assert write_calls[0]["kwargs"]["compute"] is False
    assert "storage_options" not in write_calls[0]["kwargs"]
    assert uploads == [
        {
            "local_product_path": write_calls[0]["filename_or_obj"],
            "product_path": "s3://bucket/output.zarr",
            "writer_params": {"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
        },
    ]


@pytest.mark.unit
def test_product_writer_staged_s3_upload_uses_configured_copy_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog,
):
    caplog.set_level(logging.INFO, logger="eopf.triggering.workflow_writer")
    local_product_path = tmp_path / "output.zarr"
    local_product_path.mkdir()
    copy_calls: list[dict[str, Any]] = []

    def fake_copy_to(self, target, *, max_workers=None, **copy_kwargs):
        copy_calls.append(
            {
                "source": self.fs_path,
                "target": target.fs_path,
                "max_workers": max_workers,
                "copy_kwargs": copy_kwargs,
            },
        )

    monkeypatch.setattr(AnyPath, "copy_to", fake_copy_to)
    monkeypatch.setattr(ProductWriter, "_prepare_staged_target", staticmethod(lambda *, target, mode: None))
    EOConfiguration()["triggering__stage_s3_upload_workers"] = 16
    EOConfiguration()["triggering__stage_s3_upload_options"] = {"block_size": 64 * 1024 * 1024}

    try:
        ProductWriter._upload_staged_product(
            local_product_path=str(local_product_path),
            product_path="s3://bucket/output.zarr",
            writer_params={"mode": "w", "storage_options": {"anon": True}},
        )
    finally:
        EOConfiguration()["triggering__stage_s3_upload_workers"] = 8
        EOConfiguration()["triggering__stage_s3_upload_options"] = {}

    assert copy_calls == [
        {
            "source": str(local_product_path),
            "target": "bucket/output.zarr",
            "max_workers": 16,
            "copy_kwargs": {"block_size": 64 * 1024 * 1024},
        },
    ]
    assert f"Uploaded staged S3 output s3://bucket/output.zarr from {local_product_path} in " in caplog.text


@pytest.mark.unit
def test_product_writer_staged_s3_upload_can_use_temporary_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    local_product_path = tmp_path / "output.zarr"
    local_product_path.mkdir()
    copy_calls: list[dict[str, str]] = []
    prepared_targets: list[dict[str, str]] = []
    cleaned_targets: list[str] = []

    def fake_copy_to(self, target, *, max_workers=None, **copy_kwargs):
        copy_calls.append({"source": self.fs_path, "target": target.fs_path})

    def fake_prepare_target(*, target, mode):
        prepared_targets.append({"target": target.fs_path, "mode": mode})

    monkeypatch.setattr(AnyPath, "copy_to", fake_copy_to)
    monkeypatch.setattr("eopf.triggering.workflow_writer.uuid.uuid4", lambda: "test")
    monkeypatch.setattr(
        ProductWriter,
        "_check_staged_target_can_be_replaced",
        staticmethod(lambda *, target, mode: None),
    )
    monkeypatch.setattr(ProductWriter, "_prepare_staged_target", staticmethod(fake_prepare_target))
    monkeypatch.setattr(
        ProductWriter,
        "_cleanup_staged_s3_temporary_prefix",
        staticmethod(lambda target: cleaned_targets.append(target.fs_path)),
    )
    EOConfiguration()["triggering__stage_s3_temporary_prefix"] = True

    try:
        ProductWriter._upload_staged_product(
            local_product_path=str(local_product_path),
            product_path="s3://bucket/output.zarr",
            writer_params={"mode": "w", "storage_options": {"anon": True}},
        )
    finally:
        EOConfiguration()["triggering__stage_s3_temporary_prefix"] = False

    assert copy_calls == [
        {"source": str(local_product_path), "target": "bucket/output.zarr.tmp-test"},
        {"source": "bucket/output.zarr.tmp-test", "target": "bucket/output.zarr"},
    ]
    assert prepared_targets == [{"target": "bucket/output.zarr", "mode": "w"}]
    assert cleaned_targets == ["bucket/output.zarr.tmp-test"]


@pytest.mark.unit
def test_product_writer_staged_s3_temporary_prefix_is_cleaned_after_upload_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    local_product_path = tmp_path / "output.zarr"
    local_product_path.mkdir()
    cleaned_targets: list[str] = []
    prepared_targets: list[str] = []

    def failing_copy_to(self, target, *, max_workers=None, **copy_kwargs):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(AnyPath, "copy_to", failing_copy_to)
    monkeypatch.setattr("eopf.triggering.workflow_writer.uuid.uuid4", lambda: "test")
    monkeypatch.setattr(
        ProductWriter,
        "_check_staged_target_can_be_replaced",
        staticmethod(lambda *, target, mode: None),
    )
    monkeypatch.setattr(
        ProductWriter,
        "_prepare_staged_target",
        staticmethod(lambda *, target, mode: prepared_targets.append(target.fs_path)),
    )
    monkeypatch.setattr(
        ProductWriter,
        "_cleanup_staged_s3_temporary_prefix",
        staticmethod(lambda target: cleaned_targets.append(target.fs_path)),
    )
    EOConfiguration()["triggering__stage_s3_temporary_prefix"] = True

    try:
        with pytest.raises(RuntimeError, match="upload failed"):
            ProductWriter._upload_staged_product(
                local_product_path=str(local_product_path),
                product_path="s3://bucket/output.zarr",
                writer_params={"mode": "w", "storage_options": {"anon": True}},
            )
    finally:
        EOConfiguration()["triggering__stage_s3_temporary_prefix"] = False

    assert prepared_targets == []
    assert cleaned_targets == ["bucket/output.zarr.tmp-test"]


@pytest.mark.unit
def test_product_writer_skips_s3_staging_below_minimum_size(monkeypatch: pytest.MonkeyPatch):
    EOConfiguration()["triggering__stage_s3_outputs"] = True
    EOConfiguration()["triggering__stage_s3_outputs_min_size"] = 1_000
    product = DataTree(name="product")
    product["measurements"] = DataTree(
        name="measurements",
        dataset=xr.Dataset({"measurement": ("x", [1])}),
    )
    write_calls: list[dict[str, Any]] = []

    def fake_write_datatree(dtree, filename_or_obj, engine=None, **kwargs):
        write_calls.append(
            {
                "filename_or_obj": filename_or_obj,
                "kwargs": kwargs,
            },
        )

    monkeypatch.setattr("eopf.write_datatree", fake_write_datatree)

    try:
        status = ProductWriter().write(
            product,
            eoqc_processor=_NoQCProcessor(),
            engine="cpm_zarr",
            writer_params={"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
            reopen_params=None,
            output_id="output",
            output_type=PathType.File,
            output_path="s3://bucket/output.zarr",
            apply_eoqc=False,
        )
    finally:
        EOConfiguration()["triggering__stage_s3_outputs"] = False
        EOConfiguration()["triggering__stage_s3_outputs_min_size"] = 0

    assert status.product_path == "s3://bucket/output.zarr"
    assert write_calls == [
        {
            "filename_or_obj": "s3://bucket/output.zarr",
            "kwargs": {"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
        },
    ]


@pytest.mark.unit
def test_product_writer_skips_s3_staging_for_iterable_output(monkeypatch: pytest.MonkeyPatch):
    EOConfiguration()["triggering__stage_s3_outputs"] = True
    product = DataTree(name="product")
    product["measurements"] = DataTree(
        name="measurements",
        dataset=xr.Dataset({"measurement": ("x", [1])}),
    )
    product_stream = (item for item in [product])
    write_calls: list[dict[str, Any]] = []

    def fake_write_datatree(dtree, filename_or_obj, engine=None, **kwargs):
        write_calls.append(
            {
                "dtree": dtree,
                "filename_or_obj": filename_or_obj,
                "kwargs": kwargs,
            },
        )

    monkeypatch.setattr("eopf.write_datatree", fake_write_datatree)

    try:
        status = ProductWriter().write(
            product_stream,
            eoqc_processor=_NoQCProcessor(),
            engine="cpm_zarr",
            writer_params={"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
            reopen_params=None,
            output_id="output",
            output_type=PathType.File,
            output_path="s3://bucket/output.zarr",
            apply_eoqc=False,
        )
    finally:
        EOConfiguration()["triggering__stage_s3_outputs"] = False

    assert status.product_path == "s3://bucket/output.zarr"
    assert write_calls == [
        {
            "dtree": product_stream,
            "filename_or_obj": "s3://bucket/output.zarr",
            "kwargs": {"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
        },
    ]


@pytest.mark.unit
def test_product_writer_staged_s3_remote_zarr_uses_normal_writer_when_product_fits_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    product = DataTree(name="product")
    product["measurements"] = DataTree(
        name="measurements",
        dataset=xr.Dataset({"measurement": ("x", list(range(4)))}),
    )
    write_calls: list[dict[str, Any]] = []

    def recording_write_datatree(dtree, filename_or_obj, engine=None, **kwargs):
        write_calls.append({"dtree": dtree, "filename_or_obj": filename_or_obj, "engine": engine, "kwargs": kwargs})
        Path(filename_or_obj).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("eopf.write_datatree", recording_write_datatree)
    monkeypatch.setattr(ProductWriter, "_upload_staged_product", staticmethod(lambda **_: None))
    monkeypatch.setattr(ProductWriter, "_has_remote_distributed_client", staticmethod(lambda: True))
    monkeypatch.setattr(
        "eopf.store.staging.psutil.virtual_memory",
        lambda: types.SimpleNamespace(available=100),
    )

    EOConfiguration()["triggering__stage_s3_memory_fit_safety_ratio"] = 0.5
    try:
        ProductWriter()._write_staged_s3_product(
            eo_data=product,
            engine="cpm_zarr",
            writer_params={"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
            product_path="s3://bucket/output.zarr",
        )
    finally:
        EOConfiguration()["triggering__stage_s3_memory_fit_safety_ratio"] = 0.25

    assert write_calls[0]["engine"] == "cpm_zarr"
    assert write_calls[0]["kwargs"]["compute"] is True


@pytest.mark.unit
def test_product_writer_staged_s3_remote_netcdf_rejects_large_product_before_compute(
    monkeypatch: pytest.MonkeyPatch,
):
    product = DataTree(name="product")
    product["measurements"] = DataTree(
        name="measurements",
        dataset=xr.Dataset({"measurement": ("x", list(range(4)))}),
    )

    monkeypatch.setattr("eopf.write_datatree", lambda **_: pytest.fail("large NetCDF should fail before writing"))
    monkeypatch.setattr(ProductWriter, "_upload_staged_product", staticmethod(lambda **_: None))
    monkeypatch.setattr(ProductWriter, "_has_remote_distributed_client", staticmethod(lambda: True))
    monkeypatch.setattr(
        "eopf.store.staging.psutil.virtual_memory",
        lambda: types.SimpleNamespace(available=1),
    )

    with pytest.raises(ValueError, match="outputs without a sub-DataTree writer"):
        ProductWriter()._write_staged_s3_product(
            eo_data=product,
            engine="cpm_netcdf",
            writer_params={"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
            product_path="s3://bucket/output.nc",
        )


@pytest.mark.unit
def test_product_writer_staged_s3_remote_fast_zarr_uses_subdatatree_when_product_exceeds_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    product = DataTree(name="product")
    product["measurements"] = DataTree(
        name="measurements",
        dataset=xr.Dataset({"measurement": ("x", list(range(4)))}),
    )
    write_calls: list[dict[str, Any]] = []

    def recording_write_datatree(dtree, filename_or_obj, engine=None, **kwargs):
        write_calls.append({"dtree": dtree, "filename_or_obj": filename_or_obj, "engine": engine, "kwargs": kwargs})
        Path(filename_or_obj).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("eopf.write_datatree", recording_write_datatree)
    monkeypatch.setattr(ProductWriter, "_upload_staged_product", staticmethod(lambda **_: None))
    monkeypatch.setattr(ProductWriter, "_has_remote_distributed_client", staticmethod(lambda: True))
    monkeypatch.setattr(
        "eopf.store.staging.psutil.virtual_memory",
        lambda: types.SimpleNamespace(available=1),
    )

    ProductWriter()._write_staged_s3_product(
        eo_data=product,
        engine="fast_zarr",
        writer_params={"mode": "w", "compute": False, "storage_options": {"token": "secret"}},
        product_path="s3://bucket/output.zarr",
    )

    assert write_calls[0] == {
        "dtree": product,
        "filename_or_obj": write_calls[0]["filename_or_obj"],
        "engine": "fast_zarr_subdatatree",
        "kwargs": {"mode": "w", "compute": True},
    }


@pytest.mark.unit
def test_open_product_uses_reader_params_engine_override(monkeypatch: pytest.MonkeyPatch):
    opened = DataTree()
    open_datatree = mock.Mock(return_value=opened)
    monkeypatch.setattr("eopf.open_datatree", open_datatree)
    reader_params = {"engine": "cpm_zarr", "storage_options": {"token": "cache"}}

    product = WorkflowInputManager.open_product(
        reader_params=reader_params,
        engine=None,
        path="output.zarr",
        product_id="output",
    )

    assert product is opened
    assert product.name == "output"
    assert reader_params == {"engine": "cpm_zarr", "storage_options": {"token": "cache"}}
    open_datatree.assert_called_once_with(
        engine="cpm_zarr",
        filename_or_obj="output.zarr",
        storage_options={"token": "cache"},
    )


class RecordingContextManager:
    def __init__(self, marker: str = "default") -> None:
        self.marker = marker

    def __enter__(self) -> "RecordingContextManager":
        CONTEXT_MANAGER_EVENTS.append(f"enter:{self.marker}")
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        CONTEXT_MANAGER_EVENTS.append(f"exit:{self.marker}")


def record_workflow_callback(context: WorkflowCallbackContext, marker: str | None = None) -> None:
    """Record callback invocation details for parser tests."""
    unit_identifier = context.unit.identifier if context.unit is not None else None
    WORKFLOW_CALLBACK_INVOCATIONS.append((context.event, unit_identifier, marker))


@pytest.mark.unit
def test_triggering_loads_default_configuration_by_default():
    EOConfiguration().clear_loaded_configurations()

    EORunner._load_eo_configuration({})

    assert str(EOPF_CPM_DEFAULT_CONFIG_FILE) in EOConfiguration()._param_file_list
    assert EOConfiguration().general__description == "An example configuration file with complex structures."


@pytest.mark.unit
def test_triggering_can_skip_default_configuration():
    EOConfiguration().clear_loaded_configurations()

    EORunner._load_eo_configuration({"general_configuration": {"triggering__load_default_configuration": False}})

    assert str(EOPF_CPM_DEFAULT_CONFIG_FILE) not in EOConfiguration()._param_file_list
    assert not EOConfiguration().has_value("general__description")


@pytest.mark.unit
def test_triggering_default_configuration_does_not_override_payload_values():
    EOConfiguration().clear_loaded_configurations()

    EORunner._load_eo_configuration(
        {
            "general_configuration": {
                "triggering__load_default_configuration": True,
                "general__description": "from payload",
            },
        },
    )

    assert str(EOPF_CPM_DEFAULT_CONFIG_FILE) in EOConfiguration()._param_file_list
    assert EOConfiguration().general__description == "from payload"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "expected_id"),
    [
        ({"id": "payload-run"}, "payload-run"),
        ({"general_configuration": {"triggering__id": "configured-run"}}, "configured-run"),
        (
            {
                "id": "payload-run",
                "general_configuration": {"triggering__id": "configured-run"},
            },
            "configured-run",
        ),
    ],
)
def test_triggering_loads_workflow_id(payload, expected_id):
    EOConfiguration().clear_loaded_configurations()

    EORunner._load_eo_configuration(payload)

    assert EOConfiguration().triggering__id == expected_id


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        {
            "general_configuration": {
                "triggering__use_basic_logging": True,
                "triggering__load_default_logging": True,
            },
        },
        {
            "general_configuration": {
                "triggering__use_basic_logging": True,
            },
            "logging": ["logging.json"],
        },
    ],
)
def test_basic_logging_is_exclusive_with_logging_configuration(payload):
    with pytest.raises(TriggeringConfigurationError):
        EORunner().extract_from_payload_and_init_conf_logging(payload)


@pytest.mark.unit
def test_basic_logging_overrides_implicit_handler_from_default_configuration(monkeypatch):
    original_load_default_file = EOConfiguration.load_default_file

    def load_default_file_with_early_log(self, *args, **kwargs):
        logging.warning("early log before basic logging setup")
        return original_load_default_file(self, *args, **kwargs)

    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(EOConfiguration, "load_default_file", load_default_file_with_early_log)

    EORunner(clean=True).extract_from_payload_and_init_conf_logging(
        {
            "general_configuration": {
                "logging": {"level": "DEBUG"},
                "triggering__use_basic_logging": True,
            },
            "io": {},
            "workflow": [
                {
                    "name": "unit",
                    "module": "tests.ut.computing.test_dummy",
                    "processing_unit": "TestDummyProcessor",
                },
            ],
        },
    )

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) == 1
    root_handler = root_logger.handlers[0]
    assert isinstance(root_handler, logging.StreamHandler)
    assert root_handler.formatter is not None
    assert root_handler.formatter._fmt == DEFAULT_LOG_FORMAT

    logging.getLogger("third_party_module").info("third party info emitted")
    root_handler.flush()
    assert "third party info emitted" in stderr.getvalue()


@pytest.mark.unit
def test_triggering_custom_logging_installs_password_filter(tmp_path):
    reset_logging()
    EOConfiguration().clear_loaded_configurations()
    log_file = tmp_path / "triggering.log"
    logging_config = tmp_path / "logging.json"
    logging_config.write_text(
        json.dumps(
            {
                "version": 1,
                "handlers": {
                    "file": {
                        "class": "logging.FileHandler",
                        "filename": str(log_file),
                    },
                },
                "loggers": {
                    "eopf.test.triggering.mask": {
                        "handlers": ["file"],
                        "level": "INFO",
                        "propagate": False,
                    },
                },
            },
        ),
    )

    EORunner(clean=False)._setup_logging([logging_config])
    logging.getLogger("eopf.test.triggering.mask").info("token=clear-secret")

    assert "token=****" in log_file.read_text()
    assert "clear-secret" not in log_file.read_text()
    reset_logging()
    EOConfiguration().clear_loaded_configurations()


@pytest.mark.unit
def test_triggering_password_filter_can_be_disabled(tmp_path):
    reset_logging()
    EOConfiguration().clear_loaded_configurations()
    EOConfiguration().load_dict({"triggering__logging_offuscate": False})
    log_file = tmp_path / "triggering.log"
    logging_config = tmp_path / "logging.json"
    logging_config.write_text(
        json.dumps(
            {
                "version": 1,
                "handlers": {
                    "file": {
                        "class": "logging.FileHandler",
                        "filename": str(log_file),
                    },
                },
                "root": {
                    "handlers": ["file"],
                    "level": "INFO",
                },
            },
        ),
    )

    EORunner(clean=False)._setup_logging([logging_config])
    logging.getLogger().info("token=clear-secret")

    assert "token=clear-secret" in log_file.read_text()
    reset_logging()
    EOConfiguration().clear_loaded_configurations()


@pytest.mark.unit
def test_clean_runner_clears_configuration_and_logging():
    EOConfiguration().load_dict({"test__clean_runner_marker": "kept"})
    test_logger = logging.getLogger("eopf.test.clean_runner")
    handler = logging.NullHandler()
    test_logger.addHandler(handler)

    with pytest.raises(TriggeringConfigurationError):
        EORunner(clean=True).extract_from_payload_and_init_conf_logging(
            {
                "general_configuration": {
                    "triggering__use_basic_logging": True,
                    "triggering__load_default_logging": True,
                },
            },
        )

    assert not EOConfiguration().has_value("test__clean_runner_marker")
    assert handler not in test_logger.handlers
    EOConfiguration().clear_loaded_configurations()


@pytest.mark.unit
def test_non_clean_runner_keeps_configuration_and_logging():
    reset_logging()
    EOConfiguration().clear_loaded_configurations()
    EOConfiguration().load_dict({"test__non_clean_runner_marker": "kept"})
    test_logger = logging.getLogger("eopf.test.non_clean_runner")
    handler = logging.NullHandler()
    test_logger.addHandler(handler)

    with pytest.raises(TriggeringConfigurationError):
        EORunner(clean=False).extract_from_payload_and_init_conf_logging(
            {
                "general_configuration": {
                    "triggering__use_basic_logging": True,
                    "triggering__load_default_logging": True,
                },
            },
        )

    assert EOConfiguration().get("test__non_clean_runner_marker") == "kept"
    assert handler in test_logger.handlers

    test_logger.removeHandler(handler)
    EOConfiguration().clear_loaded_configurations()
    reset_logging()


@pytest.mark.need_files
@pytest.mark.unit
@pytest.mark.dask_only
def test_extract_payload_template(TRIGGER_YAML_TEMPLATE, TEST_DATA_FAKE_SECRET, monkeypatch):
    from eopf.dask_utils.dask_context_manager import DaskContext

    monkeypatch.setenv("S3_OUTPUT_TEST_DATA_PATH", "...")
    monkeypatch.setenv("SECRET_PASS", "...")
    payload = file_utils.load_yaml_file(TRIGGER_YAML_TEMPLATE)
    parser_results = EORunner().extract_from_payload_and_init_conf_logging(
        payload,
    )
    payload = resolve_env_vars(payload)
    if isinstance(payload["workflow"], list):
        units_classes, parameters = zip(
            *[
                (
                    resolve_env_vars(unit["processing_unit"]),
                    resolve_env_vars(unit.get("parameters", {})),
                )
                for unit in payload["workflow"]
            ],
        )
        print(units_classes)
        print(parameters)
        assert all(
            (unit.processing_unit.__class__.__name__ in units_classes)
            for unit in parser_results.processing_workflow.plan.workflow
        )

        for unit in parser_results.processing_workflow.plan.workflow:
            for unit_payload in payload["workflow"]:
                if unit_payload["name"] == unit.identifier:
                    print(unit_payload["parameters"])
                    print(unit.parameters)
                    assert all(
                        key in unit.parameters and unit.parameters[key] == value
                        for key, value in unit_payload["parameters"].items()
                    )

    else:
        assert parser_results.processing_workflow.plan.workflow[0].processing_unit.__class__.__name__ == payload[
            "workflow"
        ].get(
            "processing_unit",
        )
        assert parser_results.processing_workflow.plan.workflow[0].parameters == payload["workflow"].get(
            "parameters",
            {},
        )

    inputs_products_data = payload["io"].get("input_products")
    inputs_products = parser_results.io_config.input_products
    for input_product_data in inputs_products_data:
        product = inputs_products[input_product_data["id"]]
        if product.engine is not None:
            assert EOReaderRegistry.contains(product.engine)

        assert product.path == input_product_data.get("path")
        assert product.reader_params == resolve_storage_options(
            input_product_data.get("reader_params", {}),
            input_product_data.get("path"),
        )

    output_product_data = payload["io"].get("output_products")

    print(parser_results.io_config.output_products)

    assert EOWriterRegistry.contains(output_product_data[0].get("engine"))
    assert parser_results.io_config.output_products[
        output_product_data[0]["id"]
    ].writer_params == resolve_storage_options(
        payload["io"]
        .get("output_products", {})[0]
        .get(
            "writer_params",
            {},
        ),
        output_product_data[0].get("path"),
    )

    # external import test
    empty_stores = EOReaderRegistry.get_by_name("fakereader")
    print(empty_stores)

    assert len(parser_results.context_managers) == 1
    assert isinstance(parser_results.context_managers[0], DaskContext)


@pytest.mark.need_files
@pytest.mark.unit
@pytest.mark.dask_only
def test_trigger_regex(TRIGGER_YAML_REGEX_FILE_FILLED):
    data = run_runner(TRIGGER_YAML_REGEX_FILE_FILLED)

    output_breakpoing = AnyPath.cast(data["breakpoints"]["folder"])
    assert output_breakpoing.exists()
    breakpoints_folder = output_breakpoing.ls()
    assert len(breakpoints_folder) != 0
    for brpf in breakpoints_folder:
        prod_brkp = brpf.ls()
        assert len(prod_brkp) != 0
        for pf in prod_brkp:
            loaded_prod = open_datatree(filename_or_obj=pf.get_url_and_params()[0])
            assert "measurements" in loaded_prod
            pf.rm(recursive=True)
    EOTemporaryFolder.clear(gc_collect=True)


@pytest.mark.unit
def test_triggering_regex_input_uses_regex_matching(tmp_path):
    (tmp_path / "S01SIWSLC_20260408T174632.zarr").mkdir()
    (tmp_path / "S01SIWSLC_20260408T174632.txt").touch()
    (tmp_path / "S02MSIL1C_20260408T174632.zarr").touch()

    results = WorkflowInputManager._get_input_product_path_list(
        str(tmp_path),
        PathType.Regex,
        "slcs",
        {},
        r"S01.*\.zarr",
        None,
    )

    assert set(results) == {str(tmp_path / "S01SIWSLC_20260408T174632.zarr")}


@pytest.mark.unit
@pytest.mark.real_s3
def test_triggering_regex_input_uses_regex_matching_on_s3(
    s3_test_data,
    s3_config_real,
):
    full_path = os.path.join(f"{s3_test_data[0]}://{s3_test_data[1]}", SAMPLE_OL_1_EFR_ZIP)

    results = WorkflowInputManager._get_input_product_path_list(
        full_path,
        PathType.Regex,
        "olci",
        {"storage_options": s3_config_real},
        r".*/Oa12_.*\.nc",
        None,
    )

    assert len(results) == 2
    expected_prefix = f"{s3_test_data[0]}://{s3_test_data[1]}/{SAMPLE_OL_1_EFR_ZIP}/"
    assert all(result.startswith(expected_prefix) for result in results)
    assert all(result.endswith(".nc") for result in results)


@pytest.mark.need_files
@pytest.mark.unit
@pytest.mark.dask_only
def test_trigger_temporary(TRIGGER_YAML_FILE_FILLED):
    EOTemporaryFolder.clear(gc_collect=True)
    with open(TRIGGER_YAML_FILE_FILLED) as f:
        data = yaml.safe_load(f)
    data["general_configuration"]["triggering__create_temporary"] = "true"
    data = run_runner_data(data)
    # Temporary should be cleared at the end
    assert EOTemporaryFolder not in EOTemporaryFolder._instances


@pytest.mark.unit
@pytest.mark.real_s3
@pytest.mark.dask_only
def test_trigger_temporary_shared_s3(OUTPUT_DIR, TRIGGER_YAML_FILE_FILLED, s3_output_config_real, s3_output_test_data):
    test_data_secret = {
        "test_data": s3_output_config_real,
        "secret_bindings": {f"{s3_output_test_data[1]}": "test_data"},
    }
    with (AnyPath.cast(OUTPUT_DIR) / "test_data_secret.json").open("w") as f:
        json.dump(test_data_secret, f)

    with open(TRIGGER_YAML_FILE_FILLED) as f:
        data = yaml.safe_load(f)
    data["secret"] = [os.path.join(OUTPUT_DIR, "test_data_secret.json")]
    data["general_configuration"]["triggering__create_temporary"] = "true"
    data["general_configuration"]["triggering__temporary_shared"] = "true"
    # data["general_configuration"]["temporary__folder_s3_secret"] = "test_data"
    data["general_configuration"]["temporary__folder"] = (
        f"{s3_output_test_data[0]}://{s3_output_test_data[1]}/test_s3_temp_dir"
    )

    data = run_runner_data(data)
    assert EOTemporaryFolder not in EOTemporaryFolder._instances


@pytest.mark.unit
@pytest.mark.dask_only
@pytest.mark.parametrize(
    "TRIGGER_YAML_HEAVY_FILE_FILLED",
    [{"size_x": 1000, "size_y": 500, "chunk_x": 500, "chunk_y": 500, "nb_var": 1}],
    indirect=True,
)
def test_trigger_heavy_configures_dask_logging(TRIGGER_YAML_HEAVY_FILE_FILLED):
    """
    Ensure heavy trigger tests forward worker logs and set the Dask logger level.
    """
    with open(TRIGGER_YAML_HEAVY_FILE_FILLED) as f:
        yaml_data = yaml.safe_load(f)

    dask_context_params = yaml_data["context_managers"][0]["parameters"]
    assert dask_context_params["forward_worker_logger"] == "distributed.worker"
    assert dask_context_params["forward_worker_logger_level"] == logging.INFO
    assert yaml_data["general_configuration"]["logging__dask_level"] == "DEBUG"
    assert {os.path.basename(path) for path in yaml_data["logging"]} == {"eopf.json", "dask.yaml"}


@pytest.mark.unit
@pytest.mark.dask_only
@pytest.mark.parametrize(
    "TRIGGER_YAML_HEAVY_FILE_FILLED",
    [{"size_x": 500, "size_y": 500, "chunk_x": 500, "chunk_y": 500, "nb_var": 2}],
    indirect=True,
)
@pytest.mark.parametrize("full_image_ops", [False, True], ids=["threshold_ops", "full_image_ops"])
def test_trigger_heavy(TRIGGER_YAML_HEAVY_FILE_FILLED, full_image_ops):
    """
    Run the heavy trigger with both lightweight and full image-operation processor modes.
    """
    with open(TRIGGER_YAML_HEAVY_FILE_FILLED) as f:
        yaml_data = yaml.safe_load(f)
    _set_heavy_full_image_ops(yaml_data, full_image_ops=full_image_ops)

    runner = EORunner()
    runner.run(yaml_data)
    output_product = AnyPath.cast(yaml_data["io"]["output_products"][0]["path"])
    _assert_heavy_output_product(output_product)
    EOTemporaryFolder.clear(gc_collect=True)


@pytest.mark.unit
@pytest.mark.real_s3
@pytest.mark.dask_only
@pytest.mark.parametrize(
    "TRIGGER_YAML_HEAVY_FILE_FILLED",
    ["trigger-heavy-s3.yaml"],
    indirect=True,
)
@pytest.mark.parametrize(
    "heavy_s3_writer_engine",
    ["cpm_zarr", "fast_zarr"],
    ids=["cpm_zarr", "fast_zarr"],
)
@pytest.mark.parametrize(
    "stage_s3_outputs",
    [False, True],
    ids=["direct_s3_write", "staged_s3_write"],
)
def test_trigger_heavy_real_s3_temporary_folder(
    TRIGGER_YAML_HEAVY_FILE_FILLED,
    TEST_DATA_SECRET,
    s3_output_config_real,
    s3_output_test_data,
    heavy_s3_writer_engine,
    stage_s3_outputs,
):
    """
    Run the heavy trigger with real S3 outputs and many nested Zarr groups.
    """
    with open(TRIGGER_YAML_HEAVY_FILE_FILLED) as f:
        yaml_data = yaml.safe_load(f)

    s3_protocol, s3_base_path = s3_output_test_data
    s3_run_folder = f"{s3_protocol}://{s3_base_path}/{uuid.uuid4()}/trigger-heavy"
    s3_temp_folder = f"{s3_run_folder}/tmp"
    _set_heavy_s3_paths(yaml_data, s3_run_folder=s3_run_folder, storage_options=s3_output_config_real)
    _set_heavy_s3_writer_engine(yaml_data, engine=heavy_s3_writer_engine)
    _set_heavy_s3_logging_config(yaml_data)
    yaml_data["secret"] = [str(TEST_DATA_SECRET)]
    yaml_data["general_configuration"]["triggering__create_temporary"] = True
    yaml_data["general_configuration"]["triggering__stage_s3_outputs"] = stage_s3_outputs
    yaml_data["general_configuration"]["temporary__folder"] = s3_temp_folder
    yaml_data["general_configuration"]["temporary__folder_s3_secret"] = "test_data"

    runner = EORunner()
    try:
        runner.run(yaml_data)
        output_product = AnyPath.cast(
            yaml_data["io"]["output_products"][0]["path"],
            **s3_output_config_real,
        )
        _assert_heavy_output_product(output_product)
    finally:
        s3_run_path = AnyPath.cast(s3_run_folder, **s3_output_config_real)
        if s3_run_path.exists():
            s3_run_path.rm(recursive=True)


def _set_heavy_full_image_ops(yaml_data: dict[str, Any], *, full_image_ops: bool) -> None:
    """Configure the heavy processor image-operation mode in a trigger payload."""
    yaml_data["workflow"][1].setdefault("parameters", {})["full_image_ops"] = full_image_ops


def _set_heavy_s3_paths(
    yaml_data: dict[str, Any],
    *,
    s3_run_folder: str,
    storage_options: dict[str, Any],
) -> None:
    """Rewrite heavy trigger output paths to write under one S3 run folder."""
    output_products = yaml_data["io"]["output_products"]
    if len(output_products) > 1:
        output_products[0]["path"] = f"{s3_run_folder}/output.zarr"
        output_products[1]["path"] = f"{s3_run_folder}/finals/"
    elif output_products[0]["type"] == "folder":
        output_products[0]["path"] = f"{s3_run_folder}/finals/"
    else:
        output_products[0]["path"] = f"{s3_run_folder}/{os.path.basename(output_products[0]['path'])}"
    for output_product in output_products:
        output_product.setdefault("writer_params", {})["storage_options"] = storage_options


def _set_heavy_s3_writer_engine(yaml_data: dict[str, Any], *, engine: str) -> None:
    """Configure the writer engine used by heavy S3 output products."""
    for output_product in yaml_data["io"]["output_products"]:
        output_product["engine"] = engine


def _set_heavy_s3_logging_config(yaml_data: dict[str, Any]) -> None:
    """Use non-propagating EOPF logging for verbose heavy S3 runs."""
    if not yaml_data.get("logging"):
        return
    yaml_data["logging"][0] = str(Path(__file__).parents[1] / "data" / "triggering" / "eopf-no-propagate.json")


def _assert_heavy_output_product(output_product: AnyPath) -> None:
    """Validate the primary heavy trigger output product."""
    assert output_product.exists()
    loaded_prod = open_datatree(
        filename_or_obj=output_product.get_url_and_params()[0],
        **output_product.get_url_and_params()[1],
    )
    assert "measurements" in loaded_prod


def run_runner(yaml_file):
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    return run_runner_data(data)


def run_runner_data(yaml_data):
    runner = EORunner()
    with mock.patch(
        "tests.ut.computing.test_abstract.TestAbstractProcessor.get_mandatory_input_list",
    ) as mmand:
        mmand.return_value = ["in1"]
        runner.run(yaml_data)

    output_product = Path(yaml_data["io"]["output_products"][0]["path"])
    assert output_product.exists()
    loaded_prod = open_datatree(filename_or_obj=str(output_product))
    assert "measurements" in loaded_prod
    # output_product.rm(recursive=True)
    outputs_product = AnyPath.cast(yaml_data["io"]["output_products"][1]["path"])
    assert outputs_product.exists()
    assert len(outputs_product.glob("multi*")) == 2
    outputs_product.rm(recursive=True)

    return yaml_data


@pytest.mark.unit
def test_validate_triggering_payload_structure_has_no_runtime_side_effects():
    payload = {
        "io": {
            "input_products": [
                {
                    "id": "input",
                    "path": "missing-input.zarr",
                    "engine": "missing-reader",
                },
            ],
            "output_products": [
                {
                    "id": "output",
                    "path": "missing-output.zarr",
                    "engine": "missing-writer",
                },
            ],
        },
        "config": ["/missing/config.yaml"],
        "secret": ["/missing/secret.yaml"],
        "dotenv": ["/missing/.env"],
        "logging": ["/missing/logging.yaml"],
        "context_managers": [
            {
                "module": "eopf.dask_utils.dask_context_manager",
                "context_manager": "DaskContext",
                "parameters": {"cluster_type": "local"},
            },
        ],
        "external_modules": [{"name": "missing_module"}],
        "product_convention": {
            "module": "tests.ut.triggering.test_triggering",
            "convention": "TriggeringTestProductConvention",
        },
        "workflow": [
            {
                "name": "unit",
                "module": "missing_module",
                "processing_unit": "MissingProcessor",
                "inputs": {"input": "input"},
                "outputs": {"output": "output"},
            },
        ],
    }
    original_payload = dict(payload)

    structure = validate_triggering_payload_structure(payload)

    assert payload == original_payload
    assert structure.io.io.input_products[0].engine == "missing-reader"
    assert structure.product_convention.product_convention.convention == "TriggeringTestProductConvention"
    assert structure.workflow.workflow[0].module == "missing_module"


@pytest.mark.unit
def test_triggering_configures_product_convention_before_default_lookup(monkeypatch):
    from eopf.product import product_convention_provider
    from eopf.product.product_stac_convention import StacProductConvention

    monkeypatch.setattr(product_convention_provider, "_product_convention", StacProductConvention)
    monkeypatch.setattr(product_convention_provider, "_product_convention_locked", True)

    def assert_convention_configured(_payload):
        assert product_convention_provider._product_convention is TriggeringTestProductConvention
        assert product_convention_provider._product_convention_locked is False
        raise RuntimeError("stop after convention configuration")

    monkeypatch.setattr(EORunner, "_load_eo_configuration", staticmethod(assert_convention_configured))

    with pytest.raises(RuntimeError, match="stop after convention configuration"):
        EORunner(clean=True).extract_from_payload_and_init_conf_logging(
            {
                "product_convention": {
                    "module": "tests.ut.triggering.test_triggering",
                    "convention": "TriggeringTestProductConvention",
                },
            },
        )


@pytest.mark.unit
def test_validate_triggering_payload_structure_rejects_missing_io():
    with pytest.raises(Exception, match="io"):
        validate_triggering_payload_structure({"workflow": []})


@pytest.mark.unit
def test_build_output_product_validates_fast_zarr_mode_before_write():
    with pytest.raises(
        TriggeringConfigurationError,
        match="Invalid writer configuration for output output: fast_zarr only supports mode",
    ):
        build_output_product(
            EOOutputRaw(
                id="output",
                path="output.zarr",
                engine="fast_zarr",
                writer_params={"mode": "a"},
            ),
        )


@pytest.mark.unit
def test_build_output_product_validates_generic_writer_mode_before_write():
    with pytest.raises(
        TriggeringConfigurationError,
        match="Invalid writer configuration for output output: bad not allowed by store EODataTreeZarrWriter",
    ):
        build_output_product(
            EOOutputRaw(
                id="output",
                path="output.zarr",
                engine="cpm_zarr",
                writer_params={"mode": "bad"},
            ),
        )


@pytest.mark.unit
def test_build_output_product_validates_fast_zarr_unsupported_kwargs_before_write():
    with pytest.raises(
        TriggeringConfigurationError,
        match="Invalid writer configuration for output output: fast_zarr does not support these xarray Zarr keyword "
        "arguments: append_dim",
    ):
        build_output_product(
            EOOutputRaw(
                id="output",
                path="output.zarr",
                engine="fast_zarr",
                writer_params={"append_dim": "time"},
            ),
        )


@pytest.mark.unit
def test_build_output_product_validates_subdatatree_s3_before_write():
    with pytest.raises(
        TriggeringConfigurationError,
        match="Invalid writer configuration for output output: sub-DataTree writers only support local filesystem",
    ):
        build_output_product(
            EOOutputRaw(
                id="output",
                path="s3://bucket/output.zarr",
                engine="cpm_zarr_subdatatree",
            ),
        )


@pytest.mark.unit
def test_load_triggering_payload_file_resolves_section_includes_relative_to_declaring_file(tmp_path):
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    nested_dir = payload_dir / "nested"
    nested_dir.mkdir()
    io_file = nested_dir / "io.yaml"

    (payload_dir / "trigger.yaml").write_text(
        """
workflow: nested/workflow.yaml
io: nested/io.yaml
context_managers: nested/context-managers.yaml
""",
        encoding="utf-8",
    )
    (nested_dir / "workflow.yaml").write_text(
        """
- name: unit
  module: my.module
  processing_unit: MyUnit
""",
        encoding="utf-8",
    )
    io_file.write_text(
        """
input_products: []
output_products: []
""",
        encoding="utf-8",
    )
    (nested_dir / "context-managers.yaml").write_text("[]\n", encoding="utf-8")

    payload = load_triggering_payload_file(payload_dir / "trigger.yaml")

    assert payload["workflow"][0]["name"] == "unit"
    assert payload["io"] == {"input_products": [], "output_products": []}
    assert payload["context_managers"] == []


@pytest.mark.unit
def test_load_triggering_payload_file_rejects_non_mapping_payload(tmp_path):
    payload_file = tmp_path / "payload.yaml"
    payload_file.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TriggerInvalidWorkflow, match="must contain a YAML mapping"):
        load_triggering_payload_file(payload_file)


@pytest.mark.unit
def test_load_triggering_payload_file_resolves_env_vars_in_include_paths(tmp_path, monkeypatch):
    workflow_file = tmp_path / "workflow.yaml"
    workflow_file.write_text("[]\n", encoding="utf-8")
    payload_file = tmp_path / "payload.yaml"
    payload_file.write_text(
        "workflow: $WORKFLOW_FILE\nio: {input_products: [], output_products: []}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WORKFLOW_FILE", str(workflow_file))

    payload = load_triggering_payload_file(payload_file)

    assert payload["workflow"] == []


@pytest.mark.unit
def test_prepare_triggering_payload_loads_dotenv_before_resolving_payload(tmp_path, monkeypatch):
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("TRIGGER_INPUT=input-from-dotenv.zarr\n", encoding="utf-8")
    monkeypatch.delenv("TRIGGER_INPUT", raising=False)

    payload = {
        "dotenv": [str(dotenv_file)],
        "io": {
            "input_products": [{"id": "input", "path": "$TRIGGER_INPUT"}],
            "output_products": [],
        },
    }

    resolved_payload = prepare_triggering_payload(payload)

    assert resolved_payload["io"]["input_products"][0]["path"] == "input-from-dotenv.zarr"


@pytest.mark.unit
def test_validate_triggering_payload_structure_accepts_legacy_dask_context_with_warning():
    payload = {
        "io": {"input_products": [], "output_products": []},
        "workflow": [],
        "dask_context": {"cluster_type": "local"},
    }
    original_payload = dict(payload)

    with pytest.warns(DeprecationWarning, match="top-level dask_context payload section is deprecated"):
        structure = validate_triggering_payload_structure(payload)

    assert payload == original_payload
    assert len(structure.context_managers.context_managers) == 1
    context_manager = structure.context_managers.context_managers[0]
    assert context_manager.module == "eopf.dask_utils.dask_context_manager"
    assert context_manager.context_manager == "DaskContext"
    assert context_manager.parameters == {"cluster_type": "local"}


@pytest.mark.unit
def test_legacy_dask_context_does_not_override_explicit_dask_context():
    payload = {
        "dask_context": {"cluster_type": "local"},
        "context_managers": [
            {
                "module": "eopf.dask_utils.dask_context_manager",
                "context_manager": "DaskContext",
                "parameters": {"cluster_type": "gateway"},
            },
        ],
    }

    with pytest.warns(DeprecationWarning, match="top-level dask_context payload section is deprecated"):
        normalized_payload = normalize_legacy_dask_context_section(payload)

    assert normalized_payload["context_managers"] == payload["context_managers"]
    assert "dask_context" not in normalized_payload


@pytest.mark.need_files
@pytest.mark.unit
@pytest.mark.dask_only
def test_trigger_bad_opening_mode(TRIGGER_YAML_BAD_OPEN_FILE_FILLED):
    runner = CliRunner()
    r = runner.invoke(trigger_cli, args=" ".join(["local", TRIGGER_YAML_BAD_OPEN_FILE_FILLED]))
    if r.exception is not None:
        traceback.print_exception(type(r.exception), r.exception, r.exception.__traceback__)
    print(r.exception)
    assert r.exception is not None
    assert r.exit_code == 1
    EOTemporaryFolder.clear(gc_collect=True)


@pytest.mark.need_files
@pytest.mark.unit
@pytest.mark.dask_only
def test_trigger_not_accepted_opening_mode(TRIGGER_YAML_NOT_ACCEPTED_OPEN_FILE_FILLED):
    runner = CliRunner()
    r = runner.invoke(trigger_cli, args=" ".join(["local", TRIGGER_YAML_NOT_ACCEPTED_OPEN_FILE_FILLED]))
    if r.exception is not None:
        traceback.print_exception(type(r.exception), r.exception, r.exception.__traceback__)
    print(r.exception)
    assert r.exception is not None
    assert "No registered writer" in r.output
    assert r.exit_code == 1
    EOTemporaryFolder.clear(gc_collect=True)


@pytest.mark.need_files
@pytest.mark.unit
@pytest.mark.dask_only
def test_new_parser(TRIGGER_YAML_TEMPLATE, TEST_DATA_FAKE_SECRET, monkeypatch):

    monkeypatch.setenv("S3_OUTPUT_TEST_DATA_PATH", "...")
    monkeypatch.setenv("SECRET_PASS", "...")
    payload = file_utils.load_yaml_file(TRIGGER_YAML_TEMPLATE)
    if "I/O" in payload:
        warn("Use io instead of I/O", DeprecationWarning)
        payload["io"] = payload.pop("I/O")
    pprint(payload)
    parser_results: ParsersResults = EORunner().extract_from_payload_and_init_conf_logging(
        payload,
    )
    assert parser_results is not None
    payload = resolve_env_vars(payload)
    workflow_units = load_section(payload, WorkflowSectionRaw)
    assert workflow_units is not None
    breakpoints = load_section(payload, BreakPointSectionRaw)
    assert breakpoints is not None
    config_files = load_section(payload, ConfigFilesSection)
    assert config_files is not None
    env_vars = load_section(payload, DotEnvSection)
    assert env_vars is not None
    secrets = load_section(payload, SecretSection)
    assert secrets is not None
    io = load_section(payload, EOIOSectionRaw).io
    assert io is not None


@pytest.mark.unit
def test_build_io_rejects_duplicate_adf_ids():
    raw_io = load_section(
        {
            "io": {
                "adfs": [
                    {"id": "duplicate", "path": "first"},
                    {"id": "duplicate", "path": "second"},
                ],
            },
        },
        EOIOSectionRaw,
    ).io

    with pytest.raises(TriggeringConfigurationError, match="Duplicated id.*io.adfs.*duplicate"):
        build_io(raw_io)


@pytest.mark.unit
def test_workflow_callback_section_builds_external_callbacks():
    raw_callbacks = load_section(
        {
            "workflow_callbacks": [
                {
                    "module": "tests.ut.triggering.test_triggering",
                    "function": "record_workflow_callback",
                    "parameters": {"marker": "from-payload"},
                },
            ],
        },
        WorkflowCallbacksSectionRaw,
    ).workflow_callbacks

    callbacks = build_workflow_callbacks(raw_callbacks)
    WORKFLOW_CALLBACK_INVOCATIONS.clear()
    callbacks[0](
        WorkflowCallbackContext(
            event=WORKFLOW_EVENT_BEFORE_UNIT,
            plan=mock.Mock(),
        ),
    )

    assert WORKFLOW_CALLBACK_INVOCATIONS == [("before_unit", None, "from-payload")]


@pytest.mark.unit
def test_workflow_run_emits_setup_teardown_callbacks_and_progress(caplog):
    events: list[tuple[str, dict[str, Any]]] = []

    def record_event(context: WorkflowCallbackContext) -> None:
        events.append((context.event, context.metadata))

    caplog.set_level(logging.INFO, logger="eopf.triggering.workflow")
    processing_unit = mock.Mock()
    processing_unit.identifier = "unit"
    processing_unit.processor_contract.return_value = None
    processing_unit.get_mandatory_input_list.return_value = []
    processing_unit.get_mandatory_adf_list.return_value = []
    processing_unit.run.return_value = {"out": DataTree(name="output")}
    processing_unit = _mock_processing_unit("unit")
    processing_unit.run.return_value = {}
    workflow = EOProcessorWorkFlow(
        workflow_units=[_workflow_unit(processing_unit)],
        callbacks=[record_event],
    )

    workflow._run(
        inputs={},
        output_io_products={},
        eoqc_processor=_NoQCProcessor(),
        validate=False,
        validation_mode=ValidationMode.STRUCTURE,
        available_adfs={},
    )

    assert [event for event, _ in events] == [
        WORKFLOW_EVENT_SETUP,
        WORKFLOW_EVENT_BEFORE_UNIT,
        WORKFLOW_EVENT_AFTER_UNIT,
        WORKFLOW_EVENT_TEARDOWN,
    ]
    assert events[0][1] == {"input_ids": [], "adf_ids": [], "output_ids": [], "workflow_id": ""}
    assert events[3][1]["error_count"] == 0
    assert "Workflow progress: 90% - unit 1/1 run call completed: unit in " in caplog.text
    assert "Workflow progress: 90% - unit 1/1 outputs handled: unit in " in caplog.text
    assert "Workflow progress: 90% - flushing pending output write jobs" in caplog.text
    assert "Workflow progress: 100% - workflow completed" in caplog.text


@pytest.mark.unit
def test_workflow_progress_logger_uses_configured_progress_level(caplog):
    caplog.set_level(PROGRESS_LOG_LEVEL, logger="eopf.triggering.workflow_progress")
    EOConfiguration()["logging__progress_level"] = "PROGRESS"
    try:
        WorkflowProgressLogger(total_units=0).start()
    finally:
        EOConfiguration()["logging__progress_level"] = "INFO"

    assert caplog.records[-1].levelno == PROGRESS_LOG_LEVEL
    assert caplog.records[-1].levelname == "PROGRESS"
    assert caplog.records[-1].message == "Workflow progress: 0% - starting workflow"


@pytest.mark.unit
def test_workflow_progress_logger_can_log_percentage_only(caplog):
    caplog.set_level(logging.INFO, logger="eopf.triggering.workflow_progress")
    EOConfiguration()["logging__progress_format"] = "PERCENTAGE"
    try:
        WorkflowProgressLogger(total_units=0).start()
    finally:
        EOConfiguration()["logging__progress_format"] = "FULL"

    assert caplog.records[-1].message == "0%"


@pytest.mark.unit
def test_workflow_progress_logger_uses_processing_unit_complexity(caplog):
    caplog.set_level(logging.INFO, logger="eopf.triggering.workflow_progress")
    light_unit = _mock_processing_unit("light")
    heavy_unit = _mock_processing_unit("heavy")
    light_unit.get_processing_complexity.return_value = 1.0
    heavy_unit.get_processing_complexity.return_value = 3.0

    light_description = _workflow_unit(light_unit)
    heavy_description = _workflow_unit(heavy_unit)
    progress_logger = WorkflowProgressLogger.from_workflow_units([light_description, heavy_description])

    progress_logger.unit_executed(0, light_description, 0.0)
    progress_logger.unit_outputs_handled(0, light_description, 0.0)
    progress_logger.unit_started(1, heavy_description)

    assert "Workflow progress: 22% - unit 1/2 run call completed: light in 0.00s" in caplog.text
    assert "Workflow progress: 22% - unit 1/2 outputs handled: light in 0.00s" in caplog.text
    assert "Workflow progress: 22% - running unit 2/2: heavy" in caplog.text


@pytest.mark.unit
def test_workflow_progress_logger_weights_lazy_unit_by_declared_outputs(caplog):
    caplog.set_level(logging.INFO, logger="eopf.triggering.workflow_progress")
    lazy_unit = _mock_processing_unit("lazy")
    lazy_unit.get_processing_complexity.return_value = 0.0
    lazy_description = _workflow_unit(lazy_unit, outputs={"intermediate": "cache"})
    progress_logger = WorkflowProgressLogger.from_workflow_units([lazy_description])

    progress_logger.unit_executed(0, lazy_description, 0.0)
    progress_logger.unit_outputs_handled(0, lazy_description, 0.0)
    progress_logger.flushing_outputs()

    assert "Workflow progress: 0% - unit 1/1 run call completed: lazy in 0.00s" in caplog.text
    assert "Workflow progress: 90% - unit 1/1 outputs handled: lazy in 0.00s" in caplog.text
    assert "Workflow progress: 90% - flushing pending output write jobs" in caplog.text


@pytest.mark.unit
def test_log_progress_updates_status_and_logs(caplog):
    reset_execution_status()
    caplog.set_level(logging.INFO)
    EOConfiguration()["logging__progress_format"] = "PERCENTAGE"
    try:
        progress_percent = log_progress(42, "processor step")
    finally:
        EOConfiguration()["logging__progress_format"] = "FULL"
        reset_execution_status()

    assert progress_percent == 42
    assert caplog.records[-1].message == "42%"


@pytest.mark.unit
def test_triggering_local_command_uses_status_exit_code(tmp_path, monkeypatch):
    payload_file = tmp_path / "payload.yaml"
    payload_file.write_text("{}\n", encoding="utf-8")

    def run_from_file(self: EORunner, payload_file: str, working_dir: str | None = None) -> None:
        reset_execution_status()
        set_execution_exit_code(7)

    monkeypatch.setattr(EORunner, "run_from_file", run_from_file)

    result = CliRunner().invoke(trigger_cli, args=["local", str(payload_file)])

    assert result.exit_code == 7
    reset_execution_status()


@pytest.mark.unit
def test_triggering_processing_unit_can_set_exit_code(tmp_path, monkeypatch):
    payload_file = tmp_path / "payload.yaml"
    payload_file.write_text("{}\n", encoding="utf-8")

    processing_unit = _mock_processing_unit("status_unit")

    def run_unit(inputs: MappingDataType, adfs: MappingAuxiliary, **kwargs: Any) -> dict[str, DataTree]:
        set_execution_exit_code(7)
        return {}

    processing_unit.run.side_effect = run_unit
    processing_unit.run_validating.side_effect = run_unit
    parser_results = ParsersResults(
        breakpoints=None,
        general_config={},
        processing_workflow=EOProcessorWorkFlow(workflow_units=[_workflow_unit(processing_unit)]),
        io_config=EOIOParserResult(output_products={}, input_products={}, adfs={}),
        dask_config=None,
        context_managers=[],
        logging_config=[],
        config=[],
        secret_files=[],
        eoqc_config=None,
    )

    def extract_from_payload_and_init_conf_logging(self: EORunner, payload: dict[str, Any]) -> ParsersResults:
        return parser_results

    EOConfiguration()["triggering__create_temporary"] = False
    monkeypatch.setattr(
        EORunner, "extract_from_payload_and_init_conf_logging", extract_from_payload_and_init_conf_logging,
    )
    try:
        result = CliRunner().invoke(trigger_cli, args=["local", str(payload_file)])
    finally:
        EOConfiguration()["triggering__create_temporary"] = True
        reset_execution_status()

    assert result.exit_code == 7
    processing_unit.run_validating.assert_called_once()


@pytest.mark.unit
def test_workflow_teardown_callback_runs_after_pending_output_flush():
    events: list[str] = []

    def record_event(context: WorkflowCallbackContext) -> None:
        events.append(context.event)

    processing_unit = mock.Mock()
    processing_unit.identifier = "unit"
    processing_unit.processor_contract.return_value = None
    processing_unit.get_mandatory_input_list.return_value = []
    processing_unit.get_mandatory_adf_list.return_value = []
    processing_unit.run.return_value = {"out": DataTree(name="output")}
    product_writer = mock.Mock()
    product_writer.write.return_value = OutputWriteStatus(
        writer_params={},
        reopen_params=None,
        product_path="output.zarr",
        product_name="output",
        product_id="OUTPUT",
        engine="cpm_zarr",
        job=ImmediateJob(),
        write_started_at=0.0,
        write_is_delayed=True,
    )
    workflow = EOProcessorWorkFlow(
        workflow_units=[
            WorkFlowUnitDescription(
                active=True,
                mode="",
                processing_unit=processing_unit,
                inputs={},
                adfs={},
                outputs={"out": "OUTPUT"},
                parameters={},
                step=0,
                validate=True,
            ),
        ],
        callbacks=[record_event],
    )
    workflow._product_writer = product_writer

    workflow._run(
        inputs={},
        output_io_products={
            "OUTPUT": mock.Mock(
                id="OUTPUT",
                path="output.zarr",
                path_type=mock.Mock(value="file"),
                engine="cpm_zarr",
                writer_params={},
                reopen_params=None,
                apply_eoqc=False,
            ),
        },
        eoqc_processor=_NoQCProcessor(),
        validate=False,
        validation_mode=ValidationMode.STRUCTURE,
        available_adfs={},
    )

    assert events.index(WORKFLOW_EVENT_BEFORE_OUTPUT_WRITE_FLUSH) < events.index(WORKFLOW_EVENT_TEARDOWN)
    assert events.index(WORKFLOW_EVENT_AFTER_OUTPUT_WRITE_FLUSH) < events.index(WORKFLOW_EVENT_TEARDOWN)


@pytest.mark.unit
def test_workflow_execution_report_collects_timings_and_written_products():
    processing_unit = _mock_processing_unit("unit")
    processing_unit.run.return_value = {"out": DataTree(name="output")}
    product_writer = mock.Mock()
    product_writer.write.return_value = OutputWriteStatus(
        writer_params={},
        reopen_params=None,
        product_path="output.zarr",
        product_name="output",
        product_id="OUTPUT",
        engine="cpm_zarr",
        job=ImmediateJob(),
        write_started_at=0.0,
        write_is_delayed=False,
        write_submit_seconds=0.1,
        write_finished_at=0.1,
    )
    workflow = EOProcessorWorkFlow(
        workflow_units=[_workflow_unit(processing_unit, outputs={"out": "OUTPUT"})],
    )
    workflow._product_writer = product_writer
    report = WorkflowExecutionReport(workflow_id="run-id")

    workflow._run(
        inputs={},
        output_io_products={
            "OUTPUT": EOOutputProductParserResult(
                id="OUTPUT",
                path="output.zarr",
                path_type=PathType.File,
                engine="cpm_zarr",
                writer_params={},
                reopen_params=None,
                apply_eoqc=False,
            ),
        },
        eoqc_processor=_NoQCProcessor(),
        validate=False,
        validation_mode=ValidationMode.STRUCTURE,
        available_adfs={},
        report=report,
    )

    assert report.status == "completed"
    assert report.processing_seconds >= 0.0
    assert report.output_writing_seconds >= 0.0
    assert report.units[0].identifier == "unit"
    assert report.units[0].output_names == ["out"]
    assert report.written_products[0].product_id == "OUTPUT"
    assert report.written_products[0].path == "output.zarr"
    assert report.written_products[0].unit_identifier == "unit"
    assert report.written_products[0].output_name == "out"


@pytest.mark.unit
def test_workflow_execution_report_formats_as_single_json_line():
    report = WorkflowExecutionReport(
        workflow_id="run-id",
        status="completed",
        dry_run=False,
        overall_seconds=9.591,
        input_opening_seconds=3.134,
        processing_seconds=0.024,
        output_writing_seconds=5.895,
        output_flush_seconds=0.264,
        error_count=0,
        input_ids=["L1C"],
        adf_ids=["GIP_CONVERT"],
        output_ids=["output_pvi"],
        units=[
            WorkflowUnitReport(
                identifier="example_processor_1",
                processing_seconds=0.024,
                output_writing_seconds=0.0,
                output_names=["tci"],
            ),
        ],
        written_products=[
            WrittenProductReport(
                product_id="output_pvi",
                path="outputs/S2MSIPVI_20220306T100031_00000_A0122_T626.zarr",
                unit_identifier="example_processor_1",
                output_name="tci",
                engine="cpm_zarr",
                write_total_seconds=5.895,
            ),
        ],
    )

    formatted = report.format()

    assert "\n" not in formatted
    parsed = json.loads(formatted)
    assert parsed["message"] == "Workflow execution report"
    assert parsed["workflow_id"] == "run-id"
    assert parsed["overall"] == 9.59
    assert parsed["processing_units"] == 0.02
    assert parsed["inputs"] == ["L1C"]
    assert parsed["declared_outputs"] == ["output_pvi"]
    assert parsed["units"] == [
        {
            "identifier": "example_processor_1",
            "processing": 0.02,
            "output_writing": 0.0,
            "outputs": ["tci"],
        },
    ]
    assert parsed["written_products"][0]["path"] == "outputs/S2MSIPVI_20220306T100031_00000_A0122_T626.zarr"
    assert parsed["written_products"][0]["write"] == 5.89


@pytest.mark.unit
def test_workflow_run_emits_on_unit_error_callback():
    events: list[tuple[str, dict[str, Any]]] = []

    def record_event(context: WorkflowCallbackContext) -> None:
        events.append((context.event, context.metadata))

    expected_error = TriggerInvalidWorkflow("unit failed")
    processing_unit = _mock_processing_unit("unit")
    processing_unit.run.side_effect = expected_error
    workflow = EOProcessorWorkFlow(
        workflow_units=[_workflow_unit(processing_unit)],
        callbacks=[record_event],
    )

    with pytest.raises(TriggerInvalidWorkflow, match="unit failed"):
        workflow._run(
            inputs={},
            output_io_products={},
            eoqc_processor=_NoQCProcessor(),
            validate=False,
            validation_mode=ValidationMode.STRUCTURE,
            available_adfs={},
        )

    assert [event for event, _ in events] == [
        WORKFLOW_EVENT_SETUP,
        WORKFLOW_EVENT_BEFORE_UNIT,
        WORKFLOW_EVENT_ON_UNIT_ERROR,
        WORKFLOW_EVENT_TEARDOWN,
    ]
    error_metadata = events[2][1]
    assert error_metadata["unit_index"] == 0
    assert error_metadata["exception"] is expected_error
    assert error_metadata["exception_type"] == "TriggerInvalidWorkflow"
    assert error_metadata["exception_message"] == "unit failed"
    assert events[3][1]["error_count"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("interrupt_signalled", "expected_sleep_calls"),
    [(True, [mock.call(0.25)]), (False, [])],
)
def test_workflow_run_cancels_pending_writes_on_keyboard_interrupt(
    caplog,
    interrupt_signalled,
    expected_sleep_calls,
):
    caplog.set_level(logging.WARNING, logger="eopf.triggering.workflow")

    first_unit = _mock_processing_unit("first")
    first_unit.run.return_value = {"result": DataTree()}

    interrupted_unit = _mock_processing_unit("interrupted")
    interrupted_unit.run.side_effect = KeyboardInterrupt

    workflow = EOProcessorWorkFlow(
        workflow_units=[
            _workflow_unit(first_unit, outputs={"result": "stored_output"}),
            _workflow_unit(interrupted_unit, step=1),
        ],
    )
    write_job = mock.Mock()
    workflow._product_writer = mock.Mock()
    workflow._product_writer.write.return_value = OutputWriteStatus(
        writer_params={},
        reopen_params=None,
        product_path="/tmp/output.zarr",
        product_name=None,
        product_id="stored_output",
        job=write_job,
        write_started_at=0.0,
        write_is_delayed=True,
    )

    with mock.patch(
        "eopf.triggering.workflow.signal_computation_interrupt",
        return_value=interrupt_signalled,
    ) as signal_interrupt:
        with mock.patch("eopf.triggering.workflow.time.sleep") as sleep:
            EOConfiguration()["triggering__interrupt_grace_period"] = 0.25
            try:
                with pytest.raises(KeyboardInterrupt):
                    workflow._run(
                        inputs={},
                        output_io_products={
                            "stored_output": EOOutputProductParserResult(
                                id="stored_output",
                                path="/tmp/output.zarr",
                                path_type=PathType.Folder,
                                apply_eoqc=False,
                            ),
                        },
                        eoqc_processor=_NoQCProcessor(),
                        validate=False,
                        validation_mode=ValidationMode.STRUCTURE,
                        available_adfs={},
                    )
            finally:
                EOConfiguration()["triggering__interrupt_grace_period"] = 2.0

    signal_interrupt.assert_called_once_with()
    assert sleep.mock_calls == expected_sleep_calls
    write_job.cancel.assert_called_once_with()
    write_job.result.assert_not_called()
    assert "Workflow interrupted; signalling distributed tasks and cancelling pending output jobs" in caplog.text


@pytest.mark.unit
def test_workflow_allowed_to_fail_unit_continues_without_final_error():
    events: list[tuple[str, dict[str, Any]]] = []

    def record_event(context: WorkflowCallbackContext) -> None:
        events.append((context.event, context.metadata))

    expected_error = RuntimeError("optional unit failed")
    processing_unit = _mock_processing_unit("optional")
    processing_unit.run.side_effect = expected_error
    workflow = EOProcessorWorkFlow(
        workflow_units=[_workflow_unit(processing_unit, allowed_to_fail=True)],
        callbacks=[record_event],
    )

    workflow._run(
        inputs={},
        output_io_products={},
        eoqc_processor=_NoQCProcessor(),
        validate=False,
        validation_mode=ValidationMode.STRUCTURE,
        available_adfs={},
    )

    assert [event for event, _ in events] == [
        WORKFLOW_EVENT_SETUP,
        WORKFLOW_EVENT_BEFORE_UNIT,
        WORKFLOW_EVENT_ON_UNIT_ERROR,
        WORKFLOW_EVENT_TEARDOWN,
    ]
    assert events[2][1]["exception"] is expected_error
    assert events[3][1]["error_count"] == 0


@pytest.mark.unit
def test_workflow_on_unit_error_callback_exception_goes_through_error_policy():
    def raise_callback_error(context: WorkflowCallbackContext) -> None:
        if context.event == WORKFLOW_EVENT_ON_UNIT_ERROR:
            raise RuntimeError("callback failed")

    processing_unit = _mock_processing_unit("optional")
    processing_unit.run.side_effect = RuntimeError("optional unit failed")
    workflow = EOProcessorWorkFlow(
        workflow_units=[_workflow_unit(processing_unit, allowed_to_fail=True)],
        callbacks=[raise_callback_error],
    )

    with pytest.raises(RuntimeError, match="callback failed"):
        workflow._run(
            inputs={},
            output_io_products={},
            eoqc_processor=_NoQCProcessor(),
            validate=False,
            validation_mode=ValidationMode.STRUCTURE,
            available_adfs={},
        )


@pytest.mark.unit
def test_workflow_rejects_dependency_on_allowed_to_fail_unit_output():
    optional_unit = _mock_processing_unit("optional")
    downstream_unit = _mock_processing_unit("downstream", mandatory_inputs=["source"])

    with pytest.raises(TriggerInvalidWorkflow, match="depends on output optional_output from allowed-to-fail unit"):
        EOProcessorWorkFlow(
            workflow_units=[
                _workflow_unit(
                    optional_unit,
                    outputs={"result": "optional_output"},
                    allowed_to_fail=True,
                ),
                _workflow_unit(downstream_unit, inputs={"source": "optional_output"}),
            ],
        )


@pytest.mark.unit
def test_workflow_planner_resolves_dotted_unit_name_dependency():
    calibration_unit = _mock_processing_unit("Calibration.1")
    reference_dem_unit = _mock_processing_unit("ReferenceDEM.2", mandatory_inputs=["slcs"])

    workflow = EOProcessorWorkFlow(
        workflow_units=[
            _workflow_unit(calibration_unit, outputs={"slcs": "slcs"}),
            _workflow_unit(reference_dem_unit, inputs={"slcs": "Calibration.1.slcs"}),
        ],
    )

    calibration_index = workflow.plan.punits_indices["Calibration.1"]
    reference_dem_index = workflow.plan.punits_indices["ReferenceDEM.2"]

    assert calibration_index in workflow.plan.processing_units_graph.graph[reference_dem_index]
    assert not workflow.plan.processing_units_weak_graph.is_leaf(calibration_index)


@pytest.mark.unit
def test_workflow_warns_on_optional_dependency_on_allowed_to_fail_unit_output(caplog):
    caplog.set_level(logging.WARNING, logger="eopf.triggering.workflow")
    optional_unit = _mock_processing_unit("optional")
    downstream_unit = _mock_processing_unit("downstream")

    EOProcessorWorkFlow(
        workflow_units=[
            _workflow_unit(
                optional_unit,
                outputs={"result": "optional_output"},
                allowed_to_fail=True,
            ),
            _workflow_unit(downstream_unit, inputs={"source": "optional_output"}),
        ],
    )

    assert (
        "Workflow unit downstream has optional input source wired to output optional_output "
        "from allowed-to-fail unit optional"
    ) in caplog.text


@pytest.mark.unit
def test_workflow_dry_run_validates_and_logs_without_running_units(caplog):
    caplog.set_level(logging.INFO, logger="eopf.triggering.workflow")
    processing_unit = _mock_processing_unit("unit")
    workflow = EOProcessorWorkFlow(
        workflow_units=[_workflow_unit(processing_unit)],
    )

    EOConfiguration()["triggering__dry_run"] = True
    try:
        workflow.run_workflow(
            EOIOParserResult(output_products={}, input_products={}, adfs={}),
            eoqc_cfg=None,
        )
    finally:
        EOConfiguration()["triggering__dry_run"] = False

    processing_unit.run.assert_not_called()
    processing_unit.run_validating.assert_not_called()
    assert "Dry-run mode enabled: workflow validated; no products will be opened or written" in caplog.text
    assert "Dry-run unit 1/1: unit" in caplog.text


@pytest.mark.unit
def test_workflow_rejects_empty_active_unit_list():
    processing_unit = mock.Mock()
    processing_unit.identifier = "inactive"

    with pytest.raises(TriggerInvalidWorkflow, match="at least one active processing unit"):
        EOProcessorWorkFlow(workflow_units=[])

    with pytest.raises(TriggerInvalidWorkflow, match="at least one active processing unit"):
        EOProcessorWorkFlow(
            workflow_units=[
                WorkFlowUnitDescription(
                    active=False,
                    mode="",
                    processing_unit=processing_unit,
                    inputs={},
                    adfs={},
                    outputs={},
                    parameters={},
                    step=0,
                    validate=True,
                ),
            ],
        )


@pytest.mark.unit
def test_workflow_unit_active_accepts_resolved_env_expression(monkeypatch):
    monkeypatch.setenv("MODE", "L1B")
    raw = WorkflowRaw(
        name="unit",
        module="tests.ut.computing.test_abstract",
        processing_unit="TestAbstractProcessingUnit",
        active=resolve_env_vars("'${MODE}' == 'L1B'"),
    )

    unit = build_workflow_unit(raw)

    assert unit is not None
    assert unit.active is True


@pytest.mark.unit
def test_workflow_unit_active_false_expression_skips_import(monkeypatch):
    monkeypatch.setenv("MODE", "L1A")
    raw = WorkflowRaw(
        name="unit",
        module="missing_module",
        processing_unit="MissingProcessingUnit",
        active=resolve_env_vars("'${MODE}' == 'L1B'"),
    )

    assert build_workflow_unit(raw) is None


@pytest.mark.unit
def test_workflow_unit_active_rejects_invalid_expression():
    raw = WorkflowRaw(
        name="unit",
        module="tests.ut.computing.test_abstract",
        processing_unit="TestAbstractProcessingUnit",
        active="MODE is L1B",
    )

    with pytest.raises(TriggeringConfigurationError, match="Invalid workflow active expression"):
        build_workflow_unit(raw)


@pytest.mark.unit
def test_workflow_unit_active_rejects_non_boolean_expression():
    raw = WorkflowRaw(
        name="unit",
        module="tests.ut.computing.test_abstract",
        processing_unit="TestAbstractProcessingUnit",
        active="'L1B'",
    )

    with pytest.raises(TriggeringConfigurationError, match="Expression must evaluate to a boolean"):
        build_workflow_unit(raw)


@pytest.mark.unit
def test_workflow_unit_accepts_matching_processor_version():
    raw = WorkflowRaw(
        name="unit",
        module="tests.ut.computing.test_abstract",
        processing_unit="TestHistoryProcessor",
        processor_version="1.2",
        inputs={"in": "input"},
        adfs={"dem": "dem"},
    )

    unit = build_workflow_unit(raw)

    assert unit is not None
    assert unit.processing_unit.PROCESSOR_VERSION == "1.2"


@pytest.mark.unit
def test_workflow_unit_rejects_processor_version_mismatch():
    raw = WorkflowRaw(
        name="unit",
        module="tests.ut.computing.test_abstract",
        processing_unit="TestHistoryProcessor",
        processor_version="9.9",
    )

    with pytest.raises(TriggeringConfigurationError, match="Processor version mismatch.*TestHistoryProcessor"):
        build_workflow_unit(raw)


@pytest.mark.unit
def test_context_manager_forward_worker_logger_parsing():
    payload = {
        "context_managers": [
            {
                "module": "eopf.dask_utils.dask_context_manager",
                "context_manager": "DaskContext",
                "parameters": {
                    "cluster_type": "local",
                    "wait_for_workers": True,
                    "wait_timeout": 12,
                    "wait_raises": False,
                    "connect_timeout": 34,
                    "forward_worker_logger": "distributed.worker",
                    "forward_worker_logger_level": 20,
                },
            },
        ],
    }

    raw = load_section(payload, ContextManagersSectionRaw).context_managers
    assert raw[0].parameters["wait_for_workers"] is True
    assert raw[0].parameters["wait_timeout"] == 12
    assert raw[0].parameters["wait_raises"] is False
    assert raw[0].parameters["connect_timeout"] == 34
    assert raw[0].parameters["forward_worker_logger"] == "distributed.worker"
    assert raw[0].parameters["forward_worker_logger_level"] == 20


@pytest.mark.unit
@pytest.mark.dask_only
def test_context_manager_forward_worker_logger_building():
    from eopf.dask_utils.dask_context_manager import DaskContext

    payload = {
        "context_managers": [
            {
                "module": "eopf.dask_utils.dask_context_manager",
                "context_manager": "DaskContext",
                "parameters": {
                    "cluster_type": "local",
                    "wait_for_workers": True,
                    "wait_timeout": 12,
                    "wait_raises": False,
                    "connect_timeout": 34,
                    "forward_worker_logger": "distributed.worker",
                    "forward_worker_logger_level": 20,
                },
            },
        ],
    }

    raw = load_section(payload, ContextManagersSectionRaw).context_managers
    ctx = build_context_managers(raw)[0]
    assert isinstance(ctx, DaskContext)
    assert ctx._wait_workers is True
    assert ctx._wait_timeout == 12
    assert ctx._wait_raises is False
    assert ctx._connect_timeout == 34
    assert ctx._forward_worker_logger == "distributed.worker"
    assert ctx._forward_worker_logger_level == 20


@pytest.mark.unit
def test_context_manager_building_from_payload_declaration():
    CONTEXT_MANAGER_EVENTS.clear()
    payload = {
        "context_managers": [
            {
                "module": "tests.ut.triggering.test_triggering",
                "context_manager": "RecordingContextManager",
                "parameters": {"marker": "payload"},
            },
            {
                "module": "tests.ut.triggering.test_triggering",
                "context_manager": "RecordingContextManager",
                "parameters": {"marker": "inactive"},
                "active": False,
            },
        ],
    }

    context_managers = build_context_managers(load_section(payload, ContextManagersSectionRaw).context_managers)

    assert len(context_managers) == 1
    with context_managers[0]:
        CONTEXT_MANAGER_EVENTS.append("inside")
    assert CONTEXT_MANAGER_EVENTS == ["enter:payload", "inside", "exit:payload"]
