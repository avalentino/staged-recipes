#
# Copyright (C) 2026 CS Group
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
import pytest
from xarray import DataTree

from eopf import EOConfiguration
from eopf.dask_utils import dask_context_utils, dask_helpers
from eopf.store import convert as convert_module
from eopf.store.convert import NamingStrategy

pytest_plugins = ("tests.ut.store.utils",)


@pytest.mark.unit
def test_convert_does_not_require_distributed_by_default(monkeypatch):
    EOConfiguration()["store__convert__use_dask_config"] = False

    def fail_if_called(feature: str) -> None:
        raise AssertionError(f"{feature} should not be required")

    monkeypatch.setattr(convert_module, "require_distributed", fail_if_called)

    with convert_module._setup_dask():
        pass


class _FakeProduct:
    def __init__(self, datasize=1):
        self.compute_called = False
        self.cpm = self
        self._datasize = datasize

    def compute(self):
        self.compute_called = True
        return "computed-product"

    def datasize(self):
        return self._datasize


@pytest.mark.unit
def test_staged_convert_with_remote_dask_computes_before_local_write(monkeypatch):
    product = _FakeProduct()
    write_calls = []

    monkeypatch.setattr(convert_module, "_has_remote_distributed_client", lambda: True)

    def fake_write_datatree(**kwargs):
        write_calls.append(kwargs)

    monkeypatch.setattr(convert_module, "write_datatree", fake_write_datatree)

    convert_module._write_staged_local_product(
        eop=product,
        local_product_path="/tmp/product.nc",
        target_fpath="s3://bucket/product.nc",
        target_store_kwargs={"engine": "cpm_netcdf", "storage_options": {"key": "secret"}, "mode": "w"},
        logger=convert_module.logging.getLogger("test"),
    )

    assert product.compute_called
    assert write_calls == [
        {
            "filename_or_obj": "/tmp/product.nc",
            "dtree": "computed-product",
            "engine": "cpm_netcdf",
            "mode": "w",
            "compute": True,
        },
    ]


@pytest.mark.unit
def test_staged_convert_with_remote_dask_large_netcdf_raises_before_compute(monkeypatch):
    product = _FakeProduct(datasize=10_000)

    monkeypatch.setattr(convert_module, "_has_remote_distributed_client", lambda: True)
    monkeypatch.setattr("eopf.store.staging.psutil.virtual_memory", lambda: type("Memory", (), {"available": 1})())

    def fail_if_called(**kwargs):
        raise AssertionError("large NetCDF staged output should fail before writing")

    monkeypatch.setattr(convert_module, "write_datatree", fail_if_called)

    with pytest.raises(ValueError, match="outputs without a sub-DataTree writer"):
        convert_module._write_staged_local_product(
            eop=product,
            local_product_path="/tmp/product.nc",
            target_fpath="s3://bucket/product.nc",
            target_store_kwargs={"engine": "cpm_netcdf", "storage_options": {"key": "secret"}, "mode": "w"},
            logger=convert_module.logging.getLogger("test"),
        )

    assert not product.compute_called


@pytest.mark.unit
def test_staged_convert_with_remote_dask_large_zarr_uses_subdatatree(monkeypatch):
    product = _FakeProduct(datasize=10_000)
    write_calls = []

    monkeypatch.setattr(convert_module, "_has_remote_distributed_client", lambda: True)
    monkeypatch.setattr("eopf.store.staging.psutil.virtual_memory", lambda: type("Memory", (), {"available": 1})())

    def fake_write_datatree(**kwargs):
        write_calls.append(kwargs)

    monkeypatch.setattr(convert_module, "write_datatree", fake_write_datatree)

    convert_module._write_staged_local_product(
        eop=product,
        local_product_path="/tmp/product.zarr",
        target_fpath="s3://bucket/product.zarr",
        target_store_kwargs={"engine": "cpm_zarr", "storage_options": {"key": "secret"}, "mode": "w"},
        logger=convert_module.logging.getLogger("test"),
    )

    assert not product.compute_called
    assert write_calls == [
        {
            "filename_or_obj": "/tmp/product.zarr",
            "dtree": product,
            "engine": "cpm_zarr_subdatatree",
            "mode": "w",
            "compute": True,
        },
    ]


@pytest.mark.unit
def test_convert_requires_distributed_when_dask_config_is_enabled(monkeypatch):
    EOConfiguration()["store__convert__use_dask_config"] = True

    def fail_if_called(feature: str) -> None:
        raise ImportError(f"{feature} requires distributed")

    monkeypatch.setattr(convert_module, "require_distributed", fail_if_called)

    with pytest.raises(
        ImportError,
        match="Dask-configured convert requires the Dask distributed package.*without --use-dask-config",
    ):
        convert_module._setup_dask()


@pytest.mark.unit
def test_convert_reuses_existing_dask_client(monkeypatch):
    EOConfiguration()["store__convert__use_dask_config"] = True
    existing_client = object()

    monkeypatch.setattr(convert_module, "require_distributed", lambda feature: None)
    monkeypatch.setattr(dask_helpers, "get_distributed_client", lambda: existing_client)

    def fail_if_called():
        raise AssertionError("convert should not create a Dask context when a client already exists")

    monkeypatch.setattr(dask_context_utils, "init_from_eo_configuration", fail_if_called)

    with convert_module._setup_dask():
        pass


@pytest.mark.unit
def test_convert_generated_name_registers_builtin_writer_for_engine(monkeypatch, tmp_path, isolated_writer_registry):
    product = DataTree(name="MY_PRODUCT")
    written_paths = []

    monkeypatch.setattr(convert_module, "open_datatree", lambda *args, **kwargs: product)
    monkeypatch.setattr(convert_module, "_validate_converted", lambda *args, **kwargs: None)
    monkeypatch.setattr(convert_module, "add_eopf_cpm_entry_to_history", lambda *args, **kwargs: None)

    def fake_write_datatree(filename_or_obj, dtree, **kwargs):
        written_paths.append(filename_or_obj)

    monkeypatch.setattr(convert_module, "write_datatree", fake_write_datatree)

    convert_module.convert(
        "source.SEN3",
        str(tmp_path),
        target_store_kwargs={"engine": "cpm_zarr"},
        naming_strategy=NamingStrategy.PRODUCT_NAME,
    )

    assert written_paths == [tmp_path / "MY_PRODUCT.zarr"]


@pytest.mark.unit
def test_convert_generated_name_reports_available_writers_for_unknown_engine(
    monkeypatch,
    tmp_path,
    isolated_writer_registry,
):
    product = DataTree(name="MY_PRODUCT")

    monkeypatch.setattr(convert_module, "open_datatree", lambda *args, **kwargs: product)
    monkeypatch.setattr(convert_module, "_validate_converted", lambda *args, **kwargs: None)
    monkeypatch.setattr(convert_module, "add_eopf_cpm_entry_to_history", lambda *args, **kwargs: None)

    with pytest.raises(Exception, match=r"No registered writer with format : zarr, available : .*cpm_zarr"):
        convert_module.convert(
            "source.SEN3",
            str(tmp_path),
            target_store_kwargs={"engine": "zarr"},
            naming_strategy=NamingStrategy.PRODUCT_NAME,
        )
