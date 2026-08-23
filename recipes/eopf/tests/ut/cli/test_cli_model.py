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
import os
import traceback

import pytest
from click.testing import CliRunner

from eopf.cli.cli_model import create_model, generate_model_product, validate_model
from eopf.common.file_utils import AnyPath

pytestmark = pytest.mark.dask_only
pytest.importorskip("dask")


@pytest.mark.need_files
@pytest.mark.unit
def test_cli_model(OUTPUT_DIR, S2_MSIL2A_unit):
    runner = CliRunner()
    output_model_file = os.path.join(OUTPUT_DIR, "S2_MSIL2A.json")

    r = runner.invoke(
        create_model,
        args=[S2_MSIL2A_unit, "--target-path", output_model_file],
    )
    if r.exception is not None:
        traceback.print_exception(type(r.exception), r.exception, r.exception.__traceback__)
    assert r.exception is None
    assert r.exit_code == 0
    assert AnyPath.cast(output_model_file).exists()

    r = runner.invoke(
        validate_model,
        args=[S2_MSIL2A_unit, "--model-path", output_model_file],
    )
    if r.exception is not None:
        traceback.print_exception(type(r.exception), r.exception, r.exception.__traceback__)
    assert r.exception is None
    assert r.exit_code == 0
    assert AnyPath.cast(output_model_file).exists()


@pytest.mark.unit
def test_cli_model_generate_from_model_path(tmp_path, monkeypatch):
    model_path = tmp_path / "model.json"
    target_path = tmp_path / "product.zarr"
    model_path.write_text(
        """
{
    "product_type": "DUMMY",
    "processing_version": "1.0.0",
    "model_version": "1.0.0",
    "variables": {
        "/measurements/b01": {
            "dims": ["y", "x"],
            "dtype": "float64"
        }
    }
}
""",
    )
    write_calls = []

    def fake_write_datatree(dtree, filename_or_obj, engine=None, **kwargs):
        write_calls.append(
            {
                "dtree": dtree,
                "filename_or_obj": filename_or_obj,
                "engine": engine,
                "kwargs": kwargs,
            },
        )

    monkeypatch.setattr("eopf.cli.cli_model.write_datatree", fake_write_datatree)

    result = CliRunner().invoke(
        generate_model_product,
        args=["--model-path", str(model_path), "--target-path", str(target_path)],
    )

    assert result.exception is None
    assert result.exit_code == 0
    assert write_calls[0]["filename_or_obj"] == target_path
    assert write_calls[0]["engine"] == "cpm_zarr"
    assert write_calls[0]["kwargs"]["mode"] == "w-"
    assert write_calls[0]["dtree"].cpm.product_type == "DUMMY"


@pytest.mark.unit
def test_cli_model_generate_uses_model_selection_options(tmp_path, monkeypatch):
    target_path = tmp_path / "selected.zarr"
    factory_calls = []
    write_calls = []

    class FakeModelFactory:
        def get_model_content(self, product_type, processing_version=None, model_version=None):
            factory_calls.append(
                {
                    "product_type": product_type,
                    "processing_version": processing_version,
                    "model_version": model_version,
                },
            )
            return {
                "product_type": product_type,
                "processing_version": processing_version,
                "model_version": model_version,
            }

    monkeypatch.setattr("eopf.cli.cli_model.EOPFModelFactory", lambda: FakeModelFactory())

    def fake_write_datatree(dtree, filename_or_obj, engine=None, **kwargs):
        write_calls.append(
            {
                "dtree": dtree,
                "filename_or_obj": filename_or_obj,
                "engine": engine,
                **kwargs,
            },
        )

    monkeypatch.setattr("eopf.cli.cli_model.write_datatree", fake_write_datatree)

    result = CliRunner().invoke(
        generate_model_product,
        args=[
            "DUMMY",
            "--processing-version",
            "1.0.0",
            "--model-version",
            "1.2.0",
            "--target-path",
            str(target_path),
            "--overwrite",
        ],
    )

    assert result.exception is None
    assert result.exit_code == 0
    assert factory_calls == [
        {
            "product_type": "DUMMY",
            "processing_version": "1.0.0",
            "model_version": "1.2.0",
        },
    ]
    assert write_calls[0]["filename_or_obj"] == target_path
    assert write_calls[0]["engine"] == "cpm_zarr"
    assert write_calls[0]["mode"] == "w"
