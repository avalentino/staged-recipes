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
from click.testing import CliRunner

from eopf import EOConfiguration
from eopf.cli.cli_convert import convert_cli


@pytest.mark.unit
def test_cli_convert_does_not_use_dask_config_by_default(monkeypatch):
    def fake_convert(**kwargs):
        assert not bool(EOConfiguration().get("store__convert__use_dask_config"))
        return None, "target"

    monkeypatch.setattr("eopf.cli.cli_convert.convert", fake_convert)

    result = CliRunner().invoke(convert_cli, ["source", "target"])
    print(repr(result.exception))
    assert result.exit_code == 0


@pytest.mark.unit
def test_cli_convert_can_use_dask_config(monkeypatch):
    def fake_convert(**kwargs):
        assert bool(EOConfiguration().get("store__convert__use_dask_config"))
        return None, "target"

    monkeypatch.setattr("eopf.cli.cli_convert.convert", fake_convert)

    result = CliRunner().invoke(convert_cli, ["source", "target", "--use-dask-config"])
    print(result.exception)
    assert result.exit_code == 0


@pytest.mark.unit
@pytest.mark.parametrize("naming_strategy", ["PRODUCT_NAME", "PRODUCT_ID"])
def test_cli_convert_requires_target_engine_for_generated_names(monkeypatch, naming_strategy):
    def fail_if_called(**kwargs):
        raise AssertionError("convert should not be called with invalid CLI options")

    monkeypatch.setattr("eopf.cli.cli_convert.convert", fail_if_called)

    result = CliRunner().invoke(convert_cli, ["source", "target", "--naming-strategy", naming_strategy])
    print(result.exception)
    assert result.exit_code != 0
    assert "--target-engine is required" in result.output


@pytest.mark.unit
def test_cli_convert_accepts_generated_names_with_target_engine(monkeypatch):
    def fake_convert(**kwargs):
        assert kwargs["target_store_kwargs"] == {"engine": "cpm_zarr"}
        return None, "target"

    monkeypatch.setattr("eopf.cli.cli_convert.convert", fake_convert)

    result = CliRunner().invoke(
        convert_cli,
        ["source", "target", "--naming-strategy", "PRODUCT_ID", "--target-engine", "cpm_zarr"],
    )

    assert result.exit_code == 0
