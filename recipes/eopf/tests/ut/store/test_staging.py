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

import logging

import pytest

from eopf.store.staging import prepare_staged_local_write


class _FakeProduct:
    def __init__(self, datasize=1):
        self.cpm = self
        self.compute_called = False
        self._datasize = datasize

    def compute(self):
        self.compute_called = True
        return "computed-product"

    def datasize(self):
        return self._datasize


@pytest.mark.unit
def test_prepare_staged_local_write_strips_remote_storage_options():
    product = _FakeProduct()

    staged_write = prepare_staged_local_write(
        eo_data=product,
        engine="cpm_netcdf",
        product_path="s3://bucket/product.nc",
        writer_kwargs={"mode": "w", "storage_options": {"key": "secret"}},
        has_remote_client=False,
        memory_fit_safety_ratio_config="store__convert__stage_target_memory_fit_safety_ratio",
        default_memory_fit_safety_ratio=0.25,
        operation_name="test staging",
        logger=logging.getLogger("test"),
    )

    assert staged_write.eo_data is product
    assert staged_write.engine == "cpm_netcdf"
    assert staged_write.writer_kwargs == {"mode": "w"}
    assert staged_write.uses_subdatatree_writer is False


@pytest.mark.unit
def test_prepare_staged_local_write_uses_subdatatree_for_large_remote_zarr(monkeypatch):
    product = _FakeProduct(datasize=10_000)

    monkeypatch.setattr("eopf.store.staging.psutil.virtual_memory", lambda: type("Memory", (), {"available": 1})())

    staged_write = prepare_staged_local_write(
        eo_data=product,
        engine="cpm_zarr",
        product_path="s3://bucket/product.zarr",
        writer_kwargs={"mode": "w", "storage_options": {"key": "secret"}},
        has_remote_client=True,
        memory_fit_safety_ratio_config="store__convert__stage_target_memory_fit_safety_ratio",
        default_memory_fit_safety_ratio=0.25,
        operation_name="test staging",
        logger=logging.getLogger("test"),
    )

    assert staged_write.eo_data is product
    assert staged_write.engine == "cpm_zarr_subdatatree"
    assert staged_write.writer_kwargs == {"mode": "w", "compute": True}
    assert staged_write.uses_subdatatree_writer is True
    assert not product.compute_called
