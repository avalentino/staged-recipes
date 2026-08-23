#
# Copyright (C) 2025 ESA
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
# pytestmark = pytest.mark.dask_only
# da = pytest.importorskip("dask.array")
# delayed = pytest.importorskip("dask").delayed

from typing import Any

import pytest
import numpy as np
import xarray as xr
from xarray import DataTree

pytestmark = pytest.mark.dask_only
da = pytest.importorskip("dask.array")
pytest.importorskip("distributed")

from eopf import EOConfiguration
from eopf.dask_utils.datatree_analyser import (
    AnomalyDescriptor,
    DaskAnalyzerData,
    TooBigChunksRule,
    TooSmallArrayRule,
    TooSmallChunksRule,
    TooMuchNodeRule,
)
from eopf.dask_utils.dask_helpers import estimate_limit_from_dask_conf
from eopf.dask_utils.dask_context_utils import init_from_eo_configuration


@pytest.fixture
def local_dask_config() -> dict[str, Any]:
    conf = EOConfiguration()
    conf.clear_loaded_configurations()
    conf.load_dict(
        {
            "dask_context": {
                "cluster_type": "local",
                "cluster_config": {
                    "threads_per_worker": 2,
                    "n_workers": 2,
                    "memory_limit": "8GiB",
                },
            },
        },
    )
    init_from_eo_configuration()


@pytest.fixture
def make_chunked_dataarray():
    def _factory(
        name: str,
        shape=(100, 1024, 1024),
        chunks=(1, 1024, 1024),
        dtype="float32",
        dims=("z", "y", "x"),
    ):
        data = da.random.random(shape, chunks=chunks).astype(dtype)

        return xr.DataArray(
            data,
            dims=dims,
            name=name,
            attrs={"description": "chunked DataArray"},
        )

    return _factory


@pytest.fixture
def make_test_dataset(make_chunked_dataarray):
    def _factory(label: str, chunks, dtype):
        return xr.Dataset(
            {
                "data": make_chunked_dataarray(
                    name="data",
                    shape=(10, 10980, 10980),
                    chunks=chunks,
                    dtype=dtype,
                ),
            },
            attrs={"dataset": label},
        )

    return _factory


@pytest.mark.unit
def test_analyze_datatree(local_dask_config, make_test_dataset):

    dt = DataTree(name="root")

    dt["raw"] = DataTree(dataset=make_test_dataset("raw", chunks=(3, 256, 256), dtype="float32"))
    dt["processed"] = DataTree(dataset=make_test_dataset("processed", chunks=(5, 2048, 2048), dtype="float32"))
    dt["analysis"] = DataTree(
        dataset=make_test_dataset("analysis", chunks=(5, 50 * 10980, 50 * 10980), dtype="float32"),
    )
    conf = EOConfiguration()
    dask_parameters = estimate_limit_from_dask_conf(conf)

    rules = [
        TooMuchNodeRule(max_node=dask_parameters.max_node),
        TooSmallChunksRule(min_mb=dask_parameters.min_chunk_size),
        TooBigChunksRule(max_mb=dask_parameters.max_chunk_size),
        TooSmallArrayRule(min_mb=dask_parameters.min_array_size),
    ]

    engine = DaskAnalyzerData(rules)
    detected_anomalies = engine.analyse(dt)
    expected_anomalies = [
        AnomalyDescriptor(
            level="warning",
            rule="TooMuchNode",
            node_path="",
            variable=None,
            message="The node number is too big for Dask",
            details={"current_node_size": 4, "target_node_size": 2},
        ),
        AnomalyDescriptor(
            level="warning",
            rule="TooSmallChunks",
            node_path="/raw",
            variable="data",
            message="Chunks are too small < 100 Mb (too many tasks)",
            details={"size_mb": np.float64(0.75)},
        ),
        AnomalyDescriptor(
            level="warning",
            rule="TooSmallChunks",
            node_path="/processed",
            variable="data",
            message="Chunks are too small < 100 Mb (too many tasks)",
            details={"size_mb": np.float64(80.0)},
        ),
        AnomalyDescriptor(
            level="warning",
            rule="TooBigChunks",
            node_path="/analysis",
            variable="data",
            message="Chunks are too big > 491.52 Mb (out of memory risk)",
            details={"size_mb": np.float64(2299.51)},
        ),
    ]
    assert detected_anomalies == expected_anomalies
