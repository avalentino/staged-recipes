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

import pytest

from eopf.store.convert import convert

pytestmark = pytest.mark.dask_only
pytest.importorskip("dask")
pytest.importorskip("distributed")


from eopf.common.file_utils import AnyPath


@pytest.mark.unit
@pytest.mark.parametrize(
    "product_path, glob_pattern",
    [
        pytest.param("S3_OL_1_unit", "S03OLCERR", marks=pytest.mark.need_files),
    ],
    indirect=["product_path"],
)
def test_convert(
        product_path: str,
    glob_pattern,
    OUTPUT_DIR: str,
):
    convert(product_path, os.path.join(OUTPUT_DIR, f"convert/{glob_pattern}.zarr"))
    products = AnyPath.cast(os.path.join(OUTPUT_DIR, "convert")).glob(glob_pattern + "*")
    assert len(products) > 0
    measurements = products[0].glob("measurements")
    assert len(measurements) == 1
    # products[0].rm(recursive=True)


@pytest.mark.unit
@pytest.mark.real_s3
def test_convert_s3(OUTPUT_DIR: str, s3_test_data, s3_config_real):
    protocol, base_path = s3_test_data
    input_path = f"{protocol}://{base_path}/S3B_OL_1_ERR____20230506T015316_20230506T015616_20230711T065804_0179_079_117______LR1_D_NR_003.SEN3"

    convert(
        input_path,
        os.path.join(
            OUTPUT_DIR,
            "convert_s3/S3B_OL_1_ERR____20230506T015316_20230506T015616_20230711T065804_0179_079_117______LR1_D_NR_003.zarr",
        ),
        source_store_kwargs={"storage_options": s3_config_real},
    )
    products = AnyPath.cast(os.path.join(OUTPUT_DIR, "convert_s3")).glob("S3B_OL_1_ERR*")
    assert len(products) > 0
    measurements = products[0].glob("measurements")
    assert len(measurements) == 1
    products[0].rm(recursive=True)


@pytest.mark.unit
@pytest.mark.real_s3
def test_convert_s3_with_stage_source(OUTPUT_DIR: str, s3_test_data, s3_config_real):
    """Source is downloaded to local temp before conversion — the typical S3 input scenario."""
    protocol, base_path = s3_test_data
    input_path = f"{protocol}://{base_path}/S3B_OL_1_ERR____20230506T015316_20230506T015616_20230711T065804_0179_079_117______LR1_D_NR_003.SEN3"
    output_path = os.path.join(
        OUTPUT_DIR,
        "convert_s3_stage_source/S3B_OL_1_ERR____20230506T015316_20230506T015616_20230711T065804_0179_079_117______LR1_D_NR_003.zarr",
    )

    convert(
        input_path,
        output_path,
        source_store_kwargs={"storage_options": s3_config_real},
        stage_source=True,
    )

    products = AnyPath.cast(os.path.join(OUTPUT_DIR, "convert_s3_stage_source")).glob("S3B_OL_1_ERR*")
    assert len(products) > 0
    measurements = products[0].glob("measurements")
    assert len(measurements) == 1
    products[0].rm(recursive=True)


@pytest.mark.unit
@pytest.mark.real_s3
def test_convert_s3_with_stage_source_and_stage_target(
    OUTPUT_DIR: str,
    s3_test_data,
    s3_config_real,
    s3_output_test_data,
    s3_output_config_real,
):
    """Source downloaded locally, output written locally then uploaded to S3.

    Both stage_source and stage_target are required for the typical full S3→S3
    pipeline: most product types cannot be read directly from S3.
    """
    protocol, base_path = s3_test_data
    input_path = f"{protocol}://{base_path}/S3B_OL_1_ERR____20230506T015316_20230506T015616_20230711T065804_0179_079_117______LR1_D_NR_003.SEN3"

    out_protocol, out_base_path = s3_output_test_data
    output_path = f"{out_protocol}://{out_base_path}/convert_stage_both/S3B_OL_1_ERR____20230506T015316_20230506T015616_20230711T065804_0179_079_117______LR1_D_NR_003.zarr"

    convert(
        input_path,
        output_path,
        source_store_kwargs={"storage_options": s3_config_real},
        target_store_kwargs={"engine": "cpm_zarr", "mode": "w", "storage_options": s3_output_config_real},
        stage_source=True,
        stage_target=True,
    )

    output_anypath = AnyPath.cast(output_path, **s3_output_config_real)
    assert output_anypath.exists()
    measurements = output_anypath.glob("measurements")
    assert len(measurements) == 1
    output_anypath.rm(recursive=True)
