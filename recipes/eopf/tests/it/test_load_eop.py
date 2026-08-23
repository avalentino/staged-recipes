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
import json
import os
import shutil
import time
from pathlib import Path

import pytest
from xarray import DataTree

from eopf import open_datatree, write_datatree
from eopf.common.constants import EOPF_CPM_PATH
from eopf.common.file_utils import AnyPath
from eopf.exceptions import InvalidProductError
from eopf.product.datatree_validation import ValidationMode
from eopf.store.mapping_factory import EOPFMappingFactory
from tests.it.utils import (
    EXCEPTED_MAPPINGS,
    get_test_logger,
    find_test_products, get_test_products_list, CI_CACHE_DIRS,
)

# Path to integration test data directory
INTEGRATION_DIR = Path(EOPF_CPM_PATH) / "../tests/data/integration/"
ZARR_TMP_DIR = INTEGRATION_DIR / "tmp_zarr"

# Initialize the Test logger
TEST_LOGGER = get_test_logger()

SUPPORTED_PRODUCT_TYPES = [
    (item.product_type, item.processing_version)
    for item in EOPFMappingFactory()._mapping_set
    if item.path.name not in set(EXCEPTED_MAPPINGS)
]

marked_products = []  # pytest ... -m "S01" / "S02" / "S03"
for pdt in SUPPORTED_PRODUCT_TYPES:
    mission = pdt[0][:3]
    mark = getattr(pytest.mark, mission)
    marked_products.append(pytest.param(*pdt, marks=mark))


def save_eop_to_zarr(
    eop: DataTree,
    product_type: str,
) -> Path:
    """
    Save the received EOProduct to ZARR storage format and return its Path.
    """
    # Save to ZARR
    zarr_name = f"{product_type}_test_product_{eop.name}.zarr"
    zarr_product = ZARR_TMP_DIR / zarr_name
    write_datatree(eop, zarr_product,engine="cpm_zarr")
    return zarr_product


@pytest.fixture
def zarr_temp_directory():
    """
    Fixture to create ZARR_TMP_DIR temporary directory for the tests,
    and remove it after the tests are run.
    """
    # Create the temporary directory
    os.makedirs(ZARR_TMP_DIR, exist_ok=True)
    yield ZARR_TMP_DIR  # This will be returned to the test function

    # Remove the temporary directory after the test
    shutil.rmtree(ZARR_TMP_DIR, ignore_errors=True)


def compare_conversion_performance(product_name: str, zarr_product: Path, conversion_time: float) -> bool | None:
    """
    Check if the newly created zarr is comparable to previous zarr product, comparing:
        - file size (in number of bytes)
        - number of sub-files in the zarr
        - conversion time (in seconds)

    Parameters
    ----------
    product_name: str
        Name of the product (used to find reference performances)
    zarr_product: Path
        Path to the converted Zarr product
    conversion_time: float
        Measured time for the conversion

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a performance is worse than before, with a 10% margin
    """
    zarr_product_any = AnyPath.cast(zarr_product)
    zarr_size = zarr_product_any.get_size()
    nb_files = zarr_product_any.get_number_of_files()

    ref_json_path = INTEGRATION_DIR / "ref_performances.json"
    with open(str(ref_json_path), "r") as ref_file:
        ref_dict_dict = json.load(ref_file)
        product_ref_dict = ref_dict_dict.get(product_name)

    actual_perfs = {
        product_name: {"zarr_size": zarr_size, "nb_files": nb_files, "conversion_time": conversion_time},
    }

    # Check if product has a ref size
    if product_ref_dict is None:
        raise ValueError(
            "Missing product ref performances json (located here: eopf-cpm/tests/data/integration/"
            f"ref_performances.json). Add the following: {actual_perfs}",
        )

    ref_size = product_ref_dict["zarr_size"]
    # Check if new product size is less than 110% of original size
    if not zarr_size < ref_size * 1.1:
        raise ValueError(
            f"File size changed too much. Previous file size was {ref_size / 1e6} MB, new size "
            f"is {zarr_size / 1e6}MB. Full performances dictionnary is: {actual_perfs}.",
        )

    ref_nb_files = product_ref_dict["nb_files"]
    # Check if new product has less than 110% of original number of files
    if not nb_files < ref_nb_files * 1.1:
        raise ValueError(
            f"Number of sub-files changed too much. Previous file had {ref_nb_files} sub-files, "
            f"new file has {nb_files} files. Full performances dictionnary is: {actual_perfs}",
        )

    ref_conversion_time = product_ref_dict["conversion_time"]
    # Check if new product doesn't take more than 110% of original conversion time + 5 seconds
    if not conversion_time < ref_conversion_time * 1.1 + 5:
        raise ValueError(
            f"Conversion time is longer than before. It previously took {ref_conversion_time} "
            f"seconds, it now takes {conversion_time} seconds. Full performances dictionnary "
            f"is: {actual_perfs}",
        )


@pytest.mark.integration
@pytest.mark.need_files
@pytest.mark.parametrize(
    "product_type, processing_version",
    marked_products,
)
def test_SAFE_load_to_eop_to_zarr(
    zarr_temp_directory: Path,
    product_type: str,
    processing_version: str,
):
    """
    Test Strategy:
        Retrieve mapping content given the product type and processing version
        Find test products in the DEFAULT_S3_TEST_PRODUCTS_PATHS
        Test legacy SAFE product loading into EOProduct
        Save the resulted EOProduct into ZARR storage format
        Test ZARR product loading into EOProduct
        Compare the EOProduct from legacy product loading with EOProduct from ZARR product loading
        Delete all temporary files and folders
    """
    # Temporary directory for zarr exists
    assert zarr_temp_directory.exists()

    # Retrieve and Check the mapping content for current scenario
    mapping_content = EOPFMappingFactory().get_mapping(
        product_type=product_type,
        legacy_processing_version=processing_version,
    )

    # Find test product - default only first product found is returned
    test_product = find_test_products(
        mapping_content=mapping_content,
        test_products_list=get_test_products_list(CI_CACHE_DIRS),
    )

    if not test_product:
        msg = (
            f"For product type: {mapping_content['product_type']}, "
            f"processing version: {mapping_content['processing_version']} "
            f"there is no product in test_products_list "
        )
        raise RuntimeError(msg)

    # Test product loading
    eoProduct = open_datatree(test_product, mask_and_scale=False, engine="safe")

    # Save the resulted EOProduct into ZARR storage format
    start_time = time.time()
    zarr_product = save_eop_to_zarr(eop=eoProduct, product_type=product_type)
    elapsed_time = time.time()
    conversion_time = elapsed_time - start_time

    assert zarr_product.exists(), "Conversion output was not written"

    # Test ZARR product loading into EOProduct
    z_eoProduct = open_datatree(zarr_product, mask_and_scale=False,engine="cpm_zarr")

    # Compare the EOProduct from legacy product loading with EOProduct from ZARR product loading
    assert eoProduct == z_eoProduct

    # Make sure the file size, number of subfiles and conversion didn't change
    compare_conversion_performance(product_type, zarr_product, conversion_time)

    # Delete the zarr product
    shutil.rmtree(zarr_product)
    assert not zarr_product.exists()
