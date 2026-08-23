import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union, Any

from eopf.store.mapping_factory import EOPFMappingFactory

CI_CACHE_DIRS = [
    Path("/data/cache-ci/ci_test_data/s01"),
    Path("/data/cache-ci/ci_test_data/s02"),
    Path("/data/cache-ci/ci_test_data/s03"),
]

# Mappings corresponding to unavailable products on s3 bucket
EXCEPTED_MAPPINGS = [
    "S01SN1RAW_pv1.0.0_mv1.0.0.json",
    "S01SN4RAW_pv1.0.0_mv1.0.0.json",
    "S01SENRAW_pv1.0.0_mv1.0.0.json",
    "S01SN2RAW_pv1.0.0_mv1.0.0.json",
    "S01SN6RAW_pv1.0.0_mv1.0.0.json",
    "S01SS2RAW_pv1.0.0_mv1.0.0.json",
    "S01SS5RAW_pv1.0.0_mv1.0.0.json",
    "S01SS6RAW_pv1.0.0_mv1.0.0.json",
    "S01SN5RAW_pv1.0.0_mv1.0.0.json",
    # TMP: ignore level-0 products cf https://gitlab.eopf.copernicus.eu/cpm/eopf-cpm/-/issues/1093
    # and https://gitlab.eopf.copernicus.eu/cpm/eopf-cpm/-/merge_requests/1193
    "S01AISRAW_pv1.0.0_mv1.0.0.json",
    "S01GPSRAW_pv1.0.0_mv1.0.0.json",
    "S01HKMRAW_pv1.0.0_mv1.0.0.json",
    "S01SEWRAW_pv1.0.0_mv1.0.0.json",
    "S01SIWRAW_pv1.0.0_mv1.0.0.json",
    "S01SN3RAW_pv1.0.0_mv1.0.0.json",
    "S01SS1RAW_pv1.0.0_mv1.0.0.json",
    "S01SS3RAW_pv1.0.0_mv1.0.0.json",
    "S01SS4RAW_pv1.0.0_mv1.0.0.json",
    "S01SWVRAW_pv1.0.0_mv1.0.0.json",
    "S01SRFANC_pv1.0.0_mv1.0.0.json",
    "S02MSIRAW_pv1.0.0_mv1.0.0.json",
    "S02SADRAW_pv1.0.0_mv1.0.0.json",
    "S02HKMRAW_pv1.0.0_mv1.0.0.json",
]

# not supported
EXCEPTED_SUFFIXES = [".tar", ".gz", ".zip", ".json"]

DEFAULT_S3_CONFIG = {
    "key": os.environ.get("S3_KEY"),
    "secret": os.environ.get("S3_SECRET"),
    "client_kwargs": {"endpoint_url": os.environ.get("S3_URL"), "region_name": os.environ.get("S3_REGION")},
}


def get_test_products_list(
    test_data_folder: Union[Path, list[Path]],
) -> list[Path]:

    if isinstance(test_data_folder, Path):
        test_products = test_data_folder.iterdir()
    else:
        test_products = []
        for product_list in test_data_folder:
            test_products.extend(list(product_list.iterdir()))

    # exclude products with the suffixes in EXCEPTED_SUFFIXES
    test_products = [
        test_product
        for test_product in test_products
        if set(test_product.suffixes).isdisjoint(set(EXCEPTED_SUFFIXES))
    ]
    return test_products


# Create a test logger
def get_test_logger() -> logging.Logger:

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("load_eop.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


def find_test_products(
    mapping_content: dict[str, Any],  # MappingContentType
    test_products_list: list[Path],
    return_only_one: Optional[bool] = True,
) -> Path | list[Path] | None:
    """
    .
    """
    # Get a test product as per the provided mapping content
    if return_only_one:
        for test_product in test_products_list:
            if EOPFMappingFactory()._guess_compatible(mapping_content, test_product.as_posix()):
                return test_product  # Return the first product that matches the given mapping_content
    else:
        test_products = []
        for product in test_products_list:
            if EOPFMappingFactory()._guess_compatible(mapping_content, product.as_posix()):
                test_products.append(product)
        return test_products  # Return all products that matches the given mapping_content

    raise FileNotFoundError("No product found")
