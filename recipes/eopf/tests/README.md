Test input data and licenses
============================

This document describes the test input data used by the test suite, including
their provenance, associated license, and any usage constraints.

## Default licensing and copyright

Unless otherwise specified for a particular dataset, the default license for
test input data described in this document is:

- License type: Apache License 2.0

And the default copyright holder is:

- Copyright holder: 2026 CS - SOPRA STERIA

## Datasets

### Local dataset

- Path(s): `tests/data/sample-product`
- Description: Example product used for reading and metadata validation tests
- Provenance / source: Local repository test data
- Copyright holder: 2026 CS - SOPRA STERIA
- License type: Apache License 2.0
- Source URL / S3 URI: N/A (local repository copy)

### S3 dataset input

- Path(s): `s3://dpr-cpm-input/cpm/unit_test_data`
- Description: External test data used by the unit test
- Provenance / source: External S3 storage
- Copyright holder: 2026 CS - SOPRA STERIA
- License type: Apache License 2.0
- Source URL / S3 URI: `s3://dpr-cpm-input/cpm/unit_test_data`

### S3 dataset output

- Path(s): `s3://dpr-cpm-output/cpm/unit_test_data`
- Description: External test data used by the unit test
- Provenance / source: External S3 storage
- Copyright holder: 2026 CS - SOPRA STERIA
- License type: Apache License 2.0
- Source URL / S3 URI: `s3://dpr-cpm-output/cpm/unit_test_data`

## Notes

For s3 test configuration data, please refer to the documentation:
https://cpm.pages.eopf.copernicus.eu/eopf-cpm/main/contributing.html#s3-test-cases
