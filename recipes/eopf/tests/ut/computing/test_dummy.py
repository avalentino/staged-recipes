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
import pytest

from typing import Any

from eopf.computing import EOProcessingUnit, MappingDataType


class TestDummyProcessor(EOProcessingUnit):
    __test__ = False

    def run(self, inputs: MappingDataType, **kwargs: Any) ->MappingDataType:
        return inputs

    def get_mandatory_input_list(self, **kwargs: Any) -> list[str]:
        return []


def test_legacy_processor_model_alias_warns():
    with pytest.deprecated_call(match="PROCESSOR_MODEL is deprecated; use PROCESSOR_CONTRACT instead"):
        class LegacyDummyProcessor(EOProcessingUnit):
            __test__ = False
            PROCESSOR_NAME = "LegacyDummyProcessor"
            PROCESSOR_VERSION = "1.0.0"
            PROCESSOR_MODEL = False

            def run(self, inputs: MappingDataType, **kwargs: Any) -> MappingDataType:
                return inputs

            def get_mandatory_input_list(self, **kwargs: Any) -> list[str]:
                return []

    assert LegacyDummyProcessor._processing_model is None
