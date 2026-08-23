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
import time

import pytest

pytestmark = pytest.mark.dask_only
pytest.importorskip("dask")

from eopf.computing.utils import EOHeavyProcessingUnit, EOImageGeneratorUnit, EORechunkingUnit  # noqa: E402
from eopf.dask_utils import (  # noqa: E402
    clear_computation_interrupt,
    computation_interrupted,
    get_computation_interrupt_event,
    signal_computation_interrupt,
)


def _loop_until_computation_interrupted(interrupt_event_name: str, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        if computation_interrupted(interrupt_event_name):
            return "interrupted"
        if time.monotonic() > deadline:
            raise TimeoutError("Computation interrupt event was not observed")
        time.sleep(0.05)


@pytest.mark.unit
def test_rechunk_unit(fake_quality_datatree):
    rechunk_unit = EORechunkingUnit()
    output = rechunk_unit.run(inputs={"in": fake_quality_datatree}, chunks={"dim_0": 125, "dim_1": 125})
    print(output["in"]["measurements/radiance/oa01_radiance"])


@pytest.mark.unit
def test_image_generator_unit_builds_synthetic_outputs():
    output = EOImageGeneratorUnit("generator").run(
        inputs={},
        size_x=20,
        size_y=20,
        chunk_x=10,
        chunk_y=10,
        nb_var=2,
    )

    assert set(output) == {"out", "multi1", "multi2"}
    assert "b0" in output["out"]["measurements/image"].ds.data_vars
    assert "b1" in output["out"]["measurements/image"].ds.data_vars


@pytest.mark.unit
def test_heavy_processing_unit_builds_synthetic_output_without_compute():
    generated = EOImageGeneratorUnit("generator").run(
        inputs={},
        size_x=20,
        size_y=20,
        chunk_x=10,
        chunk_y=10,
        nb_var=2,
    )

    output = EOHeavyProcessingUnit("heavy").run_validating({"in": generated["out"]})["out"]

    assert "edges_b0" in output["measurements/image/edges"].ds.data_vars
    assert "fft_b1" in output["measurements/image/fft"].ds.data_vars
    assert "denoised_b0" in output["measurements/image/denoised"].ds.data_vars
    assert "mask_b1" in output["conditions/mask"].ds.data_vars


@pytest.mark.unit
def test_submitted_computation_stops_when_interruption_event_is_set(dask_context_threads):
    pytest.importorskip("distributed")
    interrupt_event = get_computation_interrupt_event()

    assert interrupt_event is not None
    assert clear_computation_interrupt()

    future = dask_context_threads.client.submit(
        _loop_until_computation_interrupted,
        interrupt_event.name,
    )
    time.sleep(0.2)
    assert not future.done()

    assert signal_computation_interrupt()

    assert future.result(timeout=5) == "interrupted"
