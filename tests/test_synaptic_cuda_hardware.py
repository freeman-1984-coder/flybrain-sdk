"""Actual GPU conformance; fixtures validate math, not full-brain behavior."""

import json
import os

import numpy as np
import pytest

from flybrain import Connectome, Synapse
from flybrain.backends.cuda import _load_cupy
from flybrain.errors import BackendUnavailableError
from flybrain.experimental.synaptic import SynapticCPU, SynapticLIFConfig
from flybrain.experimental.synaptic_cuda import SynapticCUDA


@pytest.fixture
def device():
    try:
        return _load_cupy()
    except BackendUnavailableError as error:
        if os.environ.get("FLYBRAIN_REQUIRE_CUDA") == "1":
            pytest.fail(str(error))
        pytest.skip(f"Actual CUDA device required: {error}")


def graph():
    return Connectome(
        "synaptic-gpu-fixture",
        tuple(str(i) for i in range(6)),
        tuple(
            Synapse(*e)
            for e in [
                (0, 2, 40.0),
                (0, 2, 15.0),
                (1, 2, -33.0),
                (2, 3, 55.0),
                (3, 2, 10.0),
                (2, 2, 2.0),
                (3, 4, -20.0),
            ]
        ),
        {},
        {},
        {},
    )


def equal(cpu, gpu):
    a, b = cpu.snapshot(), gpu.snapshot()
    for field in (
        "engine",
        "fingerprint",
        "tick",
        "spikes",
        "history",
        "last_spike_tick",
        "silenced",
    ):
        assert a[field] == b[field], field
    for field in ("voltage_mv", "synaptic_mv", "rates_hz"):
        np.testing.assert_allclose(a[field], b[field], atol=1e-9, rtol=0, err_msg=field)


@pytest.mark.parametrize(
    "config",
    [
        SynapticLIFConfig(),
        SynapticLIFConfig(dt_ms=0.2),
        SynapticLIFConfig(delay_ms=0),
        SynapticLIFConfig(synapse_tau_ms=20),
    ],
)
def test_cpu_cuda_and_cross_backend_delayed_checkpoint(device, config):
    cpu = SynapticCPU(graph(), config, input_ids=("0", "1"), weight_units="mV")
    gpu = SynapticCUDA(graph(), config, input_ids=("0", "1"), weight_units="mV")
    stream = device.cuda.Stream(non_blocking=True)
    downstream = 0
    for tick in range(600):
        jumps = np.zeros(6)
        if tick in (0, 1, 2, 30, 75, 100, 180, 200, 230):
            jumps[0] = 68.75
        if tick in (25, 75, 160, 190):
            jumps[1] = 68.75
        cpu.step(jumps)
        # The engine must continue to own its stream in other CuPy contexts.
        with stream:
            gpu.step(jumps)
            equal(cpu, gpu)
        downstream += int(cpu.snapshot()["spikes"][3])
        if tick == 4:
            # A spike is in the delay buffer for positive-delay configurations.
            saved_cpu = json.loads(json.dumps(cpu.snapshot()))
            saved_gpu = json.loads(json.dumps(gpu.snapshot()))
            gpu.restore(saved_cpu)
            cpu.restore(saved_gpu)
    assert downstream > 0
    checkpoint = gpu.snapshot()
    corrupted = {**checkpoint, "voltage_mv": [float("nan")] * 6}
    with pytest.raises(ValueError):
        gpu.restore(corrupted)
    assert gpu.snapshot() == checkpoint
    with pytest.raises(ValueError, match="declared input"):
        gpu.step([0, 0, 1, 0, 0, 0])
    assert gpu.snapshot() == checkpoint


def test_silencing_queued_events_and_nonfinite_step_rollback(device):
    cpu = SynapticCPU(graph(), input_ids=("0", "1"), weight_units="mV")
    gpu = SynapticCUDA(graph(), input_ids=("0", "1"), weight_units="mV")
    for tick in range(200):
        jumps = np.zeros(6)
        jumps[0] = 68.75 if tick < 10 else 0
        if tick == 5:
            cpu.set_silenced((0,), True)
            gpu.set_silenced((0,), True)
        if tick == 50:
            cpu.set_silenced((0,), False)
            gpu.set_silenced((0,), False)
        cpu.step(jumps)
        gpu.step(jumps)
        equal(cpu, gpu)
    before = gpu.snapshot()
    with pytest.raises(ValueError, match="finite"):
        gpu.step([float("inf"), 0, 0, 0, 0, 0])
    assert gpu.snapshot() == before
    # Exercise a kernel-detected overflow, not just host input validation.
    overflow_graph = Connectome(
        "overflow-fixture",
        ("a", "b", "c"),
        (Synapse(0, 2, 1.7e308), Synapse(1, 2, 1.7e308)),
        {},
        {},
        {},
    )
    overflow = SynapticCUDA(
        overflow_graph, SynapticLIFConfig(delay_ms=0), input_ids=("a", "b"), weight_units="mV"
    )
    overflow.step([68.75, 68.75, 0])
    before = overflow.snapshot()
    with pytest.raises(ValueError, match="nonfinite"):
        overflow.step([0, 0, 0])
    assert overflow.snapshot() == before
