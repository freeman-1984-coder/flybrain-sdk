import json

import numpy as np
import pytest

from flybrain import Connectome, Synapse
from flybrain.experimental.synaptic import SynapticCPU, SynapticLIFConfig


def model(weight=55.0):
    return Connectome(
        "fixture", ("input", "output", "isolated"), (Synapse(0, 1, weight),), {}, {}, {}
    )


def brain(**kwargs):
    return SynapticCPU(model(), input_ids=("input",), weight_units="mV", **kwargs)


def test_exact_transmission_delay_and_sensory_input_schedule():
    b = brain()
    b.step([68.75, 0, 0])
    assert b.snapshot()["spikes"] == [False, False, False]
    b.step([0, 0, 0])
    assert b.snapshot()["spikes"] == [True, False, False]
    # Source spike at t=0.1 ms arrives at 1.9 ms, not before.
    for _ in range(17):
        b.step([0, 0, 0])
        assert b.snapshot()["synaptic_mv"][1] == 0
    b.step([0, 0, 0])
    assert b.snapshot()["synaptic_mv"][1] == 55
    assert b.snapshot()["voltage_mv"][1] == -52
    b.step([0, 0, 0])
    assert b.snapshot()["voltage_mv"][1] > -52


def test_synaptic_propagation_without_external_postsynaptic_current():
    b = brain()
    counts = np.zeros(3, dtype=int)
    for tick in range(1000):
        b.step([68.75 if tick < 500 and tick % 30 == 0 else 0.0, 0.0, 0.0])
        counts += b.snapshot()["spikes"]
    assert counts[0] > 0 and counts[1] > 0 and counts[2] == 0


def test_inflight_delay_buffer_survives_json_checkpoint():
    b = brain()
    b.step([68.75, 0, 0])
    b.step([0, 0, 0])
    saved = json.loads(json.dumps(b.snapshot()))
    restored = brain()
    restored.restore(saved)
    for _ in range(150):
        b.step([0, 0, 0])
        restored.step([0, 0, 0])
        assert b.snapshot() == restored.snapshot()


def test_checkpoint_rejects_other_graph_and_invalid_state_atomically():
    b = brain()
    original = b.snapshot()
    different = SynapticCPU(model(56.0), input_ids=("input",), weight_units="mV")
    with pytest.raises(ValueError, match="mismatch"):
        different.restore(original)
    for key, value in [
        ("voltage_mv", [float("nan"), 0, 0]),
        ("voltage_mv", [True, 0, 0]),
        ("history", [[True]]),
        ("last_spike_tick", [1, 1, 1]),
    ]:
        corrupted = {**original, key: value}
        with pytest.raises(ValueError):
            b.restore(corrupted)
        assert b.snapshot() == original


def test_silencing_stops_new_spikes_but_preserves_already_queued_events():
    b = brain()
    b.step([68.75, 0, 0])
    b.step([0, 0, 0])
    b.set_silenced((0,), True)
    spikes = []
    for _ in range(150):
        b.step([68.75, 0, 0])
        spikes.append(b.snapshot()["spikes"])
    assert not np.asarray(spikes)[:, 0].any()
    assert np.asarray(spikes)[:, 1].any()


def test_units_and_declared_inputs_cannot_be_silently_bypassed():
    with pytest.raises(ValueError, match="mV"):
        SynapticCPU(model(), weight_units="normalized")
    b = brain()
    saved = b.snapshot()
    with pytest.raises(ValueError, match="declared input"):
        b.step([0, 68.75, 0])
    assert b.snapshot() == saved
    for indices in [(-1,), (3,), (True,), (0.5,)]:
        with pytest.raises(ValueError):
            b.set_silenced(indices, True)


def test_nearly_equal_time_constants_approach_exact_equal_limit():
    exact = brain(config=SynapticLIFConfig(synapse_tau_ms=20))
    nearby = brain(config=SynapticLIFConfig(synapse_tau_ms=20 + 1e-10))
    for tick in range(200):
        jumps = [68.75 if tick == 0 else 0, 0, 0]
        exact.step(jumps)
        nearby.step(jumps)
        np.testing.assert_allclose(
            exact.snapshot()["voltage_mv"], nearby.snapshot()["voltage_mv"], atol=1e-9, rtol=0
        )


@pytest.mark.parametrize(
    "params", [dict(dt_ms=1), dict(delay_ms=-1), dict(synapse_tau_ms=0), dict(dt_ms=float("nan"))]
)
def test_invalid_dynamics_rejected(params):
    with pytest.raises(ValueError):
        SynapticLIFConfig(**params)
