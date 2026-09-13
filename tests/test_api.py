import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from flybrain import BackendUnavailableError, Connectome, FlyBrain, LIFConfig, Stimulus


def test_offline_and_resting_brain(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("demo tried to access network")

    monkeypatch.setattr(socket, "socket", blocked)
    brain = FlyBrain.load()
    assert len(brain.model.neuron_ids) == 12
    assert brain.step(100).spikes == (False,) * 12
    assert set(brain.action().to_dict().values()) == {0.0}


@pytest.mark.parametrize(
    "channel,active",
    [
        ("food", {"walk"}),
        ("looming_left", {"turn_right", "jump"}),
        ("looming_right", {"turn_left", "jump"}),
        ("touch", {"jump"}),
    ],
)
def test_sensory_to_motor_routing(channel, active):
    brain = FlyBrain.load()
    brain.stimulate(Stimulus(channel, duration_ms=100))
    brain.step(100)
    action = brain.action().to_dict()
    assert {key for key, value in action.items() if value > 0} == active
    assert all(0 <= v <= 1 for v in action.values())
    before = brain.state
    assert brain.action() == brain.action()
    assert brain.state == before


def test_chunking_equivalence():
    a, b = FlyBrain.load(), FlyBrain.load()
    a.stimulate("food", duration_ms=113.2)
    b.stimulate("food", duration_ms=113.2)
    a.step(150)
    for _ in range(150):
        b.step()
    assert a.state == b.state


def test_nondefault_dt_and_clear_stimuli():
    brain = FlyBrain.load(config=LIFConfig(dt_ms=0.5))
    brain.stimulate("food", duration_ms=100)
    brain.clear_stimuli()
    assert not any(brain.step(10).spikes)
    assert brain.state.time_ms == 5


def test_wasm_slot_fails_explicitly():
    with pytest.raises(BackendUnavailableError, match="not implemented"):
        FlyBrain.load(backend="wasm")


def test_unknown_backend_and_raw_model():
    with pytest.raises(ValueError, match="unknown backend"):
        FlyBrain.load(backend="metal")
    with pytest.raises(ValueError, match="raw connectome"):
        FlyBrain.load("male-cns-v1.0")


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_reject_bad_step_counts(steps):
    with pytest.raises(ValueError):
        FlyBrain.load().step(steps)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": float("nan")},
        {"strength": -0.1},
        {"strength": 1.1},
        {"duration_ms": 0},
        {"duration_ms": float("inf")},
        {"strength": True},
    ],
)
def test_reject_invalid_stimuli(kwargs):
    with pytest.raises(ValueError):
        FlyBrain.load().stimulate("food", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt_ms": 0},
        {"tau_ms": -1},
        {"rest": 1},
        {"reset": 1},
        {"refractory_ms": -1},
        {"action_rate_hz": 0},
        {"dt_ms": float("nan")},
    ],
)
def test_reject_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LIFConfig(**kwargs)


def test_local_model_and_large_ids(tmp_path):
    data = {
        "schema_version": 1,
        "name": "custom",
        "neuron_ids": ["720575940600000001"],
        "synapses": [],
        "sensory": {"food": ["720575940600000001"]},
        "motor": {"walk": ["720575940600000001"]},
    }
    model = tmp_path / "custom.json"
    model.write_text(json.dumps(data))
    brain = FlyBrain.load(model)
    brain.stimulate("food", duration_ms=100)
    assert brain.step(100).tick == 100
    assert brain.action().walk > 0
    assert brain.model.neuron_ids[0] == "720575940600000001"


def test_model_rejects_duplicate_ids_and_dangling_edges():
    data = Connectome.load().to_dict()
    data["neuron_ids"].append(data["neuron_ids"][0])
    with pytest.raises(ValueError, match="unique"):
        Connectome.from_dict(data)
    data = Connectome.load().to_dict()
    data["synapses"][0]["pre"] = "missing"
    with pytest.raises(ValueError, match="unknown neuron"):
        Connectome.from_dict(data)


def test_model_ports_and_observation_are_immutable():
    brain = FlyBrain.load()
    with pytest.raises(TypeError):
        brain.model.sensory["food"] = (1,)
    before = brain.state
    brain.stimulate("food").step(10)
    assert before.tick == 0
    assert before.voltage == (0.0,) * 12


def test_quickstart_runs():
    example = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
    result = subprocess.run(
        [sys.executable, str(example)], capture_output=True, text=True, check=True
    )
    assert "Checkpoint continuation: identical" in result.stdout
