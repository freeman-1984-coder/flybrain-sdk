import json
from pathlib import Path

import pytest

from flybrain import Connectome, FlyBrain

MODEL = Path(__file__).resolve().parents[1] / "models" / "male-cns-escape-v1" / "model.json"


def test_real_model_anatomy_and_raw_counts():
    data = json.loads(MODEL.read_text())
    assert data["report"]["neuron_count"] == 313
    assert data["report"]["edge_count"] == 20607
    assert data["report"]["raw_synapse_count"] == 79112
    assert sum(e["count"] for e in data["raw_connections"]) == 79112
    assert data["model"]["motor"]["jump"] == ["10001", "10010"]
    assert set(data["model"]["motor"]["jump"]).isdisjoint(data["model"]["sensory"]["looming_left"])


def test_real_model_stimulation_and_recovery():
    brain = FlyBrain.load(MODEL)
    assert brain.neurons.select(cell_type="DNp01") == ("10001", "10010")
    assert len(brain.neurons.select(cell_type="LC4")) == 126
    assert not any(brain.step(100).spikes)
    brain.stimulate("looming_left", duration_ms=100)
    brain.step(100)
    assert 0 < brain.action().jump <= 1
    brain.step(900)
    assert brain.action().jump < 1e-6
    assert not any(brain.state.spikes)


def test_gf_edge_ablation_removes_output():
    data = json.loads(MODEL.read_text())["model"]
    gf = set(data["motor"]["jump"])
    data["synapses"] = [e for e in data["synapses"] if e["post"] not in gf]
    brain = FlyBrain.load(Connectome.from_dict(data))
    brain.stimulate("looming_left", duration_ms=200)
    for _ in range(200):
        brain.step()
        assert brain.action().jump == 0


def test_real_bundle_checkpoint_and_config(tmp_path):
    brain = FlyBrain.load(MODEL)
    brain.stimulate("looming_right", duration_ms=100)
    brain.step(30)
    other = FlyBrain.restore(brain.save(tmp_path / "state.json"))
    for _ in range(120):
        assert brain.step() == other.step()


def test_corrupt_bundle_rejected(tmp_path):
    data = json.loads(MODEL.read_text())
    data["model"]["synapses"][0]["weight"] += 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprint"):
        FlyBrain.load(path)
