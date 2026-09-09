import json

import pytest

from flybrain import CheckpointError, FlyBrain, LIFConfig


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_exact_continuation_with_active_overlapping_inputs(tmp_path, dt):
    brain = FlyBrain.load(config=LIFConfig(dt_ms=dt))
    brain.stimulate("food", duration_ms=111.2)
    brain.stimulate("looming_left", strength=0.8, duration_ms=90)
    brain.step(13)
    path = brain.save(tmp_path / "brain.json")
    other = FlyBrain.restore(path)
    assert brain.state == other.state
    for _ in range(200):
        assert brain.step() == other.step()
        assert brain.action() == other.action()


def test_checkpoint_is_self_contained(tmp_path):
    brain = FlyBrain.load()
    brain.stimulate("touch", duration_ms=100)
    path = brain.save(tmp_path / "brain.json")
    data = json.loads(path.read_text())
    assert data["model"]["name"] == "toy-v1"
    assert data["pending_stimuli"][0]["remaining_steps"] == 100
    assert FlyBrain.restore(path).step(100) == brain.step(100)
    brain.save(path)  # safe replacement
    assert not list(tmp_path.glob(".brain.json.*"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version=999),
        lambda d: d.update(schema_version=True),
        lambda d: d.update(dynamics_revision="future"),
        lambda d: d.update(model_sha256="bad"),
        lambda d: d["state"].update(voltage=[0]),
        lambda d: d["state"].update(tick=-1),
        lambda d: d["state"]["voltage"].__setitem__(0, float("nan")),
        lambda d: d["state"]["rates_hz"].__setitem__(0, -1),
        lambda d: d["state"]["spikes"].__setitem__(0, 1),
        lambda d: d["state"]["refractory"].__setitem__(0, 999),
        lambda d: d.update(
            pending_stimuli=[{"channel": "missing", "strength": 1, "remaining_steps": 2}]
        ),
    ],
)
def test_corrupt_checkpoint_rejected(tmp_path, mutation):
    path = FlyBrain.load().save(tmp_path / "bad.json")
    data = json.loads(path.read_text())
    mutation(data)
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError):
        FlyBrain.restore(path)


def test_malformed_json_and_missing_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    with pytest.raises(CheckpointError):
        FlyBrain.restore(path)
    with pytest.raises(FileNotFoundError):
        FlyBrain.restore(tmp_path / "missing.json")
