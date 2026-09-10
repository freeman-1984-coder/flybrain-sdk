"""Cross-language replay uses recordings produced by the actual website worker."""

import json
import shutil
from pathlib import Path

import pytest

from flybrain import FlyBrain
from flybrain.experiment import replay_experiment

ROOT = Path(__file__).resolve().parents[1]


def recording(model="toy"):
    return json.loads((ROOT / f"tests/fixtures/browser-{model}.json").read_text())


@pytest.mark.parametrize("model", ["toy", "male-cns-escape-v1"])
def test_browser_recording_replays_offline(model, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("replay unexpectedly used network")

    monkeypatch.setattr("flybrain.registry.urlopen", no_network)
    if model != "toy":
        folder = tmp_path / model
        folder.mkdir()
        shutil.copyfile(ROOT / f"models/{model}/model.json", folder / "model.json")
    brain = replay_experiment(recording(model), cache_dir=tmp_path)
    assert brain.state.tick == 101
    assert max(brain.action().to_dict().values()) > 0
    checkpoint = tmp_path / "checkpoint.json"
    brain.save(checkpoint)
    # Recording ends with pending stimulation; the returned brain can continue.
    restored = FlyBrain.restore(checkpoint)
    assert brain.step(80) == restored.step(80)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(model_id="https://attacker.invalid/model"),
        lambda d: d.update(model_sha256="0" * 64),
        lambda d: d.update(model_asset_sha256="0" * 64),
        lambda d: d.update(duration_ticks=1_000_001),
        lambda d: d["commands"][1].update(tick=-1),
        lambda d: d["commands"][2].update(tick=0),
        lambda d: d["commands"][0].update(op="exec"),
        lambda d: d["commands"][1].update(enabled=1),
        lambda d: d["commands"][0].update(strength=float("nan")),
        lambda d: d["expected"]["state"]["voltage"].pop(),
        lambda d: d["expected"]["state"]["spikes"].__setitem__(0, 1),
        lambda d: d["expected"]["state"]["rates_hz"].__setitem__(0, 999),
        lambda d: d["expected"]["action"].update(walk=0.999),
        lambda d: d.pop("expected"),
    ],
)
def test_bad_recording_is_rejected(mutate):
    data = recording()
    mutate(data)
    with pytest.raises(ValueError):
        replay_experiment(data)
