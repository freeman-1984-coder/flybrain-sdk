import json
from dataclasses import replace

import pytest

from flybrain import ChannelAction, CheckpointError, Connectome, FlyBrain, LIFConfig


def test_selection_uses_annotations():
    model = replace(
        Connectome.load(),
        annotations={
            "sense.food": {"cell_type": "input", "side": "L"},
            "sense.touch": {"cell_type": "input", "side": "R"},
        },
    )
    brain = FlyBrain.load(model)
    assert brain.neurons.select(cell_type="input") == ("sense.food", "sense.touch")
    assert brain.neurons.select(cell_type="input", side="L") == ("sense.food",)
    with pytest.raises(ValueError, match="no annotation"):
        brain.neurons.select(region="eye")
    with pytest.raises(ValueError, match="no neurons"):
        brain.neurons.select(side="missing")
    with pytest.raises(TypeError):
        brain.model.annotations["sense.food"]["side"] = "R"
    assert Connectome.from_dict(model.to_dict()).fingerprint == model.fingerprint


def test_direct_drive_equals_named_input_and_supports_inhibition():
    direct, named = FlyBrain.load(), FlyBrain.load()
    direct.drive.current(["sense.food"], amplitude=2, duration_ms=100)
    named.stimulate("food", duration_ms=100)
    assert direct.step(120) == named.step(120)
    direct = FlyBrain.load()
    direct.stimulate("food", duration_ms=100)
    direct.drive.current(["sense.food"], amplitude=-2, duration_ms=100)
    assert not any(direct.step(100).spikes)
    assert all(v == 0 for v in direct.state.voltage)


def test_selected_observation_and_action_avoid_full_export(monkeypatch):
    brain = FlyBrain.load()

    def blocked():
        raise AssertionError("unnecessary full-state export")

    monkeypatch.setattr(brain._backend, "observe", blocked)
    brain.stimulate("food", duration_ms=100)
    assert brain.advance(duration_ms=100).tick == 100
    observed = brain.observe(["motor.walk"], fields=["rates_hz"])
    assert observed.neuron_ids == ("motor.walk",)
    assert set(observed.values) == {"rates_hz"}
    assert observed.values["rates_hz"][0] > 0
    assert brain.action().walk > 0
    with pytest.raises(TypeError):
        observed.values["rates_hz"] = ()
    before = observed.to_dict()
    brain.advance(duration_ms=100)
    assert observed.to_dict() == before


def test_arbitrary_readout_does_not_change_graph_and_restores(tmp_path):
    brain = FlyBrain.load()
    fingerprint = brain.model.fingerprint
    brain.bind_readout({"volume": ["motor.walk"], "flash": ["motor.jump"]}, scale_hz=200)
    brain.stimulate("food", duration_ms=100)
    brain.advance(duration_ms=30)
    other = FlyBrain.restore(brain.save(tmp_path / "custom.json"))
    assert brain.model.fingerprint == fingerprint
    assert isinstance(brain.action(), ChannelAction)
    assert set(brain.action()) == {"volume", "flash"}
    for _ in range(110):
        assert brain.step() == other.step()
        assert brain.action() == other.action()
    assert brain.action()["volume"] > 0
    assert brain.action()["flash"] == 0
    before = brain.action()
    with pytest.raises(ValueError):
        brain.bind_readout({"invalid": ["missing"]})
    assert brain.action() == before


def test_model_supports_non_motor_channels_and_empty_readout():
    brain = FlyBrain.load(replace(Connectome.load(), motor={"synth": (8,)}))
    brain.stimulate("food", duration_ms=100)
    brain.advance(duration_ms=100)
    assert brain.action()["synth"] > 0
    brain.bind_readout({})
    assert brain.action().to_dict() == {}


def test_silencing_cuts_path_and_release_restores_response(tmp_path):
    brain = FlyBrain.load()
    brain.intervene.silence(["relay.food"])
    brain.drive.current(["sense.food"], amplitude=2, duration_ms=200)
    brain.advance(duration_ms=40)
    other = FlyBrain.restore(brain.save(tmp_path / "silenced.json"))
    for _ in range(70):
        assert brain.step() == other.step()
        assert brain.action().walk == 0
    assert brain.observe(["sense.food"], fields=["rates_hz"]).values["rates_hz"][0] > 0
    brain.intervene.silence(["relay.food"], enabled=False)
    brain.advance(duration_ms=60)
    assert brain.action().walk > 0
    assert other.action().walk == 0


def test_previously_emitted_spike_survives_silencing():
    brain = FlyBrain.load()
    brain.drive.current(["sense.food"], amplitude=20, duration_ms=1)
    assert brain.step().spikes[0]
    brain.intervene.silence(["sense.food"])
    state = brain.step()
    assert state.spikes[4]  # the relay receives the previous tick's emitted spike
    assert not state.spikes[0]


@pytest.mark.parametrize("duration", [0, -1, 0.75, True, float("nan")])
def test_direct_timing_rejects_ambiguous_intervals(duration):
    brain = FlyBrain.load(config=LIFConfig(dt_ms=0.5))
    with pytest.raises(ValueError):
        brain.advance(duration_ms=duration)
    with pytest.raises(ValueError):
        brain.drive.current(["sense.food"], amplitude=1, duration_ms=duration)
    assert brain.state.tick == 0


@pytest.mark.parametrize("ids", ["sense.food", ["missing"], [1], [], ["sense.food"] * 2])
def test_invalid_selections_do_not_queue_inputs(ids):
    brain = FlyBrain.load()
    with pytest.raises(ValueError):
        brain.drive.current(ids, amplitude=2, duration_ms=10)
    assert brain.drive.snapshot() == []


def test_schema_one_checkpoint_still_loads(tmp_path):
    brain = FlyBrain.load()
    brain.stimulate("touch", duration_ms=100)
    brain.step(10)
    path = brain.save(tmp_path / "old.json")
    data = json.loads(path.read_text())
    data["schema_version"] = 1
    del data["pending_currents"], data["readout"], data["state"]["silenced"]
    path.write_text(json.dumps(data))
    assert FlyBrain.restore(path).step(100) == brain.step(100)


@pytest.mark.parametrize("missing", ["pending_currents", "readout", "silenced"])
def test_schema_two_rejects_missing_dynamic_state(tmp_path, missing):
    path = FlyBrain.load().save(tmp_path / "bad.json")
    data = json.loads(path.read_text())
    del (data["state"] if missing == "silenced" else data)[missing]
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError):
        FlyBrain.restore(path)
