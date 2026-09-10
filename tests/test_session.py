import copy
import json
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from flybrain import CheckpointError, FlyBrain
from flybrain.demos import DodgeArena, PulseScore, make_demo, render_wav
from flybrain.session import LinearEncoder, Session, replay

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "models/male-cns-escape-v1/model.json"


@pytest.mark.parametrize(
    "demo,factory", [("dodge", DodgeArena.from_snapshot), ("tones", PulseScore.from_snapshot)]
)
@pytest.mark.parametrize("model", ["toy", REAL])
def test_full_feedback_replay_and_midrun_restore(demo, factory, model, tmp_path):
    session = make_demo(demo, model=model)
    session.brain.stimulate("looming_left", duration_ms=120)
    for _ in range(3):
        session.step()
    restored = Session.restore(session.save(tmp_path / "session.json"), environment_factory=factory)
    for _ in range(50):
        assert session.step() == restored.step()
    recording = json.loads(json.dumps(session.record(80)))
    result = replay(recording, environment_factory=factory)
    assert result.summary() == session.summary()
    assert max(abs(v) for f in recording["frames"] for v in f["applied"].values()) > 0


def test_fractional_clock_has_no_accumulated_tick_loss():
    session = make_demo("tones", period_ms=16.6666667)
    ticks = [session.step().brain_tick for _ in range(180)]
    assert ticks == [int((i + 1) * Fraction("16.6666667")) for i in range(180)]
    assert set(np.diff([0] + ticks)) == {16, 17}
    with pytest.raises(ValueError, match="at least"):
        make_demo("tones", period_ms=0.5)


def test_snapshot_is_detached_and_checks_clock():
    brain = FlyBrain.load()
    brain.stimulate("food", duration_ms=100)
    snapshot = brain.snapshot()
    clone = FlyBrain.from_snapshot(snapshot)
    snapshot["state"]["voltage"][0] = -999
    assert brain.step(30) == clone.step(30)
    session = make_demo("dodge")
    data = session.snapshot()
    data["environment"]["x"] = 0
    assert session.environment.x == 0.5
    session.brain.step()
    with pytest.raises(ValueError, match="outside"):
        session.step()


def test_invalid_projection_does_not_queue_partial_current():
    session = make_demo("tones")
    session.encoder = LinearEncoder({"pulse_a": {"sense.food": 2, "missing": 2}})
    before = session.brain.snapshot()
    with pytest.raises(ValueError):
        session.step()
    assert session.brain.snapshot() == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["frames"][1]["applied"].update(tone_a=99),
        lambda d: d["frames"][0].update(brain_tick=True),
        lambda d: d["final"]["brain"]["state"]["spikes"].__setitem__(0, 1),
        lambda d: d["initial"].update(schema_version=True),
        lambda d: d["initial"].update(frame_index=99),
        lambda d: d["initial"]["environment"].update(type="arbitrary-import"),
    ],
)
def test_corrupted_recording_fails(mutate):
    recording = make_demo("tones").record(4)
    mutate(recording)
    with pytest.raises((ValueError, CheckpointError)):
        replay(recording, environment_factory=PulseScore.from_snapshot)


def test_neural_step_budget_prevents_huge_recording():
    recording = make_demo("tones").record(4)
    recording["initial"]["period_ms"] = 1e20
    with pytest.raises(ValueError, match="max_steps"):
        replay(recording, environment_factory=PulseScore.from_snapshot)


def pcm(path):
    with wave.open(str(path)) as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 16000
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")


def test_audio_comes_from_neural_output_and_resumes_without_phase_reset(tmp_path):
    session = make_demo("tones")
    initial = session.snapshot()
    first = session.record(60)
    second = Session.from_snapshot(
        session.snapshot(), environment_factory=PulseScore.from_snapshot
    ).record(60)
    entire = Session.from_snapshot(initial, environment_factory=PulseScore.from_snapshot).record(
        120
    )
    a = pcm(render_wav(first, tmp_path / "a.wav", fade_out=False))
    b = pcm(render_wav(second, tmp_path / "b.wav", fade_out=False))
    whole = pcm(render_wav(entire, tmp_path / "whole.wav", fade_out=False))
    assert np.array_equal(np.concatenate([a, b]), whole)
    assert len(whole) == 120 * 320
    assert np.max(np.abs(whole)) > 100
    assert np.max(np.abs(whole)) < 12000
    silent = make_demo("tones")
    silent.brain.intervene.silence(["motor.turn_left", "motor.turn_right"])
    silent_audio = pcm(render_wav(silent.record(120), tmp_path / "silent.wav"))
    assert not silent_audio.any()


def test_audio_rejects_timestamp_corruption_before_writing(tmp_path):
    recording = make_demo("tones").record(4)
    bad = copy.deepcopy(recording)
    bad["frames"][0]["elapsed_ms"] = 1e20
    with pytest.raises(ValueError):
        render_wav(bad, tmp_path / "bad.wav")
    assert not (tmp_path / "bad.wav").exists()


def test_requested_and_applied_actions_capture_wall_clipping():
    env = DodgeArena()
    env.x = 1
    applied = env.apply({"steer": 0.7}, 20)
    assert applied == {"steer": 0.0}
    assert env.observe()["player_x"] == 1


def test_restore_rejects_invalid_adapter_neuron_ids():
    data = make_demo("tones").snapshot()
    data["readout"]["channels"]["tone_a"]["weights"] = {"missing": 1}
    with pytest.raises(CheckpointError):
        Session.from_snapshot(data, environment_factory=PulseScore.from_snapshot)


def test_replay_viewer_escapes_recorded_text_and_omits_graph(tmp_path):
    from flybrain.viewer import write_replay_html

    recording = make_demo("dodge").record(2)
    recording["initial"]["brain"]["model"]["name"] = "</script><script>bad()</script>"
    text = write_replay_html(recording, tmp_path / "viewer.html").read_text()
    assert text.count("<script>") == 1
    assert "<script>bad()" not in text
    assert "synapses" not in text
    assert "recorded Python run" in text
    assert "Permission is hereby granted" in text
