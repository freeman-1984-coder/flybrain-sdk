import copy
from pathlib import Path

import pytest

from flybrain.demos import make_demo
from flybrain.external import ExternalController
from flybrain.session import compare

ROOT = Path(__file__).resolve().parents[1]


def controller(model="toy"):
    demo = make_demo("dodge", model=model)
    return ExternalController(demo.brain, demo.encoder, demo.readout)


@pytest.mark.parametrize("model", ["toy", ROOT / "models/male-cns-escape-v1/model.json"])
def test_external_world_matches_session_and_continues_checkpoint(model):
    expected = make_demo("dodge", model=model)
    external = controller(model)
    world = make_demo("dodge").environment
    for seq in range(150):
        action = external.offer(seq, world.observe())
        before = external.brain.snapshot()
        assert external.offer(seq, world.observe()) == action
        assert external.brain.snapshot() == before
        applied = world.apply(action["requested"], 20)
        receipt = external.acknowledge(seq, applied)
        assert external.acknowledge(seq, applied) == receipt
        frame = expected.step()
        compare(action["requested"], frame.requested)
        compare(applied, frame.applied)
        compare(world.snapshot(), frame.environment)
        if seq == 74:
            external = ExternalController.from_snapshot(external.snapshot())
    assert external.brain.snapshot() == expected.brain.snapshot()


def test_protocol_rejects_conflicts_without_changing_neural_time():
    c = controller()
    observation = {"danger_left": 1, "danger_right": 0}
    with pytest.raises(ValueError):
        c.offer(True, observation)
    c.offer(0, observation)
    before = c.brain.snapshot()
    for seq, obs in [(1, observation), (0, {**observation, "danger_left": 0})]:
        with pytest.raises(ValueError):
            c.offer(seq, obs)
    for seq, applied in [(1, {"steer": 0}), (0, {"wrong": 0}), (0, {"steer": float("nan")})]:
        with pytest.raises(ValueError):
            c.acknowledge(seq, applied)
    with pytest.raises(ValueError, match="boundary"):
        c.snapshot()
    assert c.brain.snapshot() == before
    c.acknowledge(0, {"steer": 0})
    with pytest.raises(ValueError, match="conflicting"):
        c.acknowledge(0, {"steer": 1})
    c.offer(1, observation)
    assert c.acknowledge(0, {"steer": 0})["seq"] == 0
    assert c.pending["seq"] == 1


def test_bad_observation_and_clock_do_not_advance():
    c = controller()
    before = c.brain.snapshot()
    with pytest.raises(ValueError):
        c.offer(0, {"danger_left": float("inf"), "danger_right": 0})
    assert c.brain.snapshot() == before
    c.brain.step()
    with pytest.raises(ValueError, match="clock"):
        c.offer(0, {"danger_left": 1, "danger_right": 0})


def test_failure_after_integration_requires_restore():
    c = controller()
    checkpoint = c.snapshot()

    class Broken:
        def decode(self, brain):
            raise RuntimeError("decoder failed")

    c.readout = Broken()
    with pytest.raises(RuntimeError):
        c.offer(0, {"danger_left": 1, "danger_right": 0})
    assert c.brain.progress.tick == 20
    with pytest.raises(ValueError, match="faulted"):
        c.offer(0, {"danger_left": 1, "danger_right": 0})
    assert ExternalController.from_snapshot(checkpoint).brain.progress.tick == 0


@pytest.mark.parametrize(
    "key,value", [("next_seq", 10), ("base_tick", True), ("schema_version", True)]
)
def test_checkpoint_clock_validation(key, value):
    data = copy.deepcopy(controller().snapshot())
    data[key] = value
    with pytest.raises(ValueError):
        ExternalController.from_snapshot(data)
