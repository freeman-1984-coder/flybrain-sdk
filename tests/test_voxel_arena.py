import json

import pytest

from flybrain.experimental.voxel_arena import VoxelArena


def test_food_cannot_steer_the_body_without_neural_commands():
    a = VoxelArena(food=(4, 0.5, 2))
    b = VoxelArena(food=(-6, 0.5, -6))
    assert a.observe() != b.observe()
    assert set(a.observe()) == {"odor_left", "odor_right"}
    initial = a.scene()["position"]
    for _ in range(10):
        a.apply({"speed": 0, "turn": 0})
    assert a.scene()["position"] == initial
    for _ in range(100):
        action = {"speed": 2, "turn": 0.7}
        a.apply(action)
        b.apply(action)
        assert a.scene()["position"] == b.scene()["position"]
        assert a.yaw == b.yaw


def test_solid_blocks_stop_fast_body_without_tunneling():
    arena = VoxelArena(blocks=((-2, -2),))
    for _ in range(20):
        arena.apply({"speed": 3, "turn": 0}, 100)
    assert -2.8 < arena.x <= -2.68
    assert arena.contacts > 0


def test_world_json_checkpoint_replays_odor_contacts_and_pose():
    arena = VoxelArena()
    for _ in range(40):
        arena.apply({"speed": 1, "turn": 0.1})
    restored = VoxelArena.from_snapshot(json.loads(json.dumps(arena.snapshot())))
    for _ in range(100):
        action = {"speed": 3, "turn": 0}
        assert arena.apply(action) == restored.apply(action)
        assert arena.observe() == restored.observe()
        assert arena.snapshot() == restored.snapshot()
    before = arena.snapshot()
    with pytest.raises(ValueError):
        arena.apply({"speed": 2, "turn": float("nan")})
    assert arena.snapshot() == before


def test_mirrored_world_swaps_antennae_and_preserves_mirrored_motion():
    # Reflect about z=-2, the initial forward axis. This rules out a built-in
    # left preference in antenna geometry or the body's turn convention.
    a = VoxelArena(food=(-1.6, 0.5, -0.6), blocks=())
    b = VoxelArena(food=(-1.6, 0.5, -3.4), blocks=())
    assert a.observe()["odor_right"] > a.observe()["odor_left"]
    assert b.observe()["odor_left"] > b.observe()["odor_right"]
    for _ in range(100):
        a.apply({"speed": 1, "turn": 0.8})
        b.apply({"speed": 1, "turn": -0.8})
        assert a.x == pytest.approx(b.x, abs=1e-12)
        assert a.z + b.z == pytest.approx(-4, abs=1e-12)
        assert a.yaw == pytest.approx(-b.yaw, abs=1e-12)
        assert a.observe()["odor_left"] == pytest.approx(b.observe()["odor_right"], abs=1e-12)
        assert a.observe()["odor_right"] == pytest.approx(b.observe()["odor_left"], abs=1e-12)
