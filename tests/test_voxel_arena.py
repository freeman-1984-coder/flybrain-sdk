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
