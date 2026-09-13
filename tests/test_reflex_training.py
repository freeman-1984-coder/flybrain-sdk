"""The example's controller sees neural features, never the answer label."""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "train_reflex", Path(__file__).resolve().parents[1] / "examples/train_reflex.py"
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def test_labels_do_not_enter_features():
    a = [{"rates_hz": [95, 12], "target": 0, "obstacle_side": 1}]
    b = [{"rates_hz": [95, 12], "target": 1, "obstacle_side": 0}]
    np.testing.assert_array_equal(EXAMPLE.features(a), EXAMPLE.features(b))


def test_reward_changes_learned_action_without_changing_features():
    x = np.tile([[1, 0, 1], [0, 1, 1]], (32, 1))
    target = np.tile([1, 0], 32)
    for labels in (target, 1 - target):
        weights, curve = EXAMPLE.train(x, labels)
        action = EXAMPLE.probabilities(np, x, weights) >= 0.5
        np.testing.assert_array_equal(action, labels)
        assert len(curve) == 24
