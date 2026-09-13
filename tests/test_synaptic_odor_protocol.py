"""Controls must share noise and preserve exact stimulus replay."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_protocol():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "synaptic_odor_protocol", scripts / "probe_synaptic_odor.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_controls_share_each_enabled_neurons_random_inputs():
    protocol = load_protocol()
    generators = [np.random.default_rng(20260913) for _ in range(4)]
    counts = np.zeros(68, dtype=int)
    for _ in range(1000):
        both, left, right, off = [
            protocol.input_events(rng, 35, 33, 0.015, enable_left, enable_right)
            for rng, (enable_left, enable_right) in zip(
                generators, [(True, True), (True, False), (False, True), (False, False)]
            )
        ]
        np.testing.assert_array_equal(both, left | right)
        assert not left[35:].any() and not right[:35].any() and not off.any()
        counts += both
    assert (counts > 0).all()
    assert all(rng.bit_generator.state == generators[0].bit_generator.state for rng in generators)


def test_input_rng_json_restore_replays_future_stimuli():
    protocol = load_protocol()
    rng = np.random.default_rng(20260913)
    protocol.input_events(rng, 35, 33, 0.015, False, False)
    state = json.loads(json.dumps(rng.bit_generator.state))
    expected = [protocol.input_events(rng, 35, 33, 0.015, True, True) for _ in range(100)]
    rng.bit_generator.state = state
    actual = [protocol.input_events(rng, 35, 33, 0.015, True, True) for _ in range(100)]
    np.testing.assert_array_equal(actual, expected)
