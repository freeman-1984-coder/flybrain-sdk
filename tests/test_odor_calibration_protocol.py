"""Timing and analyses must distinguish a transient response from persistent bias."""

import importlib.util
import sys
from pathlib import Path

import pytest


def protocol():
    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "odor_calibration", scripts / "calibrate_fullbrain_odor.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_counterbalanced_schedule_keeps_same_clock_and_unstimulated_recovery():
    p = protocol()
    for order in p.ORDERS:
        phases = p.phase_schedule(0.1, order)
        assert [(s["start_tick"], s["end_tick"]) for s in phases] == [
            (0, 500),
            (500, 2000),
            (2000, 4000),
            (4000, 5500),
            (5500, 8000),
        ]
        assert [s["side"] for s in phases] == [None, order[0], None, order[1], None]
        assert sum(s["duration_ms"] for s in phases) == 800


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf"), True, 0.3])
def test_invalid_timestep_cannot_silently_change_the_protocol(dt):
    with pytest.raises(ValueError):
        protocol().phase_schedule(dt, ("left", "right"))


def test_duplicate_side_cannot_be_mistaken_for_counterbalancing():
    with pytest.raises(ValueError):
        protocol().phase_schedule(0.1, ("left", "left"))


def test_summary_preserves_early_right_response_despite_later_left_bias():
    run = {
        "phases": [
            {
                "name": "first",
                "side": "right",
                "duration_ms": 150,
                "first_50ms_spikes": {"DNa02_left": 1, "DNa02_right": 2},
                "spikes": {"DNa02_left": 6, "DNa02_right": 3},
            },
            {
                "name": "recovery_first",
                "side": None,
                "duration_ms": 200,
                "tail_window_ms": 100,
                "last_100ms_spikes": {"DNa02_left": 0, "DNa02_right": 0},
                # A decaying rate estimate alone must not imply continued spikes.
                "mean_rates_hz": {"DNa02_left": 8.0, "DNa02_right": 4.0},
            },
        ]
    }
    result = protocol().summarize_run(run)
    assert result["stimuli"][0] == {
        "phase": "first",
        "side": "right",
        "onset_ipsilateral_minus_contralateral_hz": 20.0,
        "later_ipsilateral_minus_contralateral_hz": -40.0,
        "total_right_minus_left_spikes": -3,
    }
    assert result["recovery"][0]["new_tail_spikes"] == {"DNa02_left": 0, "DNa02_right": 0}


def test_ipsilateral_metric_changes_reference_with_stimulated_side():
    result = protocol().summarize_run(
        {
            "phases": [
                {
                    "name": "second",
                    "side": "left",
                    "duration_ms": 150,
                    "first_50ms_spikes": {"DNa02_left": 2, "DNa02_right": 1},
                    "spikes": {"DNa02_left": 3, "DNa02_right": 6},
                }
            ]
        }
    )
    assert result["stimuli"][0]["onset_ipsilateral_minus_contralateral_hz"] == 20
    assert result["stimuli"][0]["later_ipsilateral_minus_contralateral_hz"] == -40
