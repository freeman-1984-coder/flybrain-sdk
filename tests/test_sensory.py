import numpy as np
import pytest

from flybrain import Connectome, LIFConfig, Stimulus
from flybrain.sensory import SensoryEncoder


def test_pulse_duration_rounds_up_and_overlaps_add():
    encoder = SensoryEncoder(Connectome.load(), LIFConfig(dt_ms=1))
    encoder.add(Stimulus("food", strength=0.5, duration_ms=1.1))
    encoder.add(Stimulus("food", strength=0.25, duration_ms=1))
    assert encoder.current()[0] == pytest.approx(1.5)
    assert encoder.current()[0] == pytest.approx(1.0)
    assert np.all(encoder.current() == 0)
    assert encoder.snapshot() == []


def test_unknown_channel_does_not_queue_input():
    encoder = SensoryEncoder(Connectome.load(), LIFConfig())
    with pytest.raises(ValueError, match="unknown sensory"):
        encoder.add(Stimulus("vision"))
    assert encoder.snapshot() == []
