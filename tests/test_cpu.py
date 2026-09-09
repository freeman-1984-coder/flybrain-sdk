import math

import numpy as np
import pytest

from flybrain import Connectome, LIFConfig, Synapse
from flybrain.backends import CPUBackend


def circuit(edges=(), n=1):
    return Connectome("test", tuple(str(i) for i in range(n)), tuple(edges), {}, {}, {})


def test_exact_subthreshold_solution():
    backend = CPUBackend(circuit(), LIFConfig(threshold=10, tau_ms=10))
    for _ in range(17):
        backend.step(np.array([2.0]))
    assert backend.observe().voltage[0] == pytest.approx(2 * (1 - math.exp(-1.7)))


def test_reset_and_refractory_hold_exact_tick_count():
    backend = CPUBackend(circuit(), LIFConfig(refractory_ms=2))
    ticks = []
    for _ in range(8):
        backend.step(np.array([20.0]))
        state = backend.observe()
        if state.spikes[0]:
            ticks.append(state.tick)
        assert state.voltage[0] == 0
    assert ticks == [1, 4, 7]


def test_previous_tick_spikes_and_parallel_edge_accumulation():
    backend = CPUBackend(circuit((Synapse(0, 1, 8), Synapse(0, 1, 8)), n=2), LIFConfig())
    backend.step(np.array([20.0, 0.0]))
    assert backend.observe().spikes == (True, False)
    backend.step(np.zeros(2))
    assert backend.observe().spikes == (False, True)


def test_inhibitory_synapse_reduces_voltage():
    backend = CPUBackend(circuit((Synapse(0, 1, -16),), n=2), LIFConfig())
    backend.step(np.array([20.0, 0.0]))
    backend.step(np.zeros(2))
    assert backend.observe().voltage[1] < 0
    assert not backend.observe().spikes[1]


def test_rate_filter_is_in_hz():
    backend = CPUBackend(circuit(), LIFConfig())
    backend.step(np.array([20.0]))
    expected = 1000 * (1 - math.exp(-1 / 50))
    assert backend.observe().rates_hz[0] == pytest.approx(expected)
    backend.step(np.zeros(1))
    assert backend.observe().rates_hz[0] == pytest.approx(expected * math.exp(-1 / 50))


@pytest.mark.parametrize("current", [[1, 2], [float("nan")], [float("inf")]])
def test_bad_current_does_not_advance_backend(current):
    backend = CPUBackend(circuit(), LIFConfig())
    before = backend.observe()
    with pytest.raises(ValueError):
        backend.step(np.array(current))
    assert backend.observe() == before
