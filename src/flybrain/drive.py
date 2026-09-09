"""Direct current pulses with explicit units and tick-aligned durations."""

from math import isclose
from typing import Sequence

import numpy as np

from .config import LIFConfig, finite_number, positive_int
from .neurons import NeuronIndex


def duration_ticks(duration_ms: float, dt_ms: float) -> int:
    duration = finite_number(duration_ms, "duration_ms")
    if duration <= 0:
        raise ValueError("duration_ms must be positive")
    ratio = duration / dt_ms
    count = round(ratio)
    if count < 1 or not isclose(ratio, count, rel_tol=0, abs_tol=1e-9):
        raise ValueError("duration_ms must be an integer multiple of dt_ms")
    return count


class CurrentDrive:
    def __init__(self, neurons: NeuronIndex, config: LIFConfig):
        self.neurons = neurons
        self.config = config
        self._pending = []

    def current(
        self, ids: Sequence[str], *, amplitude: float, duration_ms: float, units: str = "normalized"
    ) -> None:
        """Add a signed current at the next tick; no sensory input_gain multiplier."""
        if units != "normalized":
            raise ValueError("lif-exact-v1 accepts normalized current units only")
        indices = self.neurons.resolve(ids)
        amplitude = finite_number(amplitude, "amplitude")
        steps = duration_ticks(duration_ms, self.config.dt_ms)
        self._pending.append({"indices": indices, "amplitude": amplitude, "remaining_steps": steps})

    def peek(self) -> np.ndarray:
        values = np.zeros(len(self.neurons.model.neuron_ids), dtype=np.float64)
        for pulse in self._pending:
            values[list(pulse["indices"])] += pulse["amplitude"]
        return values

    def consume(self) -> None:
        for pulse in self._pending:
            pulse["remaining_steps"] -= 1
        self._pending = [p for p in self._pending if p["remaining_steps"] > 0]

    def clear(self) -> None:
        self._pending.clear()

    def snapshot(self) -> list:
        ids = self.neurons.model.neuron_ids
        return [
            {
                "neuron_ids": [ids[i] for i in p["indices"]],
                "amplitude": p["amplitude"],
                "remaining_steps": p["remaining_steps"],
            }
            for p in self._pending
        ]

    def restore(self, data: list) -> None:
        if not isinstance(data, list):
            raise ValueError("pending currents must be a list")
        checked = []
        for p in data:
            checked.append(
                {
                    "indices": self.neurons.resolve(p["neuron_ids"]),
                    "amplitude": finite_number(p["amplitude"], "amplitude"),
                    "remaining_steps": positive_int(p["remaining_steps"], "remaining_steps"),
                }
            )
        self._pending = checked


class Interventions:
    def __init__(self, neurons, backend):
        self.neurons, self.backend = neurons, backend

    def silence(self, ids: Sequence[str], *, enabled: bool = True) -> None:
        """Suppress new spikes from next tick; previous spikes still propagate.

        Silenced cells are held at reset. Historical firing rates decay normally.
        Pass enabled=False to release the intervention.
        """
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        self.backend.set_silenced(self.neurons.resolve(ids), enabled)
