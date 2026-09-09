"""Finite-duration, additive sensory currents."""

from dataclasses import dataclass
from math import ceil
from typing import List

import numpy as np

from .config import LIFConfig, finite_number, positive_int
from .model import Connectome


@dataclass(frozen=True)
class Stimulus:
    channel: str
    strength: float = 1.0
    duration_ms: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("channel must be a nonempty string")
        strength = finite_number(self.strength, "strength")
        duration = finite_number(self.duration_ms, "duration_ms")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if duration <= 0:
            raise ValueError("duration_ms must be > 0")
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "duration_ms", duration)


class SensoryEncoder:
    def __init__(self, model: Connectome, config: LIFConfig) -> None:
        self.model = model
        self.config = config
        self._pending: List[dict] = []

    def add(self, stimulus: Stimulus) -> None:
        if stimulus.channel not in self.model.sensory:
            raise ValueError(
                f"unknown sensory channel {stimulus.channel!r}; "
                f"available: {', '.join(self.model.sensory)}"
            )
        self._pending.append(
            {
                "channel": stimulus.channel,
                "strength": stimulus.strength,
                "remaining_steps": ceil(stimulus.duration_ms / self.config.dt_ms),
            }
        )

    def current(self) -> np.ndarray:
        current = np.zeros(len(self.model.neuron_ids), dtype=np.float64)
        for pulse in self._pending:
            indices = list(self.model.sensory[pulse["channel"]])
            current[indices] += pulse["strength"] * self.config.input_gain
            pulse["remaining_steps"] -= 1
        self._pending = [p for p in self._pending if p["remaining_steps"] > 0]
        return current

    def clear(self) -> None:
        self._pending.clear()

    def snapshot(self) -> List[dict]:
        return [dict(p) for p in self._pending]

    def restore(self, pulses: list) -> None:
        if not isinstance(pulses, list):
            raise ValueError("pending stimuli must be a list")
        checked = []
        for pulse in pulses:
            stimulus = Stimulus(pulse["channel"], pulse["strength"])
            if stimulus.channel not in self.model.sensory:
                raise ValueError("checkpoint has an unknown sensory channel")
            remaining = positive_int(pulse["remaining_steps"], "remaining_steps")
            checked.append(
                {
                    "channel": stimulus.channel,
                    "strength": stimulus.strength,
                    "remaining_steps": remaining,
                }
            )
        self._pending = checked
