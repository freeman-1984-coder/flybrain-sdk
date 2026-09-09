"""Backend contract: fixed-step dynamics and complete, JSON-compatible state."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..config import LIFConfig
from ..model import Connectome


@dataclass(frozen=True)
class SimulationState:
    """Read-only observation after the final tick of a step() call."""

    tick: int
    time_ms: float
    voltage: Tuple[float, ...]
    spikes: Tuple[bool, ...]
    rates_hz: Tuple[float, ...]


class Backend(ABC):
    name: str
    dynamics_revision = "lif-exact-v1"

    @abstractmethod
    def __init__(self, model: Connectome, config: LIFConfig) -> None:
        """Initialize resting voltages, zero spikes/rates, and tick zero."""

    @abstractmethod
    def step(self, current: np.ndarray) -> None:
        """Advance exactly one fixed dt; use previous-tick spikes for synapses."""

    @abstractmethod
    def observe(self) -> SimulationState:
        """Return a detached observation without advancing time."""

    @abstractmethod
    def snapshot(self) -> dict:
        """Export every value necessary for deterministic continuation."""

    @abstractmethod
    def restore(self, state: dict) -> None:
        """Validate an entire state before replacing live dynamics."""
