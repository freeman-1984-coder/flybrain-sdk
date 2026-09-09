"""Explicit heuristic motor readout, separate from neural dynamics."""

from dataclasses import asdict, dataclass
from typing import Sequence

from .config import LIFConfig
from .model import Connectome


@dataclass(frozen=True)
class MotorAction:
    """Independent [0, 1] control intensities, not probabilities or physical units."""

    walk: float = 0.0
    turn_left: float = 0.0
    turn_right: float = 0.0
    jump: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class MotorDecoder:
    def __init__(self, model: Connectome, config: LIFConfig) -> None:
        unknown = set(model.motor) - set(MotorAction.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported motor channels: {sorted(unknown)}")
        self.model = model
        self.config = config

    def decode(self, rates_hz: Sequence[float]) -> MotorAction:
        values = {}
        for name, indices in self.model.motor.items():
            mean_rate = sum(rates_hz[i] for i in indices) / len(indices)
            values[name] = max(0.0, min(1.0, mean_rate / self.config.action_rate_hz))
        return MotorAction(**values)
