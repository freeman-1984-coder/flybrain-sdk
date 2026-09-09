"""Validated parameters for the dimensionless toy LIF dynamics."""

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class LIFConfig:
    dt_ms: float = 1.0
    tau_ms: float = 10.0
    rest: float = 0.0
    reset: float = 0.0
    threshold: float = 1.0
    refractory_ms: float = 2.0
    input_gain: float = 2.0
    rate_tau_ms: float = 50.0
    action_rate_hz: float = 100.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, finite_number(value, name))
        for name in ("dt_ms", "tau_ms", "input_gain", "rate_tau_ms", "action_rate_hz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.refractory_ms < 0:
            raise ValueError("refractory_ms must be >= 0")
        if self.reset >= self.threshold or self.rest >= self.threshold:
            raise ValueError("reset and rest must be below threshold")

    def to_dict(self) -> dict:
        return asdict(self)
