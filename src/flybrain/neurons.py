"""Selections use source IDs; compact numeric indices stay inside the runtime."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from .model import Connectome


@dataclass(frozen=True)
class Observation:
    """Detached values for selected cells at one tick (spikes are this tick only)."""

    tick: int
    time_ms: float
    neuron_ids: Tuple[str, ...]
    values: Mapping[str, tuple]

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "time_ms": self.time_ms,
            "neuron_ids": list(self.neuron_ids),
            "values": {k: list(v) for k, v in self.values.items()},
        }


@dataclass(frozen=True)
class Progress:
    """Clock position without transferring a full neural state."""

    tick: int
    time_ms: float


class NeuronIndex:
    def __init__(self, model: Connectome):
        self.model = model
        self._index = {n: i for i, n in enumerate(model.neuron_ids)}
        self.attributes = tuple(sorted({k for a in model.annotations.values() for k in a}))

    def resolve(self, ids: Sequence[str]) -> Tuple[int, ...]:
        if isinstance(ids, str):
            raise ValueError("pass a sequence of neuron IDs, for example ['10001']")
        ids = tuple(ids)
        if not ids or any(not isinstance(n, str) or n not in self._index for n in ids):
            raise ValueError("selection must contain known string neuron IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("selection must not contain duplicate neuron IDs")
        return tuple(self._index[n] for n in ids)

    def select(self, ids: Optional[Sequence[str]] = None, **attributes: str) -> Tuple[str, ...]:
        """Intersect exact attribute matches; preserve model order and reject empty results.

        Attributes are model-supplied (e.g. cell_type, side), never inferred from IDs.
        Cells missing a requested attribute do not match. Unknown fields are errors.
        """
        for key, value in attributes.items():
            if key not in self.attributes:
                raise ValueError(f"model has no annotation {key!r}; available: {self.attributes}")
            if not isinstance(value, str):
                raise ValueError("annotation selectors require string values")
        allowed = set(range(len(self._index))) if ids is None else set(self.resolve(ids))
        result = tuple(
            n
            for i, n in enumerate(self.model.neuron_ids)
            if i in allowed
            and all(self.model.annotations.get(n, {}).get(k) == v for k, v in attributes.items())
        )
        if not result:
            raise ValueError("selector matches no neurons")
        return result
