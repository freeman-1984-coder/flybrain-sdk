"""Explicit heuristic motor readout, separate from neural dynamics."""

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping, Sequence, Union

from .config import LIFConfig, finite_number
from .model import Connectome
from .neurons import NeuronIndex


@dataclass(frozen=True)
class MotorAction:
    """Independent [0, 1] control intensities, not probabilities or physical units."""

    walk: float = 0.0
    turn_left: float = 0.0
    turn_right: float = 0.0
    jump: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ChannelAction(Mapping):
    """Arbitrary named [0, 1] rate intensities; not probabilities."""

    channels: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "channels", MappingProxyType(dict(self.channels)))

    def __getitem__(self, name):
        return self.channels[name]

    def __iter__(self):
        return iter(self.channels)

    def __len__(self):
        return len(self.channels)

    def to_dict(self) -> dict:
        return dict(self.channels)


class MotorDecoder:
    def __init__(self, model: Connectome, config: LIFConfig) -> None:
        self.model = model
        self.config = config
        self._channels = dict(model.motor)
        self._scale = config.action_rate_hz
        self._legacy = bool(model.motor) and set(model.motor) <= set(
            MotorAction.__dataclass_fields__
        )

    @property
    def indices(self) -> tuple:
        return tuple(sorted({i for indices in self._channels.values() for i in indices}))

    def bind(self, channels: Mapping[str, Sequence[str]], *, scale_hz: float) -> None:
        """Replace the readout without changing the connectome; validate before mutation."""
        scale = finite_number(scale_hz, "scale_hz")
        if scale <= 0:
            raise ValueError("scale_hz must be positive")
        if not isinstance(channels, Mapping):
            raise ValueError("channels must map names to neuron ID sequences")
        index = NeuronIndex(self.model)
        checked = {}
        for name, ids in channels.items():
            if not isinstance(name, str) or not name:
                raise ValueError("channel names must be nonempty strings")
            checked[name] = index.resolve(ids)
        self._channels, self._scale, self._legacy = checked, scale, False

    def decode(self, rates_hz) -> Union[MotorAction, ChannelAction]:
        values = {}
        for name, indices in self._channels.items():
            mean_rate = sum(rates_hz[i] for i in indices) / len(indices)
            values[name] = max(0.0, min(1.0, mean_rate / self._scale))
        return MotorAction(**values) if self._legacy else ChannelAction(values)

    def snapshot(self) -> dict:
        return {
            "channels": {
                k: [self.model.neuron_ids[i] for i in v] for k, v in self._channels.items()
            },
            "scale_hz": self._scale,
            "legacy": self._legacy,
        }

    def restore(self, state: dict) -> None:
        legacy = state["legacy"]
        if type(legacy) is not bool:
            raise ValueError("legacy readout flag must be boolean")
        if legacy and set(state["channels"]) - set(MotorAction.__dataclass_fields__):
            raise ValueError("legacy readout has unsupported channels")
        self.bind(state["channels"], scale_hz=state["scale_hz"])
        self._legacy = legacy
