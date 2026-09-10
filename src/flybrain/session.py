"""Replaceable feature/current, rate/action and environment adapters with one clock."""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Protocol

from .brain import FlyBrain
from .config import finite_number, positive_int
from .errors import CheckpointError


def detached(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write_json(path, data):
    path = Path(path)
    raw = json.dumps(data, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


class Encoder(Protocol):
    def encode(self, observation: Mapping[str, float]) -> Mapping[str, float]:
        """Return normalized current by source neuron ID; no brain mutation."""

    def snapshot(self) -> dict: ...


class Readout(Protocol):
    def decode(self, brain: FlyBrain) -> Mapping[str, float]: ...
    def snapshot(self) -> dict: ...


class Environment(Protocol):
    def observe(self) -> Mapping[str, float]:
        """Expose current observations only."""

    def apply(self, action: Mapping[str, float], duration_ms: float) -> Mapping[str, float]:
        """Advance the environment and report controls actually applied."""

    def snapshot(self) -> dict: ...


class LinearEncoder:
    """Sparse feature -> source-ID current weights; signed normalized units."""

    def __init__(self, weights):
        if not isinstance(weights, dict):
            raise ValueError("encoder weights must be a feature mapping")
        self._weights = {}
        for feature, targets in weights.items():
            if not isinstance(feature, str) or not feature or not isinstance(targets, dict):
                raise ValueError("invalid feature mapping")
            self._weights[feature] = {}
            for neuron, gain in targets.items():
                if not isinstance(neuron, str) or not neuron:
                    raise ValueError("target IDs must be nonempty strings")
                self._weights[feature][neuron] = finite_number(gain, "gain")

    def encode(self, observation):
        result = {}
        for feature, targets in self._weights.items():
            value = finite_number(observation[feature], feature)
            for neuron, gain in targets.items():
                result[neuron] = result.get(neuron, 0) + value * gain
        return {k: finite_number(v, "projected current") for k, v in result.items()}

    def snapshot(self):
        return {"type": "linear-current-v1", "weights": detached(self._weights)}

    @classmethod
    def from_snapshot(cls, data):
        if data["type"] != "linear-current-v1":
            raise ValueError("unsupported encoder")
        return cls(data["weights"])


class RateReadout:
    """Weighted filtered-Hz readouts, with declared scale and output bounds."""

    def __init__(self, channels, *, scale_hz=100.0):
        self.scale_hz = finite_number(scale_hz, "scale_hz")
        if self.scale_hz <= 0 or not isinstance(channels, dict):
            raise ValueError("invalid readout scale or channels")
        self._channels = {}
        for name, channel in channels.items():
            if not isinstance(name, str) or not name:
                raise ValueError("output channel names must be nonempty strings")
            weights = LinearEncoder({name: channel["weights"]}).snapshot()["weights"][name]
            lower, upper = (finite_number(channel[k], k) for k in ("min", "max"))
            if lower > upper:
                raise ValueError("readout min exceeds max")
            self._channels[name] = {"weights": weights, "min": lower, "max": upper}
        self._ids = tuple(sorted({n for c in self._channels.values() for n in c["weights"]}))

    def decode(self, brain):
        rates = (
            dict(zip(self._ids, brain.observe(self._ids, fields=("rates_hz",)).values["rates_hz"]))
            if self._ids
            else {}
        )
        result = {}
        for name, channel in self._channels.items():
            value = sum(rates[n] * w for n, w in channel["weights"].items()) / self.scale_hz
            result[name] = max(channel["min"], min(channel["max"], finite_number(value, name)))
        return result

    def snapshot(self):
        return {
            "type": "weighted-rates-v1",
            "scale_hz": self.scale_hz,
            "channels": detached(self._channels),
        }

    @classmethod
    def from_snapshot(cls, data):
        if data["type"] != "weighted-rates-v1":
            raise ValueError("unsupported readout")
        return cls(data["channels"], scale_hz=data["scale_hz"])


@dataclass(frozen=True)
class Frame:
    index: int
    elapsed_ms: float
    brain_tick: int
    observation: dict
    currents: dict
    requested: dict
    applied: dict
    environment: dict

    def to_dict(self):
        return detached(asdict(self))


class Session:
    """Sample, project, integrate, decode, apply. One owner of the brain clock.

    Environment period must be >= one neural tick. Fractional ratios alternate
    tick counts using an exact decimal accumulator. No neural ticks are dropped.
    Application exceptions after integration leave a partially advanced session;
    recover from a prior checkpoint, rather than retrying the same frame.
    """

    def __init__(
        self, brain, encoder: Encoder, readout: Readout, environment: Environment, *, period_ms=20.0
    ):
        self.brain, self.encoder, self.readout, self.environment = (
            brain,
            encoder,
            readout,
            environment,
        )
        self.period_ms = finite_number(period_ms, "period_ms")
        if self.period_ms < brain.config.dt_ms:
            raise ValueError("environment period must be at least one neural tick")
        self._period = Fraction(str(self.period_ms))
        self._ratio = self._period / Fraction(str(brain.config.dt_ms))
        self.frame_index = 0
        self._base_tick = brain.progress.tick
        if isinstance(encoder, LinearEncoder):
            ids = tuple({n for targets in encoder._weights.values() for n in targets})
            if ids:
                brain.neurons.resolve(ids)
        if isinstance(readout, RateReadout) and readout._ids:
            brain.neurons.resolve(readout._ids)

    def _check_clock(self):
        expected = self._base_tick + math.floor(self.frame_index * self._ratio)
        if self.brain.progress.tick != expected:
            raise ValueError("brain clock changed outside the session; restore a checkpoint")

    def step(self):
        self._check_clock()
        observation = detached(dict(self.environment.observe()))
        currents = dict(self.encoder.encode(observation))
        if currents:
            self.brain.neurons.resolve(tuple(currents))
        groups = {}
        for neuron, value in currents.items():
            value = finite_number(value, "current")
            if value:
                groups.setdefault(value, []).append(neuron)
        ticks = math.floor((self.frame_index + 1) * self._ratio) - math.floor(
            self.frame_index * self._ratio
        )
        for amplitude, ids in groups.items():
            self.brain.drive.current(
                ids, amplitude=amplitude, duration_ms=ticks * self.brain.config.dt_ms
            )
        self.brain.advance(duration_ms=ticks * self.brain.config.dt_ms)
        requested = {k: finite_number(v, k) for k, v in self.readout.decode(self.brain).items()}
        applied = {
            k: finite_number(v, k)
            for k, v in self.environment.apply(requested, self.period_ms).items()
        }
        self.frame_index += 1
        return Frame(
            self.frame_index,
            float(self.frame_index * self._period),
            self.brain.progress.tick,
            observation,
            detached(currents),
            detached(requested),
            detached(applied),
            detached(self.environment.snapshot()),
        )

    def snapshot(self):
        self._check_clock()
        return detached(
            {
                "format": "flybrain-session",
                "schema_version": 1,
                "period_ms": self.period_ms,
                "frame_index": self.frame_index,
                "base_tick": self._base_tick,
                "brain": self.brain.snapshot(),
                "encoder": self.encoder.snapshot(),
                "readout": self.readout.snapshot(),
                "environment": self.environment.snapshot(),
            }
        )

    def save(self, path):
        return write_json(path, self.snapshot())

    @classmethod
    def from_snapshot(
        cls,
        data,
        *,
        environment_factory,
        encoder_factory=LinearEncoder.from_snapshot,
        readout_factory=RateReadout.from_snapshot,
        backend=None,
    ):
        """Factories are supplied by caller code, never imported from recorded data."""
        try:
            if (
                data["format"] != "flybrain-session"
                or type(data["schema_version"]) is not int
                or data["schema_version"] != 1
            ):
                raise ValueError("unsupported session schema")
            session = cls(
                FlyBrain.from_snapshot(data["brain"], backend=backend),
                encoder_factory(detached(data["encoder"])),
                readout_factory(detached(data["readout"])),
                environment_factory(detached(data["environment"])),
                period_ms=data["period_ms"],
            )
            session.frame_index = positive_int(data["frame_index"], "frame_index", allow_zero=True)
            session._base_tick = positive_int(data["base_tick"], "base_tick", allow_zero=True)
            session._check_clock()
            return session
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise CheckpointError(f"Invalid session: {error}") from error

    @classmethod
    def restore(cls, path, **factories):
        return cls.from_snapshot(json.loads(Path(path).read_text(encoding="utf-8")), **factories)

    def summary(self):
        result = self.snapshot()
        del result["brain"]["model"]  # Full graph already belongs to initial checkpoint.
        return result

    def record(self, frames):
        frames = positive_int(frames, "frames")
        initial = self.snapshot()
        records = [self.step().to_dict() for _ in range(frames)]
        return {
            "format": "flybrain-session-recording",
            "schema_version": 1,
            "initial": initial,
            "frames": records,
            "final": self.summary(),
        }


def compare(actual, expected, path="recording"):
    if (
        isinstance(actual, float)
        and isinstance(expected, (float, int))
        and not isinstance(expected, bool)
    ):
        if not math.isclose(actual, finite_number(expected, path), rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"replay mismatch at {path}")
    elif type(actual) is dict and type(expected) is dict and actual.keys() == expected.keys():
        for k in actual:
            compare(actual[k], expected[k], f"{path}.{k}")
    elif type(actual) is list and type(expected) is list and len(actual) == len(expected):
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}[{i}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"replay mismatch at {path}")


def replay(recording, *, max_frames=10000, max_steps=1_000_000, **factories):
    """Re-run environment feedback and verify every frame plus complete final state."""
    if (
        recording["format"] != "flybrain-session-recording"
        or type(recording["schema_version"]) is not int
        or recording["schema_version"] != 1
    ):
        raise ValueError("unsupported recording")
    frames = recording["frames"]
    if not isinstance(frames, list) or not 0 < len(frames) <= positive_int(
        max_frames, "max_frames"
    ):
        raise ValueError("invalid recording length")
    session = Session.from_snapshot(recording["initial"], **factories)
    steps = math.floor((session.frame_index + len(frames)) * session._ratio) - math.floor(
        session.frame_index * session._ratio
    )
    if steps > positive_int(max_steps, "max_steps"):
        raise ValueError("recording exceeds max_steps")
    for expected in frames:
        compare(session.step().to_dict(), expected)
    compare(session.summary(), recording["final"])
    return session
