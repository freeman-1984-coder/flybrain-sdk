"""Small deterministic game and sonification environments using the same Session.

All projections and controls are engineered application choices. Neither demo
claims biological behavior, musical understanding, training or task learning.
"""

import math
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

from .brain import FlyBrain
from .config import finite_number, positive_int
from .session import LinearEncoder, RateReadout, Session, detached


def bounded(value, name, lower, upper):
    value = finite_number(value, name)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return value


class DodgeArena:
    """A normalized 1D player avoids descending obstacles. No scripted rescue."""

    def __init__(self, seed=7):
        self.seed = positive_int(seed, "seed", allow_zero=True)
        if self.seed >= 2**32:
            raise ValueError("seed must fit uint32")
        self.rng = self.seed
        self.frame = self.collisions = self.passed = 0
        self.x = 0.5
        self.blocks = []
        self._spawn()

    def _spawn(self):
        self.rng = (1664525 * self.rng + 1013904223) % 2**32
        self.blocks.append({"x": 0.15 + 0.7 * self.rng / 2**32, "y": 0.0})

    def observe(self):
        left = right = 0.0
        for block in self.blocks:
            dx = block["x"] - self.x
            intensity = max(0, min(1, (block["y"] - 0.1) / 0.65)) * max(0, 1 - abs(dx) / 0.6)
            if dx <= 0:
                left = max(left, intensity)
            if dx >= 0:
                right = max(right, intensity)
        return {
            "danger_left": left,
            "danger_right": right,
            "player_x": self.x,
            "collisions": self.collisions,
            "passed": self.passed,
        }

    def apply(self, action, duration_ms):
        dt = finite_number(duration_ms, "duration_ms") / 1000
        if dt <= 0:
            raise ValueError("duration_ms must be positive")
        steer = max(-1, min(1, finite_number(action["steer"], "steer")))
        previous_x = self.x
        self.x = max(0, min(1, self.x + steer * dt))
        remaining = []
        for block in self.blocks:
            y = block["y"] + 0.8 * dt
            if y >= 1:
                if abs(block["x"] - self.x) < 0.13:
                    self.collisions += 1
                else:
                    self.passed += 1
            else:
                remaining.append({"x": block["x"], "y": y})
        self.blocks = remaining
        self.frame += 1
        if self.frame % 45 == 0:
            self._spawn()
        return {"steer": (self.x - previous_x) / dt}

    def snapshot(self):
        return detached(
            {
                "type": "dodge-arena-v1",
                "seed": self.seed,
                "rng": self.rng,
                "frame": self.frame,
                "x": self.x,
                "blocks": self.blocks,
                "collisions": self.collisions,
                "passed": self.passed,
            }
        )

    @classmethod
    def from_snapshot(cls, data):
        if data["type"] != "dodge-arena-v1":
            raise ValueError("unsupported environment")
        result = cls(data["seed"])
        result.rng = positive_int(data["rng"], "rng", allow_zero=True)
        if result.rng >= 2**32:
            raise ValueError("rng must fit uint32")
        result.frame = positive_int(data["frame"], "frame", allow_zero=True)
        result.x = bounded(data["x"], "x", 0, 1)
        result.collisions = positive_int(data["collisions"], "collisions", allow_zero=True)
        result.passed = positive_int(data["passed"], "passed", allow_zero=True)
        if not isinstance(data["blocks"], list) or len(data["blocks"]) > 10000:
            raise ValueError("invalid obstacle list")
        result.blocks = [
            {"x": bounded(b["x"], "x", 0, 1), "y": bounded(b["y"], "y", 0, 1)}
            for b in data["blocks"]
        ]
        return result


class PulseScore:
    """Alternating synthetic pulses; neural rates control two oscillator gains."""

    def __init__(self):
        self.elapsed = Fraction(0)
        self.last_applied = {"tone_a": 0.0, "tone_b": 0.0}

    def observe(self):
        phase = self.elapsed % 1000
        return {
            "pulse_a": float(0 <= phase < 100 or 500 <= phase < 600),
            "pulse_b": float(250 <= phase < 350 or 750 <= phase < 850),
        }

    def apply(self, action, duration_ms):
        duration_ms = finite_number(duration_ms, "duration_ms")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        values = {k: max(0, min(1, finite_number(action[k], k))) for k in self.last_applied}
        self.elapsed += Fraction(str(duration_ms))
        self.last_applied = values
        return dict(values)

    def snapshot(self):
        return {
            "type": "pulse-score-v1",
            "elapsed": [self.elapsed.numerator, self.elapsed.denominator],
            "last_applied": dict(self.last_applied),
        }

    @classmethod
    def from_snapshot(cls, data):
        if data["type"] != "pulse-score-v1":
            raise ValueError("unsupported environment")
        result = cls()
        numerator, denominator = data["elapsed"]
        result.elapsed = Fraction(
            positive_int(numerator, "elapsed", allow_zero=True),
            positive_int(denominator, "denominator"),
        )
        result.last_applied = {
            k: bounded(data["last_applied"][k], k, 0, 1) for k in result.last_applied
        }
        return result


def make_demo(
    name, *, model="toy", download=False, cache_dir=None, period_ms=20.0, seed=7, backend="cpu"
):
    """Compose a known demo; source models still require explicit download opt-in."""
    if name not in ("dodge", "tones"):
        raise ValueError("choose dodge or tones")
    brain = FlyBrain.load(model, backend=backend, download=download, cache_dir=cache_dir)
    ports = brain.model.to_dict()["sensory"]
    if brain.model.name == "toy-v1":
        inputs = (ports["looming_left"], ports["looming_right"])
        outputs = (["motor.turn_right"], ["motor.turn_left"])
    elif (
        brain.model.fingerprint
        == "47a0c91b3eba9c606c29faef58fe15c8846fc049c4314a5e3b6912ab22c5e68a"
    ):
        inputs = (ports["looming_left"], ports["looming_right"])
        outputs = (["10010"], ["10001"])
    else:
        raise ValueError(
            "demo preset supports toy-v1 or the pinned MaleCNS escape model; "
            "compose Session explicitly for another model"
        )
    features = ("danger_left", "danger_right") if name == "dodge" else ("pulse_a", "pulse_b")
    encoder = LinearEncoder(
        {feature: dict.fromkeys(ids, 2.0) for feature, ids in zip(features, inputs)}
    )
    a, b = outputs
    if name == "dodge":
        channels = {
            "steer": {
                "weights": {**dict.fromkeys(a, 1 / len(a)), **dict.fromkeys(b, -1 / len(b))},
                "min": -1.0,
                "max": 1.0,
            }
        }
        environment = DodgeArena(seed)
    else:
        channels = {
            key: {"weights": dict.fromkeys(ids, 1 / len(ids)), "min": 0.0, "max": 1.0}
            for key, ids in zip(("tone_a", "tone_b"), outputs)
        }
        environment = PulseScore()
    return Session(brain, encoder, RateReadout(channels), environment, period_ms=period_ms)


def render_wav(recording, path, *, sample_rate=16000, fade_out=True):
    """Render applied tone gains as original audio; support resumed recordings.

    Phase uses absolute session time; initial gain comes from the environment
    checkpoint. Set fade_out=False when joining clips without an end fade.
    This is offline sonification, not musical understanding or a trained policy.
    """
    sample_rate = positive_int(sample_rate, "sample_rate")
    if not 8000 <= sample_rate <= 192000 or type(fade_out) is not bool:
        raise ValueError("invalid sample_rate or fade_out")
    initial = recording["initial"]
    if initial["environment"]["type"] != "pulse-score-v1":
        raise ValueError("tone rendering requires a pulse-score recording")
    frames = recording["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("no audio frames")
    start_index = positive_int(initial["frame_index"], "frame_index", allow_zero=True)
    period = finite_number(initial["period_ms"], "period_ms")
    if period <= 0:
        raise ValueError("period_ms must be positive")
    start_sample = round(start_index * period * sample_rate / 1000)
    end_sample = round((start_index + len(frames)) * period * sample_rate / 1000)
    if not 0 < end_sample - start_sample <= 10_000_000:
        raise ValueError("audio export must contain 1..10000000 samples")
    previous_gain = np.array(
        [bounded(initial["environment"]["last_applied"][k], k, 0, 1) for k in ("tone_a", "tone_b")]
    )
    checked = []
    previous_sample = start_sample
    for offset, frame in enumerate(frames, 1):
        index = start_index + offset
        if type(frame["index"]) is not int or frame["index"] != index:
            raise ValueError("audio frame indices must be consecutive")
        elapsed = finite_number(frame["elapsed_ms"], "elapsed_ms")
        if not math.isclose(elapsed, index * period, rel_tol=0, abs_tol=1e-7):
            raise ValueError("audio frame clock mismatch")
        end = round(elapsed * sample_rate / 1000)
        if end <= previous_sample:
            raise ValueError("audio frames must advance by at least one sample")
        gain = np.array([bounded(frame["applied"][k], k, 0, 1) for k in ("tone_a", "tone_b")])
        checked.append((previous_sample, end, gain))
        previous_sample = end
    with wave.open(str(Path(path)), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        for start, end, gain in checked:
            ramp = np.linspace(previous_gain, gain, end - start, endpoint=False)
            indices = np.arange(start, end, dtype=np.float64)
            phase = 2 * math.pi * indices[:, None] / sample_rate * np.array([440.0, 660.0])
            samples = 0.18 * np.sum(ramp * np.sin(phase), axis=1)
            if fade_out:
                fade = min(end_sample - start_sample, sample_rate // 100)
                samples *= np.clip((end_sample - 1 - indices) / max(1, fade - 1), 0, 1)
            output.writeframesraw(np.rint(samples * 32767).astype("<i2").tobytes())
            previous_gain = gain
    return Path(path)
