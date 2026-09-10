"""Replay circuit-lab commands against a verified model, never executable code."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .brain import FlyBrain
from .config import LIFConfig, finite_number, positive_int
from .registry import model_info


def replay_experiment(
    recording: dict,
    *,
    download: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    max_steps: int = 1_000_000,
) -> FlyBrain:
    """Recreate a browser experiment and verify its final observation.

    Downloads use only the installed catalog, never a URL from the recording.
    Return the live brain on success; raise ValueError on invalid data or a
    numerical mismatch. Final floats use absolute tolerance 1e-9, spikes exact.
    This verifies the recorded endpoint, not every intermediate browser tick.
    """
    if not isinstance(recording, dict):
        raise ValueError("recording must be an object")
    try:
        return _replay(recording, download, cache_dir, max_steps)
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid experiment structure: {error}") from error


def _replay(data, download, cache_dir, max_steps):
    if (
        data["format"] != "flybrain-experiment"
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["dynamics_revision"] != "lif-exact-v1"
    ):
        raise ValueError("unsupported experiment format or dynamics")
    limit = positive_int(max_steps, "max_steps")
    duration = positive_int(data["duration_ticks"], "duration_ticks", allow_zero=True)
    if duration > limit:
        raise ValueError("experiment exceeds max_steps")
    entry = model_info(data["model_id"])
    if entry["status"] not in ("ready", "builtin"):
        raise ValueError("experiment needs a runnable catalog model")
    expected_asset = (
        entry["assets"]["model"]["checksum"].removeprefix("sha256:")
        if entry["status"] == "ready"
        else None
    )
    if data["model_asset_sha256"] != expected_asset:
        raise ValueError("model asset checksum differs from installed catalog")
    config = LIFConfig(**data["config"])
    commands = data["commands"]
    if not isinstance(commands, list) or len(commands) > 10_000:
        raise ValueError("commands must be a list of at most 10000 entries")
    previous = 0
    for command in commands:
        tick = positive_int(command["tick"], "command tick", allow_zero=True)
        if not previous <= tick <= duration:
            raise ValueError("command ticks must be ordered within experiment duration")
        if command["op"] not in ("stimulate", "silence", "inject"):
            raise ValueError("unknown experiment command")
        previous = tick
    brain = FlyBrain.load(data["model_id"], config=config, download=download, cache_dir=cache_dir)
    if data["model_sha256"] != brain.model.fingerprint:
        raise ValueError("model fingerprint mismatch")
    # Match JS's explicit model ports rather than Python's legacy four channels.
    brain.bind_readout(brain.model.to_dict()["motor"])
    tick = 0
    for command in commands:
        delta = command["tick"] - tick
        if delta:
            brain.step(delta)
        tick = command["tick"]
        if command["op"] == "stimulate":
            brain.stimulate(
                command["channel"],
                strength=command["strength"],
                duration_ms=command["duration_ms"],
            )
        elif command["op"] == "silence":
            if type(command["enabled"]) is not bool:
                raise ValueError("silence enabled must be boolean")
            brain.intervene.silence(command["ids"], enabled=command["enabled"])
        else:
            brain.drive.current(
                command["ids"],
                amplitude=command["amplitude"],
                duration_ms=command["duration_ms"],
            )
    if duration > tick:
        brain.step(duration - tick)
    actual = brain.state
    expected = data["expected"]["state"]
    if (
        positive_int(expected["tick"], "expected tick", allow_zero=True) != actual.tick
        or finite_number(expected["time_ms"], "time_ms") != actual.time_ms
    ):
        raise ValueError("final clock mismatch")
    for field in ("voltage", "rates_hz", "spikes"):
        values = expected[field]
        if not isinstance(values, list) or len(values) != len(brain.model.neuron_ids):
            raise ValueError(f"invalid expected {field} shape")
        if field == "spikes":
            if any(type(x) is not bool for x in values) or tuple(values) != actual.spikes:
                raise ValueError("final spikes mismatch")
        else:
            values = [finite_number(x, field) for x in values]
            if not np.allclose(values, getattr(actual, field), rtol=0, atol=1e-9):
                raise ValueError(f"final {field} mismatch")
    action = data["expected"]["action"]
    actual_action = brain.action().to_dict()
    if not isinstance(action, dict) or set(action) != set(actual_action):
        raise ValueError("final action channels mismatch")
    for name, value in actual_action.items():
        if abs(finite_number(action[name], name) - value) > 1e-9:
            raise ValueError(f"final action {name} mismatch")
    return brain
