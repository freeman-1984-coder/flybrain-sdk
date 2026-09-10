"""Standalone playback for recorded demo sessions; never masquerades as live simulation."""

import json
from importlib.resources import files
from pathlib import Path


def write_replay_html(recording, path):
    """Create a network-free HTML viewer without embedding the full connectome."""
    initial = recording["initial"]
    kind = initial["environment"]["type"]
    if kind not in ("dodge-arena-v1", "pulse-score-v1") or not recording["frames"]:
        raise ValueError("viewer requires a nonempty dodge or tones recording")
    packet = {
        "kind": kind,
        "model": initial["brain"]["model"]["name"],
        "fingerprint": initial["brain"]["model_sha256"],
        "period_ms": initial["period_ms"],
        "initial_silenced": sum(initial["brain"]["state"].get("silenced", [])),
        "initial": initial["environment"],
        "frames": [
            {
                k: f[k]
                for k in (
                    "index",
                    "elapsed_ms",
                    "brain_tick",
                    "observation",
                    "requested",
                    "applied",
                    "environment",
                )
            }
            for f in recording["frames"]
        ],
    }
    payload = json.dumps(packet, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = files("flybrain.data").joinpath("replay.html").read_text(encoding="utf-8")
    html = html.replace("__DATA__", payload)
    Path(path).write_text(html, encoding="utf-8")
    return Path(path)
