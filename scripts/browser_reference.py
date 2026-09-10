"""Generate deterministic Python traces for the actual JS CPU runtime tests."""

import json
from pathlib import Path

from flybrain import FlyBrain, LIFConfig

root = Path(__file__).resolve().parents[1]
folder = root / "packages/js/tests/fixtures"
folder.mkdir(parents=True, exist_ok=True)
(root / "site/toy-bundle.json").write_text(
    json.dumps(
        {
            "format": "flybrain-model-bundle",
            "schema_version": 1,
            "model": FlyBrain.load().model.to_dict(),
            "config": LIFConfig().to_dict(),
            "model_sha256": FlyBrain.load().model.fingerprint,
        },
        indent=2,
    )
    + "\n"
)
trials = []
for model, config in [
    ("toy", LIFConfig(dt_ms=0.5)),
    (root / "models/male-cns-escape-v1/model.json", None),
]:
    brain = FlyBrain.load(model, config=config)
    channel = "food" if model == "toy" else "looming_left"
    target = ["relay.food"] if model == "toy" else ["10001", "10010"]
    commands = [
        {"tick": 0, "op": "stimulate", "channel": channel, "strength": 0.8, "duration_ms": 113.2},
        {"tick": 21, "op": "silence", "ids": target, "enabled": True},
        {"tick": 52, "op": "silence", "ids": target, "enabled": False},
        {"tick": 110, "op": "inject", "ids": target, "amplitude": -2, "duration_ms": 20},
    ]
    trace = []
    for tick in range(250):
        for cmd in commands:
            if cmd["tick"] != tick:
                continue
            if cmd["op"] == "stimulate":
                brain.stimulate(
                    cmd["channel"], strength=cmd["strength"], duration_ms=cmd["duration_ms"]
                )
            elif cmd["op"] == "silence":
                brain.intervene.silence(cmd["ids"], enabled=cmd["enabled"])
            else:
                brain.drive.current(
                    cmd["ids"], amplitude=cmd["amplitude"], duration_ms=cmd["duration_ms"]
                )
        state = brain.step()
        trace.append(
            {
                "tick": state.tick,
                "voltage": state.voltage,
                "spikes": state.spikes,
                "rates_hz": state.rates_hz,
                "action": brain.action().to_dict(),
            }
        )
    trials.append(
        {
            "model": "toy" if model == "toy" else "real",
            "config": brain.config.to_dict(),
            "commands": commands,
            "trace": trace,
        }
    )
(folder / "python-reference.json").write_text(json.dumps(trials, separators=(",", ":")) + "\n")
print("Generated two Python reference trials (500 ticks total).")
