"""Python feedback reference for the live JavaScript environment adapter."""

import json
from pathlib import Path

from flybrain.demos import make_demo
from flybrain.session import LinearEncoder, RateReadout

ROOT = Path(__file__).resolve().parents[1]
events = [
    {"frame": 0, "command": {"op": "add", "x": 0.3, "y": 0.7}},
    {"frame": 20, "command": {"op": "silence", "enabled": True}},
    {"frame": 45, "command": {"op": "silence", "enabled": False}},
    {"frame": 55, "command": {"op": "clear"}},
    {"frame": 55, "command": {"op": "add", "x": 0.7, "y": 0.7}},
    {"frame": 65, "command": {"op": "configure", "input_gain": 3, "output_gain": 1.5}},
    {"frame": 100, "command": {"op": "configure", "input_gain": 0, "output_gain": 0}},
    {"frame": 110, "command": {"op": "configure", "input_gain": 2, "output_gain": 1}},
]
trials = []
for model in ["toy", ROOT / "models/male-cns-escape-v1/model.json"]:
    session = make_demo("dodge", model=model)
    encoder = session.encoder.snapshot()
    channels = session.readout.snapshot()["channels"]
    output_ids = list(channels["steer"]["weights"])
    trace = []
    for frame in range(150):
        for event in events:
            if event["frame"] != frame:
                continue
            command = event["command"]
            if command["op"] == "add":
                session.environment.blocks.append({k: command[k] for k in ("x", "y")})
            elif command["op"] == "clear":
                session.environment.blocks.clear()
            elif command["op"] == "silence":
                session.brain.intervene.silence(output_ids, enabled=command["enabled"])
            else:
                session.encoder = LinearEncoder(
                    {
                        k: dict.fromkeys(v, command["input_gain"])
                        for k, v in encoder["weights"].items()
                    }
                )
                session.readout = RateReadout(
                    {
                        "steer": {
                            "weights": {
                                k: v * command["output_gain"]
                                for k, v in channels["steer"]["weights"].items()
                            },
                            "min": -1,
                            "max": 1,
                        }
                    }
                )
        result = session.step().to_dict()
        del result["currents"]
        result["rates_hz"] = session.brain.observe(output_ids, fields=("rates_hz",)).values[
            "rates_hz"
        ]
        trace.append(result)
    state = session.brain.snapshot()["state"]
    trials.append(
        {
            "model": "toy" if model == "toy" else "real",
            "events": events,
            "trace": trace,
            "final_state": state,
        }
    )
path = ROOT / "packages/js/tests/fixtures/dodge-reference.json"
path.write_text(json.dumps(trials, separators=(",", ":")) + "\n")
print("Generated 300 Python feedback frames, including scene edits, gains and silencing.")
