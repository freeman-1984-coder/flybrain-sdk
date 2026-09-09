"""Measure actual end-to-end CPU stepping, including observations and readout."""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from flybrain import FlyBrain

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="toy")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
if args.steps <= 0 or args.repeats <= 0:
    parser.error("steps and repeats must be positive")
timings = []
for _ in range(args.repeats):
    brain = FlyBrain.load(args.model)
    channel = "looming_left" if "looming_left" in brain.sensory_channels else "food"
    brain.stimulate(channel, duration_ms=args.steps * brain.config.dt_ms)
    start = perf_counter()
    for _ in range(args.steps):
        brain.step()
        brain.action()
    timings.append(perf_counter() - start)
report = {
    "model": brain.model.name,
    "model_sha256": brain.model.fingerprint,
    "neurons": len(brain.model.neuron_ids),
    "edges": len(brain.model.synapses),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "os": platform.system(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "dt_ms": brain.config.dt_ms,
    "simulated_ms": args.steps * brain.config.dt_ms,
    "wall_seconds": timings,
    "median_wall_seconds": median(timings),
    "real_time_factor": args.steps * brain.config.dt_ms / (1000 * median(timings)),
    "scope": "End-to-end step() and action(), loading excluded. Applies only to this subgraph.",
}
text = json.dumps(report, indent=2)
print(text)
if args.output:
    args.output.write_text(text + "\n", encoding="utf-8")
