"""Run the real anatomical escape subgraph; no GPU or Arrow runtime needed."""

import argparse
from pathlib import Path

from flybrain import FlyBrain

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--download", action="store_true", help="fetch the reviewed catalog bundle")
args = parser.parse_args()
local = Path(__file__).resolve().parents[1] / "models" / "male-cns-escape-v1" / "model.json"
brain = (
    FlyBrain.load("male-cns-escape-v1", download=True) if args.download else FlyBrain.load(local)
)
print(
    f"{brain.model.name}: {len(brain.model.neuron_ids)} neurons, "
    f"{len(brain.model.synapses)} anatomical edges"
)
print("Real wiring + simplified assumed LIF dynamics; this is not a complete biological fly.")
brain.stimulate("looming_left", strength=1, duration_ms=100)
for _ in range(10):
    brain.step(20)
    print(f"t={brain.state.time_ms:5.0f} ms  GF jump readout={brain.action().jump:.3f}")
