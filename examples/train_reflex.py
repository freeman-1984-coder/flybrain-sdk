"""Train an external action readout from a fixed real connectome's firing rates.

One-step contextual bandit: dodge opposite the approaching obstacle (+1), or
collide (0). This is engineered readout learning, not biological plasticity.
No labels, cue direction, or reward enter the readout's inference features.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from flybrain import FlyBrain

ROOT = Path(__file__).resolve().parents[1]
IDS = ("10010", "10001")


def collect(model, backend, seed, count):
    rng = np.random.default_rng(seed)
    sides = np.tile([0, 1], count // 2)
    rng.shuffle(sides)
    records = []
    for index, side in enumerate(sides):
        brain = FlyBrain.load(model, backend=backend)
        strength = float(rng.uniform(0.65, 1.0))
        distractor = float(rng.uniform(0, 0.25))
        channels = ("looming_left", "looming_right")
        brain.stimulate(channels[side], strength=strength, duration_ms=80)
        brain.stimulate(channels[1 - side], strength=distractor, duration_ms=80)
        brain.advance(duration_ms=80)
        rates = list(brain.observe(IDS, fields=("rates_hz",)).values["rates_hz"])
        records.append(
            {
                "trial": index,
                "obstacle_side": int(side),
                "strength": strength,
                "distractor": distractor,
                "rates_hz": rates,
                "target": int(1 - side),
            }
        )
    return records


def features(records):
    # Rates only; bias permits a learned action preference.
    rates = np.asarray([r["rates_hz"] for r in records], dtype=np.float64) / 100
    return np.column_stack([rates, np.ones(len(records))])


def probabilities(xp, x, weights):
    return 1 / (1 + xp.exp(-xp.clip(x @ weights, -30, 30)))


def train(x, targets, *, xp=np, epochs=24, seed=23):
    """REINFORCE with sampled actions and binary reward; fixed 0.5 baseline."""
    rng = np.random.default_rng(seed)
    x = xp.asarray(x)
    targets = xp.asarray(targets)
    weights = xp.zeros(x.shape[1], dtype=xp.float64)
    curve = []
    for epoch in range(epochs):
        p = probabilities(xp, x, weights)
        action = (xp.asarray(rng.random(len(x))) < p).astype(xp.int32)
        reward = (action == targets).astype(xp.float64)
        weights += 0.8 * x.T @ ((reward - 0.5) * (action - p)) / len(x)
        greedy = probabilities(xp, x, weights) >= 0.5
        curve.append(
            {
                "epoch": epoch + 1,
                "sampled_reward": float(reward.mean()),
                "train_accuracy": float((greedy == targets).mean()),
            }
        )
    return weights, curve


def evaluate(records, weights):
    p = probabilities(np, features(records), np.asarray(weights))
    trials = [
        {
            **r,
            "probability_right": float(prob),
            "action": int(prob >= 0.5),
            "success": int(prob >= 0.5) == r["target"],
        }
        for r, prob in zip(records, p)
    ]
    return {"accuracy": float(np.mean([t["success"] for t in trials])), "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("reflex-training.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    brain = FlyBrain.load(ROOT / "models/male-cns-escape-v1/model.json")
    model = brain.model
    xp = np
    if args.backend == "cuda":
        from flybrain.backends.cuda import _load_cupy

        xp = _load_cupy()
    training = collect(model, args.backend, 101, 64)
    # Separate seeds and amplitudes; held-out outcomes never update weights.
    heldout = collect(model, args.backend, 202, 80)
    before = evaluate(heldout, [0, 0, 0])
    weights, curve = train(features(training), [r["target"] for r in training], xp=xp)
    weights = weights.get().tolist() if args.backend == "cuda" else weights.tolist()
    after = evaluate(heldout, weights)
    checkpoint = {
        "format": "flybrain-external-reflex-readout-v1",
        "neuron_ids": list(IDS),
        "scale_hz": 100,
        "bias_feature": 1,
        "weights": weights,
        "model_fingerprint": model.fingerprint,
        "actions": ["left", "right"],
    }
    # Evaluate again after an actual JSON round trip of the portable weights.
    restored = json.loads(json.dumps(checkpoint))
    assert evaluate(heldout, restored["weights"]) == after
    report = {
        "format": "flybrain-reflex-training-v1",
        "backend": args.backend,
        "model": model.name,
        "neurons": len(model.neuron_ids),
        "model_fingerprint": model.fingerprint,
        "checkpoint": checkpoint,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_seed": 101,
        "evaluation_seed": 202,
        "optimizer_seed": 23,
        "training_trials": len(training),
        "evaluation_trials": len(heldout),
        "epochs": len(curve),
        "curve": curve,
        "before": before,
        "after": after,
        "seconds": time.perf_counter() - start,
        "limitations": [
            "Fixed real anatomical subgraph; LIF parameters and input mapping assumed.",
            "Only the external three-weight action readout is learned.",
            "Two-choice 80ms obstacle task; trials reset neural state.",
            "Held-out trials vary stimulus amplitude, not task or environment.",
            "Untrained zero weights choose right on ties; chance baseline is 50%.",
            "No claim of biological learning, whole-fly intelligence or broad generalization.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    template = Path(__file__).with_name("reflex_replay.html")
    if template.exists():
        args.output.with_suffix(".html").write_text(
            template.read_text().replace(
                "__EXPERIMENT_JSON__", json.dumps(report).replace("<", "\\u003c")
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "before": before["accuracy"],
                "after": after["accuracy"],
                "backend": args.backend,
                "seconds": report["seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
