"""CUDA-only full FlyWire sensory propagation experiment, not a behavior claim.

Five matched conditions start from exactly the same saved resting state. No
global background current, random input, decoder training or graph pruning.
Reports every condition, including absent downstream responses. Source data and
annotations must already exist locally; see docs/olfactory-experiment.md.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from build_olfactory_map import build
from validate_fullbrain import digest, load_fullbrain

from flybrain.backends.cuda import CUDABackend
from flybrain.config import LIFConfig
from flybrain.neurons import NeuronIndex
from flybrain.olfaction import OlfactoryDrive

CONDITIONS = (
    ("no_odor", 0.0, 0.0, False),
    ("left_odor", 1.0, 0.0, False),
    ("right_odor", 0.0, 1.0, False),
    ("bilateral_odor", 1.0, 1.0, False),
    ("bilateral_odor_ORNs_silenced", 1.0, 1.0, True),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "flybrain-full-odor-probe-v1",
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": digest(__file__),
        "config": LIFConfig().to_dict(),
        "protocol": {
            "pre_ms": 100,
            "odor_ms": 500,
            "post_ms": 200,
            "trace_interval_ms": 10,
            "stimulus": "Or42b/DM1 single-channel food-odor approximation",
            "max_current": 5.0,
            "half_concentration": 0.2,
            "background_current": 0.0,
            "weight_policy": "unchanged full-brain benchmark",
            "concentration_units": "dimensionless, not a measured chemical dose",
        },
        "conditions": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        started = time.perf_counter()
        mapping = build(args.annotations, args.data_dir / "proofread_root_ids_783.npy")
        model, report["model"] = load_fullbrain(args.data_dir)
        report["mapping"] = mapping
        index = NeuronIndex(model)
        groups = {name: index.resolve(ids) for name, ids in mapping["groups"].items()}
        selected = tuple(sorted({i for values in groups.values() for i in values}))
        selected_index = {i: j for j, i in enumerate(selected)}
        positions = {k: [selected_index[i] for i in v] for k, v in groups.items()}
        brain = CUDABackend(model, LIFConfig())  # No CPU fallback permitted.
        brain._stream.synchronize()
        cp = brain._cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        report["device"] = {
            "name": props["name"].decode(),
            "memory_bytes": props["totalGlobalMem"],
            "cupy": cp.__version__,
            "driver": cp.cuda.runtime.driverGetVersion(),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
        }
        report["load_seconds"] = time.perf_counter() - started
        report["cuda_kernel_sha256"] = digest(
            Path(__file__).parents[1] / "src/flybrain/backends/cuda.py"
        )
        odor = OlfactoryDrive(
            model,
            left_ids=mapping["groups"]["ORN_DM1_left"],
            right_ids=mapping["groups"]["ORN_DM1_right"],
        )
        zero = np.zeros(len(model.neuron_ids))
        # Same graph + backend state, through JSON, as in the full benchmark.
        rest = json.loads(json.dumps(brain.snapshot(), allow_nan=False))
        for name, left, right, silence in CONDITIONS:
            brain.restore(rest)
            if silence:
                brain.set_silenced(groups["ORN_DM1_left"] + groups["ORN_DM1_right"], True)
            current = odor.current(left=left, right=right)
            item = {
                "name": name,
                "antenna_concentrations": [left, right],
                "silence_input_neurons": silence,
                "spikes": {group: 0 for group in groups},
                "trace": [],
            }
            started = time.perf_counter()
            spike_totals = np.zeros(len(selected), dtype=np.int64)
            for tick in range(800):
                brain.step(current if 100 <= tick < 600 else zero)
                observed = brain.observe_selected(selected, ("spikes",))
                spike_totals += np.asarray(observed["spikes"], dtype=np.int64)
                if (tick + 1) % 10 == 0:
                    observed = brain.observe_selected(selected, ("voltage", "rates_hz"))
                    voltage = np.asarray(observed["voltage"])
                    rates = np.asarray(observed["rates_hz"])
                    if not np.isfinite(voltage).all() or not np.isfinite(rates).all():
                        raise ValueError("nonfinite neural response")
                    item["trace"].append(
                        {
                            "time_ms": tick + 1,
                            "groups": {
                                group: {
                                    "mean_rate_hz": float(rates[pos].mean()),
                                    "max_voltage": float(voltage[pos].max()),
                                }
                                for group, pos in positions.items()
                            },
                        }
                    )
            item["spikes"] = {k: int(spike_totals[p].sum()) for k, p in positions.items()}
            item["wall_seconds"] = time.perf_counter() - started
            report["conditions"].append(item)
            save()
            print(json.dumps({k: v for k, v in item.items() if k != "trace"}), flush=True)
        conditions = {item["name"]: item for item in report["conditions"]}
        for name in ("no_odor", "bilateral_odor_ORNs_silenced"):
            if any(conditions[name]["spikes"].values()):
                raise AssertionError("unexpected spikes in matched no-input control")
        for side in ("left", "right"):
            counts = conditions[f"{side}_odor"]["spikes"]
            if counts[f"ORN_DM1_{side}"] == 0:
                raise AssertionError("input stimulation failed to produce ORN spikes")
        report["findings"] = {
            "input_stimulation_and_silencing_checks": "passed",
            "bilateral_descending_spikes": conditions["bilateral_odor"]["spikes"]["descending"],
            "interpretation": (
                "Presence of descending spikes does not prove odor-directed navigation; "
                "absence means this preset has not demonstrated the required motor response. "
                "No behavior, banana recognition, flight or learning was tested here."
            ),
        }
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()


if __name__ == "__main__":
    main()
