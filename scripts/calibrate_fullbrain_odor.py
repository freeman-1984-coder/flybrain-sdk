"""Prespecified full-graph CUDA pilot: recurrent gain, side changes, recovery.

Four contact scales and both side orders are always reported. No food world,
decoder fitting, pruning or successful-run selection. This one-seed pilot can
identify candidates for later independent validation, not validate physiology.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from build_olfactory_map import build
from probe_synaptic_odor import input_events
from validate_fullbrain import digest, load_fullbrain

from flybrain.config import finite_number
from flybrain.experimental.synaptic import SynapticLIFConfig
from flybrain.experimental.synaptic_cuda import SynapticCUDA
from flybrain.neurons import NeuronIndex

CONTACT_SCALES_MV = (0.05, 0.10, 0.175, 0.275)
ORDERS = (("left", "right"), ("right", "left"))
PHASES = (
    ("rest", 50),
    ("first", 150),
    ("recovery_first", 200),
    ("second", 150),
    ("recovery_second", 250),
)


def phase_schedule(dt_ms, order):
    dt_ms = finite_number(dt_ms, "dt_ms")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    if tuple(order) not in ORDERS:
        raise ValueError("expected both left/right orders")
    tick = 0
    result = []
    for name, ms in PHASES:
        steps = round(ms / dt_ms)
        if steps <= 0 or not np.isclose(steps * dt_ms, ms, atol=1e-12, rtol=0):
            raise ValueError("phase durations must be exact multiples of dt_ms")
        side = order[0] if name == "first" else order[1] if name == "second" else None
        result.append(
            {
                "name": name,
                "side": side,
                "start_tick": tick,
                "end_tick": tick + steps,
                "duration_ms": ms,
            }
        )
        tick += steps
    return result


def summarize_run(run):
    """Separate transient side responses from cumulative bias and new tail spikes.

    DNa02 has exactly one annotated cell on each side in this pinned mapping.
    Recovery counts are observed spikes, never the exponentially smoothed rate.
    These descriptive metrics do not select a gain or certify navigation.
    """
    summary = {"stimuli": [], "recovery": []}
    for phase in run["phases"]:
        if phase["side"] is not None:
            side = phase["side"]
            other = "right" if side == "left" else "left"
            onset = phase["first_50ms_spikes"]
            total = phase["spikes"]
            ipsi, contra = f"DNa02_{side}", f"DNa02_{other}"
            later_ms = phase["duration_ms"] - 50
            summary["stimuli"].append(
                {
                    "phase": phase["name"],
                    "side": side,
                    "onset_ipsilateral_minus_contralateral_hz": (onset[ipsi] - onset[contra])
                    * 1000
                    / 50,
                    "later_ipsilateral_minus_contralateral_hz": (
                        (total[ipsi] - onset[ipsi]) - (total[contra] - onset[contra])
                    )
                    * 1000
                    / later_ms,
                    "total_right_minus_left_spikes": total["DNa02_right"] - total["DNa02_left"],
                }
            )
        elif phase["name"].startswith("recovery"):
            summary["recovery"].append(
                {
                    "phase": phase["name"],
                    "tail_window_ms": phase["tail_window_ms"],
                    "new_tail_spikes": phase["last_100ms_spikes"],
                }
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, seed = SynapticLIFConfig(), 20260914
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema": "flybrain-odor-gain-pilot-v1",
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.to_dict(),
        "numpy": np.__version__,
        "source_sha256": {
            str(p.relative_to(root)): digest(p)
            for p in (
                Path(__file__).resolve(),
                root / "scripts/build_olfactory_map.py",
                root / "scripts/probe_synaptic_odor.py",
                root / "scripts/validate_fullbrain.py",
                *sorted((root / "src/flybrain/experimental").glob("*.py")),
            )
        },
        "protocol": {
            "contact_scales_mv": list(CONTACT_SCALES_MV),
            "orders": ORDERS,
            "seed": seed,
            "rng": "NumPy PCG64",
            "input_rate_hz": 150,
            "input_jump_mv": 68.75,
            "background_input": 0,
            "sensory_model": "Bernoulli p=rate*dt; DM1 single-channel approximation",
            "input_jump_independent_of_contact_scale": True,
            "graph_policy": "every official neuron and aggregate edge retained",
            "state_policy": "reset only between runs; never reset between phases",
            "scope": "one-seed pilot; no fitted decoder, behavior or biological validation",
            "schedule": phase_schedule(config.dt_ms, ORDERS[0]),
            "analysis": (
                "onset 50 ms and later ipsi-minus-contra DNa02 Hz; new recovery tail spikes"
            ),
            "selection_policy": "report all eight runs; no automatic gain or decoder selection",
        },
        "models": [],
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = args.output.with_suffix(".partial.json")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output)

    save()
    try:
        mapping = build(args.annotations, args.data_dir / "proofread_root_ids_783.npy")
        report["mapping"] = mapping
        report["mapping_input_units_note"] = (
            "Protocol defines actual mV inputs, not old map labels."
        )
        input_ids = mapping["groups"]["ORN_DM1_left"] + mapping["groups"]["ORN_DM1_right"]
        left_n, right_n = (len(mapping["groups"][f"ORN_DM1_{s}"]) for s in ("left", "right"))
        for scale in CONTACT_SCALES_MV:
            model, provenance = load_fullbrain(args.data_dir, contact_mv=scale)
            report["models"].append(provenance)
            index = NeuronIndex(model)
            inputs = np.asarray(index.resolve(input_ids), dtype=np.int64)
            groups = {k: index.resolve(v) for k, v in mapping["groups"].items()}
            selected = tuple(sorted({i for values in groups.values() for i in values}))
            offsets = {i: j for j, i in enumerate(selected)}
            positions = {k: [offsets[i] for i in values] for k, values in groups.items()}
            brain = SynapticCUDA(model, config, input_ids=input_ids, weight_units="mV")
            cp = brain._cp
            props = cp.cuda.runtime.getDeviceProperties(brain._device.id)
            report["device"] = {
                "name": props["name"].decode(),
                "memory_bytes": props["totalGlobalMem"],
                "cupy": cp.__version__,
                "driver": cp.cuda.runtime.driverGetVersion(),
                "runtime": cp.cuda.runtime.runtimeGetVersion(),
            }
            rest = json.loads(json.dumps(brain.snapshot(), allow_nan=False))
            jumps = np.zeros(brain.n)
            for order in ORDERS:
                brain.restore(rest)
                rng = np.random.default_rng(seed)
                run = {
                    "contact_mv": scale,
                    "order": list(order),
                    "graph_sha256": provenance["graph_arrays_sha256"],
                    "selected_neuron_ids": [model.neuron_ids[i] for i in selected],
                    "phases": [],
                    "trace": [],
                }
                started = time.perf_counter()
                for phase in phase_schedule(config.dt_ms, order):
                    totals = np.zeros(len(selected), dtype=np.int64)
                    onset = np.zeros(len(selected), dtype=np.int64)
                    tail = np.zeros(len(selected), dtype=np.int64)
                    input_count = np.zeros(2, dtype=np.int64)
                    for tick in range(phase["start_tick"], phase["end_tick"]):
                        events = input_events(
                            rng,
                            left_n,
                            right_n,
                            150 * config.dt_ms / 1000,
                            phase["side"] == "left",
                            phase["side"] == "right",
                        )
                        jumps[inputs] = events * 68.75
                        input_count += [int(events[:left_n].sum()), int(events[left_n:].sum())]
                        brain.step(jumps)
                        spikes = np.asarray(
                            brain.observe_selected(selected, ("spikes",))["spikes"], dtype=np.int64
                        )
                        totals += spikes
                        if tick - phase["start_tick"] < round(50 / config.dt_ms):
                            onset += spikes
                        if tick >= phase["end_tick"] - round(100 / config.dt_ms):
                            tail += spikes
                        if (tick + 1) % 100 == 0:
                            rates = np.asarray(
                                brain.observe_selected(selected, ("rates_hz",))["rates_hz"]
                            )
                            run["trace"].append(
                                {
                                    "time_ms": (tick + 1) * config.dt_ms,
                                    "phase": phase["name"],
                                    "mean_rates_hz": {
                                        k: float(rates[pos].mean()) for k, pos in positions.items()
                                    },
                                }
                            )
                    item = {
                        **phase,
                        "input_events": input_count.tolist(),
                        "spikes": {k: int(totals[pos].sum()) for k, pos in positions.items()},
                        "first_50ms_spikes": {
                            k: int(onset[pos].sum()) for k, pos in positions.items()
                        },
                        "last_100ms_spikes": {
                            k: int(tail[pos].sum()) for k, pos in positions.items()
                        },
                        "tail_window_ms": min(100, phase["duration_ms"]),
                        "per_neuron_spikes": totals.tolist(),
                        "first_50ms_per_neuron_spikes": onset.tolist(),
                    }
                    if phase["name"] == "rest" and totals.any():
                        raise AssertionError("fresh no-input baseline must remain silent")
                    run["phases"].append(item)
                run["wall_seconds"] = time.perf_counter() - started
                run["summary"] = summarize_run(run)
                report["runs"].append(run)
                save()
                print(
                    json.dumps(
                        {
                            "contact_mv": scale,
                            "order": order,
                            "wall_seconds": run["wall_seconds"],
                            "spikes": {p["name"]: p["spikes"] for p in run["phases"]},
                        }
                    ),
                    flush=True,
                )
            brain._stream.synchronize()
            del brain, model, rest, index
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        if len(report["runs"]) != 8:
            raise AssertionError("all eight prespecified runs must complete")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()


if __name__ == "__main__":
    main()
