"""Full FlyWire CUDA odor experiment with separately versioned mV dynamics.

Keeps every official neuron/edge. Seeded discrete Poisson inputs target only
annotated DM1 ORNs. No CPU fallback, background current, navigation controller,
parameter search, graph pruning or success-based selection of conditions.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from build_olfactory_map import build
from validate_fullbrain import digest, load_fullbrain

from flybrain.experimental.synaptic import SynapticCPU, SynapticLIFConfig
from flybrain.experimental.synaptic_cuda import SynapticCUDA
from flybrain.neurons import NeuronIndex

CONDITIONS = (
    ("no_odor", False, False, False),
    ("left_odor", True, False, False),
    ("right_odor", False, True, False),
    ("bilateral_odor", True, True, False),
    ("bilateral_odor_ORNs_silenced", True, True, True),
)


def input_events(rng, left_count, right_count, probability, left, right):
    """Matched random draws even when a side is disabled; one event/cell/tick.

    Bernoulli p=rate*dt is the discrete PoissonGroup approximation. It is not
    an exact continuous-time Poisson process or a calibrated chemical dose.
    """
    events = rng.random(left_count + right_count) < probability
    if not left:
        events[:left_count] = False
    if not right:
        events[left_count:] = False
    return events


def compare(cpu, gpu):
    expected, actual = cpu.snapshot(), gpu.snapshot()
    for field in (
        "engine",
        "fingerprint",
        "tick",
        "spikes",
        "history",
        "silenced",
        "last_spike_tick",
    ):
        if expected[field] != actual[field]:
            raise AssertionError(f"CPU/GPU mismatch: {field}")
    errors = {}
    for field in ("voltage_mv", "synaptic_mv", "rates_hz"):
        a, b = np.asarray(expected[field]), np.asarray(actual[field])
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=0, err_msg=field)
        errors[field] = float(np.max(np.abs(a - b)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = SynapticLIFConfig()
    seed, rate_hz, jump_mv, contact_mv = 20260913, 150.0, 68.75, 0.275
    probability = rate_hz * config.dt_ms / 1000
    report = {
        "schema": "flybrain-full-synaptic-odor-v1",
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.to_dict(),
        "numpy": np.__version__,
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): digest(p)
            for p in (
                Path(__file__).resolve(),
                Path(__file__).with_name("validate_fullbrain.py").resolve(),
                Path(__file__).with_name("build_olfactory_map.py").resolve(),
                *sorted(
                    (Path(__file__).resolve().parents[1] / "src/flybrain/experimental").glob("*.py")
                ),
            )
        },
        "protocol": {
            "pre_ms": 100,
            "odor_ms": 500,
            "post_ms": 200,
            "trace_interval_ms": 10,
            "seed": seed,
            "rng": "NumPy PCG64",
            "input_rate_hz": rate_hz,
            "input_jump_mv": jump_mv,
            "contact_mv": contact_mv,
            "background_input": 0,
            "sensory_model": "Bernoulli p=rate*dt, at most one input event/cell/tick",
            "stimulus": "Or42b/DM1 single-channel food-odor approximation",
            "calibration": "nominal research parameters; not a measured banana dose",
            "input_refractory_ms": 0,
            "parameter_source": (
                "https://github.com/philshiu/Drosophila_brain_model/"
                "blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py"
            ),
        },
        "conditions": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".partial.json")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        mapping = build(args.annotations, args.data_dir / "proofread_root_ids_783.npy")
        model, report["model"] = load_fullbrain(args.data_dir, contact_mv=contact_mv)
        report["mapping"] = mapping
        index = NeuronIndex(model)
        groups = {name: index.resolve(ids) for name, ids in mapping["groups"].items()}
        left_ids, right_ids = (mapping["groups"][f"ORN_DM1_{side}"] for side in ("left", "right"))
        input_ids = left_ids + right_ids
        inputs = np.asarray(index.resolve(input_ids), dtype=np.int64)
        selected = tuple(sorted({i for values in groups.values() for i in values}))
        offsets = {i: j for j, i in enumerate(selected)}
        positions = {name: [offsets[i] for i in values] for name, values in groups.items()}
        brain = SynapticCUDA(model, config, input_ids=input_ids, weight_units="mV")
        props = brain._cp.cuda.runtime.getDeviceProperties(brain._device.id)
        report["device"] = {
            "name": props["name"].decode(),
            "memory_bytes": props["totalGlobalMem"],
            "cupy": brain._cp.__version__,
            "driver": brain._cp.cuda.runtime.driverGetVersion(),
            "runtime": brain._cp.cuda.runtime.runtimeGetVersion(),
        }
        rest = json.loads(json.dumps(brain.snapshot(), allow_nan=False))
        jumps = np.zeros(brain.n)
        cpu = SynapticCPU(model, config, input_ids=input_ids, weight_units="mV")
        rng = np.random.default_rng(seed)
        parity = {"ticks": 200, "observations": [], "input_events": 0}
        started = time.perf_counter()
        for tick in range(200):
            events = input_events(rng, len(left_ids), len(right_ids), probability, True, True)
            jumps[inputs] = events * jump_mv
            parity["input_events"] += int(events.sum())
            cpu.step(jumps)
            brain.step(jumps)
            if (tick + 1) % 40 == 0:
                parity["observations"].append({"tick": tick + 1, "errors": compare(cpu, brain)})
        if parity["input_events"] == 0:
            raise AssertionError("full-graph parity must exercise active sensory input")
        # Round-trip device state through JSON and replay the same input RNG.
        checkpoint = json.loads(json.dumps(brain.snapshot(), allow_nan=False))
        rng_state = json.loads(json.dumps(rng.bit_generator.state))

        def replay():
            for _ in range(20):
                events = input_events(rng, len(left_ids), len(right_ids), probability, True, True)
                jumps[inputs] = events * jump_mv
                brain.step(jumps)
            return brain.snapshot()

        expected = replay()
        brain.restore(checkpoint)
        rng.bit_generator.state = rng_state
        if replay() != expected:
            raise AssertionError("full-graph device + input RNG checkpoint replay differed")
        parity.update(
            status="passed",
            json_checkpoint_replay_ticks=20,
            wall_seconds=time.perf_counter() - started,
        )
        report["full_graph_cpu_cuda_parity"] = parity
        save()
        print("Full-graph active parity and delayed checkpoint replay passed", flush=True)
        del cpu, checkpoint, expected
        for name, left, right, silence in CONDITIONS:
            brain.restore(rest)
            if silence:
                brain.set_silenced(inputs, True)
            rng = np.random.default_rng(seed)
            totals = np.zeros(len(selected), dtype=np.int64)
            item = {"name": name, "silenced": silence, "input_events": [0, 0], "trace": []}
            started = time.perf_counter()
            for tick in range(8000):
                active = 1000 <= tick < 6000
                events = input_events(
                    rng,
                    len(left_ids),
                    len(right_ids),
                    probability,
                    left and active,
                    right and active,
                )
                jumps[inputs] = events * jump_mv
                item["input_events"][0] += int(events[: len(left_ids)].sum())
                item["input_events"][1] += int(events[len(left_ids) :].sum())
                brain.step(jumps)
                totals += np.asarray(
                    brain.observe_selected(selected, ("spikes",))["spikes"], dtype=np.int64
                )
                if (tick + 1) % 100 == 0:
                    observed = brain.observe_selected(selected, ("voltage_mv", "rates_hz"))
                    voltage, rates = (np.asarray(observed[k]) for k in ("voltage_mv", "rates_hz"))
                    item["trace"].append(
                        {
                            "time_ms": (tick + 1) * config.dt_ms,
                            "groups": {
                                group: {
                                    "mean_rate_hz": float(rates[pos].mean()),
                                    "max_voltage_mv": float(voltage[pos].max()),
                                    "cumulative_spikes": int(totals[pos].sum()),
                                }
                                for group, pos in positions.items()
                            },
                        }
                    )
            item["spikes"] = {group: int(totals[pos].sum()) for group, pos in positions.items()}
            item["wall_seconds"] = time.perf_counter() - started
            report["conditions"].append(item)
            save()
            print(json.dumps({k: v for k, v in item.items() if k != "trace"}), flush=True)
        conditions = {item["name"]: item for item in report["conditions"]}
        for name in ("no_odor", "bilateral_odor_ORNs_silenced"):
            if any(conditions[name]["spikes"].values()):
                raise AssertionError("spikes in matched no-input control")
        for side in ("left", "right"):
            if conditions[f"{side}_odor"]["spikes"][f"ORN_DM1_{side}"] == 0:
                raise AssertionError("stimulated ORNs did not spike")
        report["findings"] = {
            "input_and_silencing_controls": "passed",
            "bilateral_descending_spikes": conditions["bilateral_odor"]["spikes"]["descending"],
            "interpretation": (
                "Report includes absent responses. Descending spikes alone do not prove "
                "odor-directed navigation. No banana recognition, flight or learning tested."
            ),
        }
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()


if __name__ == "__main__":
    main()
