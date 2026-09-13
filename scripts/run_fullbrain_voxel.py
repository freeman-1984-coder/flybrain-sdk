"""Record a complete FlyWire CUDA brain controlling a ground voxel body.

Experimental engineering readout: DNa02 rate difference controls turn; summed
rates control speed. The speed mapping is not biologically established. No
target coordinate, odor concentration or rescue reflex enters the motor readout.
Both active and input-silenced runs are retained, regardless of food contact.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from build_olfactory_map import build
from validate_fullbrain import digest, load_fullbrain

from flybrain.experimental.synaptic import SynapticLIFConfig
from flybrain.experimental.synaptic_cuda import SynapticCUDA
from flybrain.experimental.voxel_arena import VoxelArena
from flybrain.neurons import NeuronIndex


def motor_readout(left_hz, right_hz):
    """Fixed, untrained engineering gains. Inputs are neural firing rates only."""
    return {
        "speed": min(3.0, max(0.0, (left_hz + right_hz) * 0.03)),
        "turn": 2.0 * math.tanh((right_hz - left_hz) / 40.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config, seed = SynapticLIFConfig(), 20260913
    report = {
        "schema": "flybrain-voxel-recording-v1",
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.to_dict(),
        "seed": seed,
        "source_sha256": {
            str(p.relative_to(root)): digest(p)
            for p in (
                Path(__file__).resolve(),
                root / "scripts/validate_fullbrain.py",
                root / "scripts/build_olfactory_map.py",
                root / "src/flybrain/olfaction.py",
                *sorted((root / "src/flybrain/experimental").glob("*.py")),
            )
        },
        "protocol": {
            "duration_ms": 2000,
            "body_step_ms": 20,
            "neural_steps_per_frame": 200,
            "input_jump_mv": 68.75,
            "contact_mv": 0.275,
            "odor_rate_hz": "180*c/(0.2+c); c is a dimensionless antenna sample",
            "stimulus": "DM1 single-channel food odor, not a full banana mixture",
            "readout": "speed=clip(0.03*(DNa02_L+DNa02_R),0,3); turn=2*tanh((DNa02_R-DNa02_L)/40)",
            "units": "rates Hz; speed game units/s; turn radians/s",
            "limitations": [
                "Untrained engineering readout; forward-speed mapping not established biologically",
                "Kinematic walking body, not flight or musculoskeletal simulation",
                "Toy Gaussian odor field without wind, turbulence or obstacle occlusion",
                "Recorded GPU frames, not live browser inference",
                "No direct motor input, background drive, odor-bearing controller or rescue reflex",
                "Success or failure of both runs reported without seed selection",
            ],
        },
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
        model, report["model"] = load_fullbrain(args.data_dir, contact_mv=0.275)
        report["mapping"] = mapping
        # The map's historical dimensionless-current label does not configure this engine.
        report["mapping_input_units_note"] = "Actual input units and strengths are in protocol."
        index = NeuronIndex(model)
        left_ids, right_ids = (mapping["groups"][f"ORN_DM1_{s}"] for s in ("left", "right"))
        input_ids = left_ids + right_ids
        inputs = np.asarray(index.resolve(input_ids), dtype=np.int64)
        outputs = index.resolve(mapping["groups"]["DNa02_left"] + mapping["groups"]["DNa02_right"])
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
        for silence in (False, True):
            world = VoxelArena(food=(-1.6, 0.5, -0.6))
            brain.restore(rest)
            brain.set_silenced(inputs, silence)
            rng = np.random.default_rng(seed)
            run = {
                "name": "ORNs_silenced" if silence else "neural_control",
                "initial": world.scene(),
                "frames": [],
            }
            started = time.perf_counter()

            def advance_frame():
                odor = world.observe()
                rates = np.asarray(
                    [
                        180 * odor[f"odor_{side}"] / (0.2 + odor[f"odor_{side}"])
                        for side in ("left", "right")
                    ]
                )
                probability = np.repeat(rates, [len(left_ids), len(right_ids)]) * 0.0001
                input_count = np.zeros(2, dtype=np.int64)
                output_count = np.zeros(2, dtype=np.int64)
                for _ in range(200):
                    events = rng.random(len(inputs)) < probability
                    jumps[inputs] = events * 68.75
                    input_count += [
                        int(events[: len(left_ids)].sum()),
                        int(events[len(left_ids) :].sum()),
                    ]
                    brain.step(jumps)
                    output_count += np.asarray(
                        brain.observe_selected(outputs, ("spikes",))["spikes"], dtype=np.int64
                    )
                neural = brain.observe_selected(outputs, ("rates_hz",))["rates_hz"]
                action = motor_readout(*neural)
                world.apply(action, 20)
                return {
                    **world.scene(),
                    "odor": odor,
                    "input_rate_hz": rates.tolist(),
                    "input_events": input_count.tolist(),
                    "output_spikes": output_count.tolist(),
                    "output_rates_hz": list(neural),
                    "action": action,
                }

            for frame in range(100):
                record = advance_frame()
                run["frames"].append(record)
                if frame == 19:
                    # A saved body state plus brain delay buffers and sensory RNG,
                    # restored before the next frame. No hidden renderer state.
                    saved_brain = json.loads(json.dumps(brain.snapshot(), allow_nan=False))
                    saved_world = json.loads(json.dumps(world.snapshot(), allow_nan=False))
                    saved_rng = json.loads(json.dumps(rng.bit_generator.state))
                    expected_frame = advance_frame()
                    expected_brain = brain.snapshot()
                    expected_rng = rng.bit_generator.state
                    brain.restore(saved_brain)
                    world = VoxelArena.from_snapshot(saved_world)
                    rng.bit_generator.state = saved_rng
                    if (
                        advance_frame() != expected_frame
                        or brain.snapshot() != expected_brain
                        or rng.bit_generator.state != expected_rng
                    ):
                        raise AssertionError("joint brain/world/RNG replay differed")
                    brain.restore(saved_brain)
                    world = VoxelArena.from_snapshot(saved_world)
                    rng.bit_generator.state = saved_rng
                    run["checkpoint_replay"] = {
                        "at_frame": 20,
                        "future_frames": 1,
                        "full_brain_world_and_rng_exact": True,
                    }
                if (frame + 1) % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "run": run["name"],
                                "frame": frame + 1,
                                "position": world.scene()["position"],
                                "rates_hz": record["output_rates_hz"],
                            }
                        ),
                        flush=True,
                    )
            run["wall_seconds"] = time.perf_counter() - started
            run["final_distance"] = math.dist(world.scene()["position"], world.food)
            run["initial_distance"] = math.dist(run["initial"]["position"], world.food)
            run["reached_food"] = world.reached_food
            if silence and world.scene()["position"] != run["initial"]["position"]:
                raise AssertionError("silenced control moved unexpectedly")
            report["runs"].append(run)
            save()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()


if __name__ == "__main__":
    main()
