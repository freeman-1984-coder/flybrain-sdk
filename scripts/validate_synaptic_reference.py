"""Independent Brian2 oracle for the experimental mV synaptic CPU engine.

Install Brian2==2.9.0 and numpy==1.26.4 separately; no Brian2 runtime dependency.
Tests explicit scheduled input events (not matching two different RNGs).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain import Connectome, Synapse
from flybrain.experimental.synaptic import SynapticCPU, SynapticLIFConfig


def compare_case(config):
    import brian2 as b

    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = config.dt_ms * b.ms
    # Parallel excitatory edges, inhibitory input, recurrent excitation, autapse,
    # and a disconnected neuron. Counts below are fixture choices, not fly data.
    edges = [
        (0, 2, 40.0),
        (0, 2, 15.0),
        (1, 2, -33.0),
        (2, 3, 55.0),
        (3, 2, 10.0),
        (2, 2, 2.0),
        (3, 4, -20.0),
    ]
    model = Connectome(
        "oracle-fixture",
        tuple(str(i) for i in range(6)),
        tuple(Synapse(*edge) for edge in edges),
        {},
        {},
        {},
    )
    cpu = SynapticCPU(model, config, input_ids=("0", "1"), weight_units="mV")
    n_steps = round(100 / config.dt_ms)
    # Rounding here constructs a known clock-aligned experimental schedule.
    events = sorted(
        [(round(t / config.dt_ms), 0) for t in [1.0, 5.0, 10.0, 17.0, 25.0, 40.0]]
        + [(round(t / config.dt_ms), 1) for t in [6.0, 17.0, 23.0, 40.0]]
        + [(round(1 / config.dt_ms) + offset, 0) for offset in (1, 2)]
    )
    injected = np.zeros((n_steps, 6))
    for tick, neuron in events:
        injected[tick, neuron] = 68.75  # 0.275 mV * 250 in the published stimulus.
    namespace = {
        "v_rest": config.rest_mv * b.mV,
        "v_reset": config.reset_mv * b.mV,
        "v_threshold": config.threshold_mv * b.mV,
        "tau_mem": config.membrane_tau_ms * b.ms,
        "tau_syn": config.synapse_tau_ms * b.ms,
    }
    # Use the SAME symbol when the time constants are equal so Brian2 can
    # simplify the degenerate linear system analytically, rather than evaluate
    # a generic expression containing division by (tau_mem - tau_syn) at zero.
    syn_tau_symbol = "tau_mem" if config.synapse_tau_ms == config.membrane_tau_ms else "tau_syn"
    group = b.NeuronGroup(
        6,
        f"""dv/dt = (v_rest - v + g)/tau_mem : volt (unless refractory)
              dg/dt = -g/{syn_tau_symbol} : volt (unless refractory)
              rfc : second""",
        threshold="v > v_threshold",
        reset="v = v_reset; g = 0*mV",
        refractory="rfc",
        method="linear",
        namespace=namespace,
    )
    group.v = config.rest_mv * b.mV
    group.g = 0 * b.mV
    group.rfc = config.refractory_ms * b.ms
    group.rfc[:2] = 0 * b.ms
    synapses = b.Synapses(
        group, group, "weight : volt", on_pre="g_post += weight", delay=config.delay_ms * b.ms
    )
    synapses.connect(i=[e[0] for e in edges], j=[e[1] for e in edges])
    synapses.weight = np.array([e[2] for e in edges]) * b.mV
    source = b.SpikeGeneratorGroup(
        2, [e[1] for e in events], np.array([e[0] for e in events]) * config.dt_ms * b.ms
    )
    external = b.Synapses(source, group, on_pre="v_post += 68.75*mV")
    external.connect(i=[0, 1], j=[0, 1])
    monitor = b.StateMonitor(group, ("v", "g"), record=True, when="end")
    spikes = b.SpikeMonitor(group)
    b.Network(group, synapses, source, external, monitor, spikes).run(n_steps * config.dt_ms * b.ms)
    expected_v = np.array(monitor.v / b.mV).T
    expected_g = np.array(monitor.g / b.mV).T
    expected_spikes = np.zeros((n_steps, 6), dtype=bool)
    expected_ticks = np.rint(np.array(spikes.t / b.ms) / config.dt_ms).astype(int)
    expected_spikes[expected_ticks, np.array(spikes.i)] = True
    actual_v, actual_g, actual_spikes = [], [], []
    checkpoint = None
    for tick in range(n_steps):
        cpu.step(injected[tick])
        observed = cpu.observe_selected(tuple(range(6)))
        actual_v.append(observed["voltage_mv"])
        actual_g.append(observed["synaptic_mv"])
        actual_spikes.append(observed["spikes"])
        # Save with the first spike still queued for transmission in default case.
        if tick == round(1 / config.dt_ms) + 2:
            checkpoint = json.loads(json.dumps(cpu.snapshot(), allow_nan=False))
    actual_v, actual_g = np.asarray(actual_v), np.asarray(actual_g)
    np.testing.assert_allclose(actual_v, expected_v, rtol=0, atol=1e-9)
    np.testing.assert_allclose(actual_g, expected_g, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(actual_spikes, expected_spikes)
    assert expected_spikes[:, 2:4].any(), "oracle must exercise downstream propagation"
    final = cpu.snapshot()
    cpu.restore(checkpoint)
    for tick in range(cpu.tick, n_steps):
        cpu.step(injected[tick])
    assert cpu.snapshot() == final, "delay-buffer checkpoint replay differs"
    return {
        "config": config.to_dict(),
        "ticks": n_steps,
        "voltage_max_error_mv": float(np.max(np.abs(actual_v - expected_v))),
        "synaptic_max_error_mv": float(np.max(np.abs(actual_g - expected_g))),
        "spikes_per_cell": expected_spikes.sum(axis=0).tolist(),
        "spike_times_equal": True,
        "checkpoint_replay_exact": True,
    }


def main():
    import brian2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "running",
        "oracle": "Brian2",
        "version": brian2.__version__,
        "numpy": np.__version__,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["implementation_sha256"] = hashlib.sha256(
        (Path(__file__).parents[1] / "src/flybrain/experimental/synaptic.py").read_bytes()
    ).hexdigest()
    report["validation_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        for config in (
            SynapticLIFConfig(),
            SynapticLIFConfig(dt_ms=0.2),
            SynapticLIFConfig(delay_ms=0),
            SynapticLIFConfig(synapse_tau_ms=20),
        ):
            result = compare_case(config)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
