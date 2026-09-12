"""Opt-in full FlyWire v783 hardware benchmark; never substitutes a subgraph.

Uses the SDK's unmodified CPUBackend / CUDABackend and Connectome classes.
Download the two official Zenodo files before running. No network access here.
"""

import argparse
import gc
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np

from flybrain.backends.cpu import CPUBackend
from flybrain.backends.cuda import CUDABackend
from flybrain.config import LIFConfig
from flybrain.model import Connectome, Synapse

SOURCES = {
    "proofread_root_ids_783.npy": "e0e6c19732fd8c7a4e39a2d170105421",
    "proofread_connections_783.feather": "f48f972d262323a102aed49af1396b8a",
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def map_source_ids(ids, source_ids):
    """Match large IDs exactly; mixed uint64/int64 search may promote to float64."""
    ids, source_ids = np.asarray(ids), np.asarray(source_ids)
    if ids.dtype.kind not in "iu" or source_ids.dtype.kind not in "iu":
        raise ValueError("Source neuron IDs must have integer storage")
    if (ids < 0).any() or (source_ids < 0).any():
        raise ValueError("Source neuron IDs must be nonnegative")
    ids = ids.astype(np.uint64, copy=False)
    source_ids = source_ids.astype(np.uint64, copy=False)
    indices = np.searchsorted(ids, source_ids)
    if (indices >= len(ids)).any() or not np.array_equal(ids[indices], source_ids):
        raise ValueError("Source edge references an unknown neuron")
    return indices


def load_fullbrain(data_dir):
    import pyarrow.feather as feather

    sources = {}
    for filename, expected in SOURCES.items():
        path = data_dir / filename
        if digest(path, "md5") != expected:
            raise ValueError(f"Official source checksum mismatch: {filename}")
        sources[filename] = {"md5": expected, "sha256": digest(path), "bytes": path.stat().st_size}
    ids = np.sort(np.load(data_dir / "proofread_root_ids_783.npy", allow_pickle=False))
    if len(ids) != 139255 or len(np.unique(ids)) != len(ids):
        raise ValueError("Expected all 139255 unique official proofread IDs")
    nts = ["ach_avg", "gaba_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
    columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count", *nts]
    table = feather.read_table(data_dir / "proofread_connections_783.feather", columns=columns)
    raw_pre = table[columns[0]].to_numpy()
    raw_post = table[columns[1]].to_numpy()
    pre, post = map_source_ids(ids, raw_pre), map_source_ids(ids, raw_post)
    counts = table["syn_count"].to_numpy().astype(np.int64)
    if (counts <= 0).any():
        raise ValueError("Source synapse counts must be positive")
    # Explicit assumed Dale-like sign per source neuron: weighted mean NT scores
    # over all outgoing contacts. GABA/glutamate negative; other labels positive.
    # This approximation does not implement receptors or neuromodulation.
    scores = np.zeros((len(ids), len(nts)))
    nonfinite = 0
    for i, name in enumerate(nts):
        values = table[name].to_numpy()
        nonfinite += int((~np.isfinite(values)).sum())
        scores[:, i] = np.bincount(pre, weights=np.nan_to_num(values) * counts, minlength=len(ids))
    labels = scores.argmax(axis=1)
    signs = np.where(np.isin(labels, [1, 2]), -1.0, 1.0)
    incoming = np.bincount(post, weights=counts, minlength=len(ids))
    weights = signs[pre] * 2.5 * counts / incoming[post]
    report = {
        "dataset": "FlyWire v783 / Zenodo 10676866",
        "source_url": "https://zenodo.org/records/10676866",
        "sources": sources,
        "neurons": len(ids),
        "source_rows": len(table),
        "retained_rows": len(table),
        "discarded_rows": 0,
        "synaptic_contacts": int(counts.sum()),
        "rows_are": "neuron-pair/neuropil aggregates, retained as parallel weighted edges",
        "isolated_neurons": int(
            (
                np.bincount(pre, minlength=len(ids)) + np.bincount(post, minlength=len(ids)) == 0
            ).sum()
        ),
        "nt_score_nonfinite_entries": nonfinite,
        "neurons_without_nt_evidence": int((scores.sum(axis=1) == 0).sum()),
        "assumed_sign_counts": {
            "positive": int((signs > 0).sum()),
            "negative": int((signs < 0).sum()),
        },
        "weight_policy": "incoming abs-sum 2.5; GABA/GLUT negative, others/unknown positive",
        "minimum_contacts": 1,
        "additional_pruning": False,
        "graph_arrays_sha256": hashlib.sha256(
            ids.tobytes()
            + pre.astype("<i4").tobytes()
            + post.astype("<i4").tobytes()
            + weights.astype("<f8").tobytes()
        ).hexdigest(),
    }
    del table, raw_pre, raw_post, counts, scores, values
    gc.collect()
    # A generator avoids holding a second complete tuple during validation.
    model = Connectome(
        "flywire-full-v783",
        tuple(str(int(n)) for n in ids),
        (Synapse(int(a), int(b), float(w)) for a, b, w in zip(pre, post, weights)),
        {},
        {},
        {"dataset": "FlyWire v783", "status": "assumed LIF benchmark"},
    )
    return model, report


def compare(cpu, gpu):
    expected, actual = cpu.snapshot(), gpu.snapshot()
    for field in ("spikes", "refractory", "silenced"):
        np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
    errors = {}
    for field in ("voltage", "rates_hz"):
        a, b = np.asarray(actual[field]), np.asarray(expected[field])
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-10, err_msg=field)
        errors[field] = float(np.max(np.abs(a - b)))
    assert actual["tick"] == expected["tick"]
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stability-ticks", type=int, default=10000)
    args = p.parse_args()
    if args.stability_ticks < 1000:
        p.error("Full benchmark requires at least 1000 stability ticks")
    report = {
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "script_sha256": digest(__file__),
        "config": LIFConfig().to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        started = time.perf_counter()
        model, report["model"] = load_fullbrain(args.data_dir)
        report["model_load_seconds"] = time.perf_counter() - started
        print(json.dumps(report["model"]), flush=True)
        print("Constructing SDK CPU and CUDA backends", flush=True)
        started = time.perf_counter()
        cpu = CPUBackend(model, LIFConfig())
        report["cpu_init_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        gpu = CUDABackend(model, LIFConfig())
        gpu._stream.synchronize()
        report["cuda_init_and_compile_seconds"] = time.perf_counter() - started
        cp = gpu._cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        report["device"] = {
            "name": props["name"].decode(),
            "memory_bytes": props["totalGlobalMem"],
            "cupy": cp.__version__,
            "driver": cp.cuda.runtime.driverGetVersion(),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
        }
        report["cuda_kernel_sha256"] = digest(
            Path(__file__).parents[1] / "src/flybrain/backends/cuda.py"
        )
        n = len(model.neuron_ids)
        zero = np.zeros(n)
        for _ in range(20):
            cpu.step(zero)
            gpu.step(zero)
        compare(cpu, gpu)
        assert not any(gpu.snapshot()["spikes"])
        report["rest_check"] = "20 ticks; all neurons at rest, no spikes"
        rng = np.random.default_rng(20260913)
        currents = rng.uniform(0.5, 2.5, size=(8, n))
        parity, spikes = [], 0
        for tick in range(200):
            current = currents[(tick // 25) % 8]
            cpu.step(current)
            gpu.step(current)
            spikes += int(cpu._spikes.sum())
            if tick % 25 == 24:
                parity.append({"tick": gpu.tick, **compare(cpu, gpu)})
        assert spikes > 0, "Parity must exercise active spikes, not only rest"
        report["parity"] = {
            "ticks": 200,
            "all_neurons_compared_every": 25,
            "exact_spikes_and_refractory": True,
            "atol": 1e-10,
            "total_spikes": spikes,
            "checks": parity,
        }
        print("Full-network CPU/CUDA parity passed", flush=True)
        save()
        checkpoint = json.loads(json.dumps(gpu.snapshot(), allow_nan=False))
        for tick in range(30):
            gpu.step(currents[tick % 8])
        end = gpu.snapshot()
        gpu.restore(checkpoint)
        for tick in range(30):
            gpu.step(currents[tick % 8])
        assert gpu.snapshot() == end
        report["checkpoint_replay_exact"] = True
        timings = {}
        for name, backend in (("cpu", cpu), ("cuda", gpu)):
            for _ in range(100):
                backend.step(currents[0])
            durations = []
            for repeat in range(3):
                start = time.perf_counter()
                for tick in range(1000):
                    backend.step(currents[(tick // 125) % 8])
                if name == "cuda":
                    gpu._stream.synchronize()
                durations.append(time.perf_counter() - start)
                print(
                    f"{name} repeat {repeat}: {durations[-1]:.3f}s per simulated second", flush=True
                )
            timings[name] = {
                "wall_seconds": durations,
                "median_seconds": float(np.median(durations)),
                "simulated_ms_per_repeat": 1000,
                "warmup_ticks": 100,
                "includes": "SDK step, host input upload and per-tick error sync",
            }
        report["timings"] = timings
        save()
        trace, population_spikes = [], 0
        ever_active = np.zeros(n, dtype=bool)
        start = time.perf_counter()
        # Inspect actual device state under its owning stream every tick, accumulating
        # spikes on device; periodic complete snapshots check finite bounded states.
        with gpu._device, gpu._stream:
            active = cp.zeros(n, dtype=cp.bool_)
            total = cp.zeros((), dtype=cp.int64)
        for tick in range(args.stability_ticks):
            gpu.step(currents[(tick // 125) % 8])
            with gpu._device, gpu._stream:
                active |= gpu._spikes
                total += gpu._spikes.sum()
            if tick % 100 == 99:
                state = gpu.snapshot()
                voltage, rates = np.asarray(state["voltage"]), np.asarray(state["rates_hz"])
                assert np.isfinite(voltage).all() and np.isfinite(rates).all()
                assert (voltage < 1.0).all() and (rates >= 0).all() and (rates <= 1000).all()
                trace.append(
                    {
                        "tick": gpu.tick,
                        "voltage_min": float(voltage.min()),
                        "voltage_max": float(voltage.max()),
                        "mean_rate_hz": float(rates.mean()),
                        "spiking_neurons": int(sum(state["spikes"])),
                    }
                )
            if tick % 1000 == 999:
                print(f"Stability {tick + 1}/{args.stability_ticks} ticks", flush=True)
        with gpu._device, gpu._stream:
            population_spikes = int(total.get())
            ever_active = active.get()
        report["stability"] = {
            "ticks": args.stability_ticks,
            "simulated_ms": args.stability_ticks,
            "wall_seconds_including_observation": time.perf_counter() - start,
            "total_spikes": population_spikes,
            "ever_spiking_neurons": int(ever_active.sum()),
            "trace": trace,
        }
        report["gpu_pool_used_bytes"] = cp.get_default_memory_pool().used_bytes()
        report["gpu_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
        report["host_peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    print("FULL BRAIN CUDA VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    main()
