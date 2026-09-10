"""Actual-device conformance gate and small-model latency report; never an emulator."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(report):
    import numpy as np

    from flybrain import FlyBrain
    from flybrain.backends.cuda import _load_cupy

    cp = _load_cupy()
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = device["name"]
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": name.decode() if isinstance(name, bytes) else name,
        "compute_capability": [device["major"], device["minor"]],
        "device_memory_bytes": device["totalGlobalMem"],
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
    }
    # A positive device count is not a conformance pass. Require every hardware
    # test to execute successfully; a skipped test fails this gate.
    with tempfile.TemporaryDirectory() as folder:
        results = Path(folder) / "pytest.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_cuda_hardware.py",
                "-q",
                f"--junitxml={results}",
            ],
            cwd=ROOT,
            env={**os.environ, "FLYBRAIN_REQUIRE_CUDA": "1"},
            check=True,
            timeout=300,
        )
        suites = list(ET.parse(results).getroot().iter("testsuite"))
        counts = {
            k: sum(int(s.get(k, 0)) for s in suites)
            for k in ("tests", "failures", "errors", "skipped")
        }
        if counts["tests"] < 10 or any(counts[k] for k in ("failures", "errors", "skipped")):
            raise RuntimeError(f"Incomplete CUDA validation: {counts}")
        report["conformance"] = counts
    report["benchmarks"] = []
    for model in ("toy", ROOT / "models/male-cns-escape-v1/model.json"):
        loaded = FlyBrain.load(model)
        measurements = {}
        for backend in ("cpu", "cuda"):
            cold_start = time.perf_counter()
            brain = FlyBrain.load(model, backend=backend)
            measurements[backend + "_load_seconds"] = time.perf_counter() - cold_start
            brain.stimulate(brain.sensory_channels[0], duration_ms=4000)
            brain.advance(duration_ms=100)
            times = []
            for _ in range(3):
                start = time.perf_counter()
                brain.advance(duration_ms=1000)
                brain.action()
                if backend == "cuda":
                    cp.cuda.get_current_stream().synchronize()
                times.append(time.perf_counter() - start)
            measurements[backend] = {"seconds": times, "median_seconds": statistics.median(times)}
        report["benchmarks"].append(
            {
                "model": loaded.model.name,
                "fingerprint": loaded.model.fingerprint,
                "neurons": len(loaded.model.neuron_ids),
                "edges": len(loaded.model.synapses),
                "config": loaded.config.to_dict(),
                "simulated_ms_per_repeat": 1000,
                "warmup_ms": 100,
                "timings": measurements,
            }
        )
    report["limitations"] = [
        "Correctness-first CUDA: host current upload and scalar synchronization each tick.",
        "These timings cover only toy and 313-cell circuit, not whole-brain scaling.",
        "Loading/compilation reported separately; execution includes inputs and action readout.",
        "Conformance applies to tested cases and reported device/software, not all CUDA hardware.",
    ]
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "format": "flybrain-cuda-validation",
        "schema_version": 1,
        "status": "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                list((ROOT / "src/flybrain").rglob("*.py"))
                + [ROOT / "tests/test_cuda_hardware.py", Path(__file__).resolve()]
            )
        },
    }
    code = 0
    try:
        validate(report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        code = 1
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output), "error": report.get("error")},
            indent=2,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
