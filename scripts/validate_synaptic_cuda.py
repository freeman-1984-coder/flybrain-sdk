"""Fail-closed actual-GPU gate for the experimental synaptic mV engine."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    from flybrain.backends.cuda import _load_cupy

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "status": "failed",
        "python": platform.python_version(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *sorted((root / "src/flybrain/experimental").glob("*.py")),
                root / "tests/test_synaptic_cuda_hardware.py",
                Path(__file__),
            ]
        },
    }
    code = 0
    try:
        cp = _load_cupy()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        report["device"] = {
            "name": props["name"].decode(),
            "cupy": cp.__version__,
            "memory_bytes": props["totalGlobalMem"],
            "driver": cp.cuda.runtime.driverGetVersion(),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
        }
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "pytest.xml"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_synaptic_cuda_hardware.py",
                    "-q",
                    f"--junitxml={xml}",
                ],
                cwd=root,
                env={**os.environ, "FLYBRAIN_REQUIRE_CUDA": "1"},
                check=True,
                timeout=300,
            )
            suites = ET.parse(xml).getroot().iter("testsuite")
            counts = {k: 0 for k in ("tests", "skipped", "errors", "failures")}
            for suite in suites:
                for k in counts:
                    counts[k] += int(suite.get(k, 0))
            report["cases"] = counts
            if counts["tests"] != 5 or any(counts[k] for k in ("skipped", "errors", "failures")):
                raise RuntimeError("all five actual GPU cases must execute and pass")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        code = 1
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
