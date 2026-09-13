"""Optional, manually invoked Modal T4 validation. Running this can incur charges.

Only the source package, one test file, validator and small checked-in model are
uploaded. No home directory, credential files, raw datasets or .git are included.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.6", "pytest==8.4.2", "cupy-cuda12x[ctk]==14.2.0")
    .add_local_dir(ROOT / "src/flybrain", "/project/src/flybrain", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "tests/test_cuda_hardware.py", "/project/tests/test_cuda_hardware.py")
    .add_local_file(ROOT / "scripts/validate_cuda.py", "/project/scripts/validate_cuda.py")
    .add_local_file(
        ROOT / "models/male-cns-escape-v1/model.json",
        "/project/models/male-cns-escape-v1/model.json",
    )
    .env({"PYTHONPATH": "/project/src"})
)
app = modal.App("flybrain-cuda-validation", image=image)


@app.function(
    gpu="T4",
    cpu=1,
    memory=4096,
    timeout=600,
    retries=0,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
)
def validate():
    import subprocess
    import sys

    path = Path("/tmp/cuda-validation.json")
    result = subprocess.run(
        [sys.executable, "/project/scripts/validate_cuda.py", "--output", str(path)],
        cwd="/project",
        timeout=480,
    )
    report = json.loads(path.read_text()) if path.exists() else {"status": "failed"}
    report["runner_exit_code"] = result.returncode
    return report


@app.local_entrypoint()
def main(output: str = "cuda-validation.json"):
    report = validate.remote()
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"CUDA report: {output}; status: {report['status']}")
    if report["status"] != "passed" or report["runner_exit_code"] != 0:
        raise RuntimeError("CUDA validation failed; inspect the report and remote logs")
