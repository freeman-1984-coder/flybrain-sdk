"""Dependency and diagnostic behavior, not simulated GPU conformance."""

import subprocess
import sys

import pytest

from flybrain import BackendUnavailableError, FlyBrain, available_backends
from flybrain.backends import cuda


def test_cpu_import_does_not_import_cupy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from flybrain import FlyBrain; "
            "FlyBrain.load().step(); assert 'cupy' not in sys.modules",
        ],
        check=True,
    )


def test_missing_cupy_has_actionable_error(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("test: no CuPy")

    monkeypatch.setattr(cuda, "import_module", missing)
    assert available_backends() == {"cpu": True, "cuda": False, "wasm": False}
    with pytest.raises(BackendUnavailableError, match="NVIDIA driver"):
        FlyBrain.load(backend="cuda")
    assert FlyBrain.load(backend="cpu").step().tick == 1
