"""Named backend slots; unavailable backends never silently fall back."""

from ..config import LIFConfig
from ..errors import BackendUnavailableError
from ..model import Connectome
from .base import Backend, SimulationState
from .cpu import CPUBackend

__all__ = ["Backend", "CPUBackend", "SimulationState", "available_backends", "create_backend"]


def available_backends() -> dict:
    return {"cpu": True, "wasm": False, "cuda": False}


def create_backend(name: str, model: Connectome, config: LIFConfig) -> Backend:
    if name == "cpu":
        return CPUBackend(model, config)
    if name in ("wasm", "cuda"):
        raise BackendUnavailableError(
            f"Backend {name!r} is a reserved interface, not implemented in this release. "
            "Use backend='cpu'. No CUDA required."
        )
    raise ValueError(f"unknown backend {name!r}; expected cpu, wasm, or cuda")
