"""Developer-facing simulation facade and portable checkpoint format."""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from .backends import SimulationState, create_backend
from .config import LIFConfig, positive_int
from .errors import CheckpointError
from .model import Connectome
from .motor import MotorAction, MotorDecoder
from .sensory import SensoryEncoder, Stimulus

PathLike = Union[str, Path]


class FlyBrain:
    def __init__(
        self, model: Connectome, *, backend: str = "cpu", config: Optional[LIFConfig] = None
    ) -> None:
        self._model = model
        self._config = config if config is not None else LIFConfig()
        self._backend = create_backend(backend, model, self.config)
        self._sensory = SensoryEncoder(model, self.config)
        self._motor = MotorDecoder(model, self.config)

    @property
    def model(self) -> Connectome:
        return self._model

    @property
    def config(self) -> LIFConfig:
        return self._config

    @property
    def backend(self) -> str:
        return self._backend.name

    @property
    def sensory_channels(self) -> tuple:
        return tuple(self.model.sensory)

    @property
    def state(self) -> SimulationState:
        return self._backend.observe()

    @classmethod
    def load(
        cls,
        model: Union[PathLike, Connectome] = "toy",
        *,
        backend: str = "cpu",
        config: Optional[LIFConfig] = None,
    ) -> "FlyBrain":
        """Load the bundled toy or a local versioned JSON model, without network I/O."""
        if isinstance(model, str) and model in ("male-cns-v1.0", "flywire-v783"):
            raise ValueError(
                f"{model!r} is raw connectome data, not a runnable model yet. "
                "Use fetch_model() to download; see docs/real-data.md for conversion."
            )
        connectome = model if isinstance(model, Connectome) else Connectome.load(model)
        return cls(connectome, backend=backend, config=config)

    def stimulate(
        self, channel: Union[str, Stimulus], *, strength: float = 1.0, duration_ms: float = 20.0
    ) -> "FlyBrain":
        """Queue an additive current pulse, beginning on the next simulation tick."""
        if isinstance(channel, Stimulus):
            if strength != 1.0 or duration_ms != 20.0:
                raise ValueError("pass either a Stimulus or channel keyword parameters")
            stimulus = channel
        else:
            stimulus = Stimulus(channel, strength, duration_ms)
        self._sensory.add(stimulus)
        return self

    def clear_stimuli(self) -> None:
        """Cancel pending inputs; existing neural activity continues naturally."""
        self._sensory.clear()

    def step(self, steps: int = 1) -> SimulationState:
        """Advance a positive integer number of fixed ticks and return the final state."""
        positive_int(steps, "steps")
        for _ in range(steps):
            self._backend.step(self._sensory.current())
        return self.state

    def action(self) -> MotorAction:
        """Read motor intensities without advancing time or consuming state."""
        return self._motor.decode(self.state.rates_hz)

    def save(self, path: PathLike) -> Path:
        """Atomically write a self-contained JSON checkpoint (never pickle)."""
        path = Path(path)
        payload = {
            "schema_version": 1,
            "dynamics_revision": self._backend.dynamics_revision,
            "backend": self.backend,
            "model": self.model.to_dict(),
            "model_sha256": self.model.fingerprint,
            "config": self.config.to_dict(),
            "state": self._backend.snapshot(),
            "pending_stimuli": self._sensory.snapshot(),
        }
        raw = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def restore(cls, path: PathLike, *, backend: Optional[str] = None) -> "FlyBrain":
        """Construct a new brain, including active stimuli and all dynamic state."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if type(data["schema_version"]) is not int or data["schema_version"] != 1:
                raise ValueError("unsupported checkpoint schema_version")
            model = Connectome.from_dict(data["model"])
            if model.fingerprint != data["model_sha256"]:
                raise ValueError("checkpoint model fingerprint mismatch")
            brain = cls(
                model,
                backend=backend if backend is not None else data["backend"],
                config=LIFConfig(**data["config"]),
            )
            if data["dynamics_revision"] != brain._backend.dynamics_revision:
                raise ValueError("incompatible dynamics_revision")
            brain._backend.restore(data["state"])
            brain._sensory.restore(data["pending_stimuli"])
            return brain
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
            raise CheckpointError(f"Invalid checkpoint: {exc}") from exc
