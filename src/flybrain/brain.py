"""Developer-facing simulation facade and portable checkpoint format."""

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from .backends import SimulationState, create_backend
from .config import LIFConfig, positive_int
from .drive import CurrentDrive, Interventions, duration_ticks
from .errors import CheckpointError
from .model import Connectome
from .motor import ChannelAction, MotorAction, MotorDecoder
from .neurons import NeuronIndex, Observation, Progress
from .registry import list_models, resolve_model
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
        self.neurons = NeuronIndex(model)
        self.drive = CurrentDrive(self.neurons, self.config)
        self.intervene = Interventions(self.neurons, self._backend)

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
        download: bool = False,
        cache_dir: Optional[PathLike] = None,
    ) -> "FlyBrain":
        """Load toy, a local model/bundle, or a catalog model with opt-in downloads."""
        if isinstance(model, Connectome):
            return cls(model, backend=backend, config=config)
        if model == "toy":
            return cls(Connectome.load(), backend=backend, config=config)
        if isinstance(model, str) and model in {e["id"] for e in list_models()}:
            model = resolve_model(model, download=download, cache_dir=cache_dir)
        data = json.loads(Path(model).read_text(encoding="utf-8"))
        if data.get("format") == "flybrain-model-bundle":
            if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
                raise ValueError("unsupported model bundle schema")
            connectome = Connectome.from_dict(data["model"])
            if connectome.fingerprint != data["model_sha256"]:
                raise ValueError("model bundle fingerprint mismatch")
            if config is None:
                config = LIFConfig(**data["config"])
        else:
            connectome = Connectome.from_dict(data)
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
        self._advance_ticks(steps)
        return self.state

    def _advance_ticks(self, steps: int) -> None:
        for _ in range(steps):
            self._backend.step(self._sensory.peek() + self.drive.peek())
            self._sensory.consume()
            self.drive.consume()

    def advance(self, *, duration_ms: float) -> Progress:
        """Advance an exact multiple of dt without exporting all neural arrays."""
        self._advance_ticks(duration_ticks(duration_ms, self.config.dt_ms))
        return Progress(self._backend.tick, self._backend.tick * self.config.dt_ms)

    def observe(
        self,
        ids: Optional[Sequence[str]] = None,
        *,
        fields: Sequence[str] = ("voltage", "spikes", "rates_hz"),
    ) -> Observation:
        """Read selected cells only. spikes means final-tick events, not window counts."""
        ids = (
            self.model.neuron_ids
            if ids is None
            else tuple(ids)
            if not isinstance(ids, str)
            else ids
        )
        indices = self.neurons.resolve(ids)
        if isinstance(fields, str):
            raise ValueError("fields must be a sequence of field names")
        fields = tuple(fields)
        if (
            not fields
            or len(set(fields)) != len(fields)
            or set(fields) - {"voltage", "spikes", "rates_hz"}
        ):
            raise ValueError("fields must be unique names from voltage, spikes, rates_hz")
        return Observation(
            self._backend.tick,
            self._backend.tick * self.config.dt_ms,
            tuple(ids),
            self._backend.observe_selected(indices, fields),
        )

    def bind_readout(
        self, channels: Mapping[str, Sequence[str]], *, scale_hz: Optional[float] = None
    ) -> "FlyBrain":
        """Bind arbitrary channel names to mean-rate readouts; no graph changes."""
        self._motor.bind(
            channels, scale_hz=self.config.action_rate_hz if scale_hz is None else scale_hz
        )
        return self

    def action(self) -> Union[MotorAction, ChannelAction]:
        """Read motor intensities without advancing time or consuming state."""
        indices = self._motor.indices
        rates = self._backend.observe_selected(indices, ("rates_hz",))["rates_hz"]
        return self._motor.decode(dict(zip(indices, rates)))

    def save(self, path: PathLike) -> Path:
        """Atomically write a self-contained JSON checkpoint (never pickle)."""
        path = Path(path)
        payload = {
            "schema_version": 2,
            "dynamics_revision": self._backend.dynamics_revision,
            "backend": self.backend,
            "model": self.model.to_dict(),
            "model_sha256": self.model.fingerprint,
            "config": self.config.to_dict(),
            "state": self._backend.snapshot(),
            "pending_stimuli": self._sensory.snapshot(),
            "pending_currents": self.drive.snapshot(),
            "readout": self._motor.snapshot(),
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
            if type(data["schema_version"]) is not int or data["schema_version"] not in (1, 2):
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
            if data["schema_version"] == 2:
                if "silenced" not in data["state"]:
                    raise ValueError("checkpoint is missing intervention state")
                brain.drive.restore(data["pending_currents"])
                brain._motor.restore(data["readout"])
            return brain
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
            raise CheckpointError(f"Invalid checkpoint: {exc}") from exc
