"""A small, versioned interchange format; neuron identifiers stay strings."""

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple, Union

from .config import finite_number


@dataclass(frozen=True)
class Synapse:
    pre: int
    post: int
    weight: float


@dataclass(frozen=True)
class Connectome:
    name: str
    neuron_ids: Tuple[str, ...]
    synapses: Tuple[Synapse, ...]
    sensory: Mapping[str, Tuple[int, ...]]
    motor: Mapping[str, Tuple[int, ...]]
    provenance: Mapping[str, str]
    annotations: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model name must be a nonempty string")
        ids = tuple(self.neuron_ids)
        if not ids or any(not isinstance(n, str) or not n for n in ids):
            raise ValueError("neuron_ids must contain nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("neuron_ids must be unique")
        object.__setattr__(self, "neuron_ids", ids)
        edges = []
        for edge in self.synapses:
            for index in (edge.pre, edge.post):
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(ids)
                ):
                    raise ValueError("synapse index out of bounds")
            edges.append(Synapse(edge.pre, edge.post, finite_number(edge.weight, "weight")))
        object.__setattr__(self, "synapses", tuple(edges))
        for kind in ("sensory", "motor"):
            ports = {}
            for name, indices in getattr(self, kind).items():
                indices = tuple(indices)
                if not isinstance(name, str) or not name or not indices:
                    raise ValueError(f"{kind} ports need a name and at least one neuron")
                if any(
                    isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(ids)
                    for i in indices
                ):
                    raise ValueError(f"invalid neuron index in {kind} port {name}")
                if len(set(indices)) != len(indices):
                    raise ValueError(f"duplicate neuron in {kind} port {name}")
                ports[name] = indices
            object.__setattr__(self, kind, MappingProxyType(ports))
        if any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in self.provenance.items()
        ):
            raise ValueError("provenance must map strings to strings")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        annotations = {}
        known_ids = set(ids)
        for neuron_id, record in self.annotations.items():
            if neuron_id not in known_ids or not isinstance(record, Mapping):
                raise ValueError("annotations must map known neuron IDs to attribute mappings")
            if any(not isinstance(k, str) or not isinstance(v, str) for k, v in record.items()):
                raise ValueError("neuron annotations must map strings to strings")
            annotations[neuron_id] = MappingProxyType(dict(record))
        object.__setattr__(self, "annotations", MappingProxyType(annotations))

    @classmethod
    def from_dict(cls, data: dict) -> "Connectome":
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported connectome schema_version (expected 1)")
        ids = tuple(data["neuron_ids"])
        if any(not isinstance(n, str) for n in ids):
            raise ValueError("neuron IDs must be strings, including large source IDs")
        index = {neuron_id: i for i, neuron_id in enumerate(ids)}
        try:
            edges = tuple(
                Synapse(index[e["pre"]], index[e["post"]], e["weight"]) for e in data["synapses"]
            )
            sensory = {k: tuple(index[n] for n in v) for k, v in data["sensory"].items()}
            motor = {k: tuple(index[n] for n in v) for k, v in data["motor"].items()}
        except KeyError as exc:
            raise ValueError(f"missing model field or unknown neuron ID: {exc}") from exc
        return cls(
            data["name"],
            ids,
            edges,
            sensory,
            motor,
            data.get("provenance", {}),
            data.get("annotations", {}),
        )

    @classmethod
    def load(cls, source: Union[str, Path] = "toy") -> "Connectome":
        if source == "toy":
            raw = files("flybrain.data").joinpath("toy.json").read_text(encoding="utf-8")
        else:
            raw = Path(source).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(raw))

    def to_dict(self) -> dict:
        data = {
            "schema_version": 1,
            "name": self.name,
            "neuron_ids": list(self.neuron_ids),
            "synapses": [
                {"pre": self.neuron_ids[e.pre], "post": self.neuron_ids[e.post], "weight": e.weight}
                for e in self.synapses
            ],
            "sensory": {k: [self.neuron_ids[i] for i in v] for k, v in self.sensory.items()},
            "motor": {k: [self.neuron_ids[i] for i in v] for k, v in self.motor.items()},
            "provenance": dict(self.provenance),
        }
        # Preserve schema-1 fingerprints for existing models without annotations.
        if self.annotations:
            data["annotations"] = {n: dict(a) for n, a in self.annotations.items()}
        return data

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
