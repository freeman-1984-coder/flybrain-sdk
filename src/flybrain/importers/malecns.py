"""Convert a frozen MaleCNS selection into an explicitly parameterized model.

The source wiring/counts are anatomical data. The conversion policy is a modeling
assumption, not a fitted physiological model. All source hashes must be specified.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Union

from ..config import LIFConfig, finite_number, positive_int
from ..model import Connectome
from ..registry import _digest


def import_malecns(
    data_dir: Union[str, Path], recipe: Union[str, Path, dict], output: Union[str, Path]
) -> Path:
    """Stream Feather v2 batches and write a self-contained model bundle.

    ``recipe`` freezes neuron IDs, type/side-based ports, per-type signs, incoming
    normalization, LIF parameters, sources and checksums. No network requests occur.
    Install ``flybrain-sdk[datasets]`` to enable this converter.
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise ImportError(
            "MaleCNS conversion requires pip install 'flybrain-sdk[datasets]'"
        ) from exc

    if not isinstance(recipe, dict):
        recipe = json.loads(Path(recipe).read_text(encoding="utf-8"))
    if type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise ValueError("unsupported import recipe schema")
    ids = recipe["neuron_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or len(set(ids)) != len(ids)
        or any(not isinstance(n, str) or not n.isdecimal() for n in ids)
    ):
        raise ValueError("recipe neuron_ids must be unique decimal strings")
    ids = sorted(ids, key=int)
    id_set = set(ids)
    config = LIFConfig(**recipe["config"])
    policy = recipe["weights"]
    gain = finite_number(policy["incoming_gain"], "incoming_gain")
    if gain <= 0 or policy.get("unknown_policy") != "error":
        raise ValueError("positive incoming_gain and explicit unknown_policy='error' required")
    minimum = positive_int(policy["min_synapses"], "min_synapses")
    type_signs = policy["type_signs"]
    if any(type(v) is not int or v not in (-1, 1) for v in type_signs.values()):
        raise ValueError("type_signs must contain integer +1 or -1 values")
    data_dir = Path(data_dir)
    sources = recipe["sources"]
    paths = {}
    for name in ("annotations", "connections", "neurotransmitters"):
        source = sources[name]
        if Path(source["filename"]).name != source["filename"]:
            raise ValueError("source filename must be a basename")
        path = data_dir / source["filename"]
        if not source["sha256"] or _digest(path) != source["sha256"]:
            raise ValueError(f"source checksum mismatch: {name}")
        paths[name] = path

    def require_columns(table, columns):
        missing = set(columns) - set(table.column_names)
        if missing:
            raise ValueError(f"source schema missing columns: {sorted(missing)}")

    annotations = feather.read_table(paths["annotations"])
    require_columns(annotations, ["bodyId", "type", "somaSide", "statusLabel"])
    values = pa.array([int(n) for n in ids], type=pa.int64())
    selected = annotations.filter(pc.is_in(annotations["bodyId"], value_set=values))
    selected = selected.select(["bodyId", "type", "somaSide", "statusLabel"])
    cells = {str(row["bodyId"]): row for row in selected.to_pylist()}
    if set(cells) != id_set or len(selected) != len(ids):
        raise ValueError("selection has missing or duplicate annotation IDs")
    if any(row["type"] not in type_signs for row in cells.values()):
        raise ValueError("selected cell type has no explicit sign policy")
    nt = feather.read_table(paths["neurotransmitters"])
    require_columns(nt, ["body", "consensus_nt", "predicted_nt_confidence"])
    nt = nt.filter(pc.is_in(nt["body"], value_set=values))
    transmitters = {str(row["body"]): row for row in nt.to_pylist()}
    if set(transmitters) != id_set or len(nt) != len(ids):
        raise ValueError("selection has missing or duplicate transmitter records")

    ports = {}
    for kind in ("sensory", "motor"):
        ports[kind] = {}
        for name, selector in recipe["ports"][kind].items():
            matches = [
                n
                for n in ids
                if cells[n]["type"] in selector["types"]
                and ("side" not in selector or cells[n]["somaSide"] == selector["side"])
            ]
            if not matches:
                raise ValueError(f"port {name!r} selects no neurons")
            ports[kind][name] = matches

    # Feather v2 uses the Arrow IPC file format. Batches bound peak memory even
    # when the full source contains many millions of non-neuronal segments.
    counts = defaultdict(int)
    scanned = retained = 0
    with pa.memory_map(str(paths["connections"]), "r") as source:
        reader = ipc.open_file(source)
        missing = {"body_pre", "body_post", "weight"} - set(reader.schema.names)
        if missing:
            raise ValueError(f"connection schema missing columns: {sorted(missing)}")
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            scanned += len(batch)
            mask = pc.and_(
                pc.is_in(batch["body_pre"], value_set=values),
                pc.is_in(batch["body_post"], value_set=values),
            )
            for row in batch.filter(mask).to_pylist():
                count = positive_int(row["weight"], "raw synapse count")
                counts[(str(row["body_pre"]), str(row["body_post"]))] += count
                retained += 1
    counts = {edge: count for edge, count in counts.items() if count >= minimum}
    if not counts:
        raise ValueError("selected graph has no retained connections")
    incoming = defaultdict(int)
    for (_, post), count in counts.items():
        incoming[post] += count
    edges = sorted(counts, key=lambda e: (int(e[0]), int(e[1])))
    normalized = [
        {
            "pre": pre,
            "post": post,
            "weight": type_signs[cells[pre]["type"]] * gain * counts[(pre, post)] / incoming[post],
        }
        for pre, post in edges
    ]
    model = Connectome.from_dict(
        {
            "schema_version": 1,
            "name": recipe["model_id"],
            "neuron_ids": ids,
            "synapses": normalized,
            **ports,
            "provenance": {
                "source": "MaleCNS v1.0 real anatomical subgraph",
                "dataset_version": "1.0-minconf-0.5",
                "license": "CC-BY-4.0",
                "model_status": "experimental; physiological parameters and readout are assumed",
                "recipe_sha256": hashlib.sha256(
                    json.dumps(recipe, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
            },
            "annotations": {
                n: {
                    "cell_type": cells[n]["type"],
                    "side": cells[n]["somaSide"] or "unknown",
                    "status": cells[n]["statusLabel"] or "unknown",
                    "consensus_nt": transmitters[n]["consensus_nt"] or "unknown",
                }
                for n in ids
            },
        }
    )
    report = {
        "neuron_count": len(ids),
        "edge_count": len(edges),
        "raw_synapse_count": sum(counts.values()),
        "source_connection_rows": scanned,
        "within_selection_rows": retained,
        "outside_selection_rows": scanned - retained,
        "type_counts": dict(sorted(Counter(c["type"] for c in cells.values()).items())),
    }
    bundle = {
        "format": "flybrain-model-bundle",
        "schema_version": 1,
        "model": model.to_dict(),
        "model_sha256": model.fingerprint,
        "config": config.to_dict(),
        "recipe": recipe,
        "report": report,
        "neurons": [
            {
                "id": n,
                "type": cells[n]["type"],
                "side": cells[n]["somaSide"],
                "status": cells[n]["statusLabel"],
                "consensus_nt": transmitters[n]["consensus_nt"],
                "nt_confidence": transmitters[n]["predicted_nt_confidence"],
            }
            for n in ids
        ],
        "raw_connections": [
            {"pre": pre, "post": post, "count": counts[(pre, post)]} for pre, post in edges
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output
