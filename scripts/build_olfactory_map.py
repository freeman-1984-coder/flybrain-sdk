"""Resolve pinned FlyWire annotations against the complete official v783 ID set.

No download or inferred hemisphere assignment. The output records exact IDs,
source checksums and mapping assumptions for downstream CUDA experiments.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ANNOTATION_COMMIT = "8587524c1748ce5ef2080822a2fc890fc03bf597"
ANNOTATION_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    + ANNOTATION_COMMIT
    + "/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
ANNOTATION_SHA256 = "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be"
ROOTS_SHA256 = "7c7b7e818e9232e5ab64793d52ba20574dbe9da3a6509fbc74193b0f259a01be"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_populations(rows, neuron_ids):
    """Keep IDs as strings throughout; reject uncertain/missing bilateral inputs."""
    known = set(neuron_ids)
    groups = {f"ORN_DM1_{side}": [] for side in ("left", "right")}
    groups.update({f"DNa02_{side}": [] for side in ("left", "right")})
    groups.update({"ALPN": [], "MBON": [], "descending": []})
    seen = set()
    for row in rows:
        neuron_id = row["root_id"]
        if not neuron_id.isascii() or not neuron_id.isdecimal():
            raise ValueError("root_id must be an exact decimal string")
        if neuron_id in seen:
            raise ValueError("duplicate annotation root_id")
        seen.add(neuron_id)
        cell_type, side = row["cell_type"], row["side"]
        targets = []
        if cell_type == "ORN_DM1":
            if row["super_class"] != "sensory" or row["cell_class"] != "olfactory":
                raise ValueError("ORN_DM1 has conflicting sensory annotation")
            if side not in ("left", "right"):
                raise ValueError("ORN_DM1 side is missing/ambiguous; never split arbitrarily")
            targets.append(f"ORN_DM1_{side}")
        if cell_type == "DNa02":
            if row["super_class"] != "descending" or side not in ("left", "right"):
                raise ValueError("DNa02 has conflicting descending/side annotation")
            targets.append(f"DNa02_{side}")
        for column, value in (
            ("cell_class", "ALPN"),
            ("cell_class", "MBON"),
            ("super_class", "descending"),
        ):
            if row[column] == value:
                targets.append(value)
        if targets and neuron_id not in known:
            raise ValueError(f"selected annotation references unknown neuron {neuron_id}")
        for target in targets:
            groups[target].append(neuron_id)
    if any(not values for values in groups.values()):
        raise ValueError("one or more required sensory/readout populations are empty")
    return {name: sorted(values, key=int) for name, values in groups.items()}


def build(annotations, roots):
    if sha256(annotations) != ANNOTATION_SHA256 or sha256(roots) != ROOTS_SHA256:
        raise ValueError("pinned source checksum mismatch")
    ids = np.load(roots, allow_pickle=False)
    if ids.dtype.kind not in "iu" or len(ids) != 139255 or len(np.unique(ids)) != 139255:
        raise ValueError("expected all 139255 official integer neuron IDs")
    with annotations.open(newline="", encoding="utf-8") as f:
        groups = resolve_populations(csv.DictReader(f, delimiter="\t"), map(str, ids))
    if len(groups["ORN_DM1_left"]) != 35 or len(groups["ORN_DM1_right"]) != 33:
        raise ValueError("pinned DM1 population counts changed")
    return {
        "schema": "flybrain-olfactory-map-v1",
        "model": "flywire-full-v783",
        "neuron_count": 139255,
        "annotation_source": {
            "url": ANNOTATION_URL,
            "commit": ANNOTATION_COMMIT,
            "sha256": ANNOTATION_SHA256,
            "license": "not specified in the pinned repository; do not infer SDK MIT coverage",
            "citation": "Schlegel et al. 2024; Matsliah et al. 2024; Berg et al. 2025/2026",
        },
        "root_ids_sha256": ROOTS_SHA256,
        "stimulus": {
            "label": "ethyl-acetate / Or42b channel approximation",
            "receptor": "Or42b",
            "glomerulus": "DM1",
            "evidence": "https://www.nature.com/articles/s41598-017-13015-w",
            "scope": (
                "One responsive receptor channel; "
                "not the complete ethyl-acetate or banana response"
            ),
            "current_units": "dimensionless; not measured current or calibrated odor concentration",
            "hemispheres": "source annotation side; no invented left/right assignments",
        },
        "groups": groups,
        "counts": {name: len(values) for name, values in groups.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.annotations, args.roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
