import hashlib
import json

import pytest

pa = pytest.importorskip("pyarrow")
feather = pytest.importorskip("pyarrow.feather")

from flybrain import FlyBrain, LIFConfig  # noqa: E402
from flybrain.importers import import_malecns  # noqa: E402


def fixture(tmp_path):
    tables = {
        "annotations": pa.table(
            {
                "bodyId": [1, 2, 3, 4],
                "type": ["LC4", "LPLC2", "DNp01", "X"],
                "somaSide": ["L"] * 4,
                "statusLabel": ["fixture"] * 4,
            }
        ),
        "connections": pa.table(
            {"body_pre": [1, 1, 2, 4], "body_post": [3, 3, 3, 3], "weight": [2, 3, 5, 100]}
        ),
        "neurotransmitters": pa.table(
            {
                "body": [1, 2, 3],
                "consensus_nt": ["acetylcholine"] * 3,
                "predicted_nt_confidence": [0.9] * 3,
            }
        ),
    }
    sources = {}
    for name, table in tables.items():
        path = tmp_path / f"{name}.feather"
        feather.write_feather(table, path, chunksize=2)
        sources[name] = {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    recipe = {
        "schema_version": 1,
        "model_id": "fixture",
        "neuron_ids": ["1", "2", "3"],
        "sources": sources,
        "config": LIFConfig().to_dict(),
        "weights": {
            "incoming_gain": 14,
            "min_synapses": 1,
            "type_signs": {"LC4": 1, "LPLC2": 1, "DNp01": 1},
            "unknown_policy": "error",
        },
        "ports": {
            "sensory": {"looming_left": {"types": ["LC4", "LPLC2"], "side": "L"}},
            "motor": {"jump": {"types": ["DNp01"]}},
        },
    }
    return recipe


def test_converter_preserves_counts_aggregates_and_excludes(tmp_path):
    recipe = fixture(tmp_path)
    path = import_malecns(tmp_path, recipe, tmp_path / "model.json")
    bundle = json.loads(path.read_text())
    assert bundle["report"]["raw_synapse_count"] == 10
    assert bundle["report"]["source_connection_rows"] == 4
    assert bundle["report"]["outside_selection_rows"] == 1
    assert [e["count"] for e in bundle["raw_connections"]] == [5, 5]
    assert [e["weight"] for e in bundle["model"]["synapses"]] == [7, 7]
    brain = FlyBrain.load(path)
    brain.stimulate("looming_left", duration_ms=100)
    brain.step(100)
    assert brain.action().jump > 0
    second = import_malecns(tmp_path, recipe, tmp_path / "again.json")
    assert path.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("problem", ["checksum", "missing_id", "missing_type", "empty_port"])
def test_converter_rejects_bad_recipe_before_output(tmp_path, problem):
    recipe = fixture(tmp_path)
    if problem == "checksum":
        recipe["sources"]["connections"]["sha256"] = "0" * 64
    elif problem == "missing_id":
        recipe["neuron_ids"].append("999")
    elif problem == "missing_type":
        del recipe["weights"]["type_signs"]["LC4"]
    else:
        recipe["ports"]["motor"]["jump"]["types"] = ["missing"]
    target = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        import_malecns(tmp_path, recipe, target)
    assert not target.exists()


def test_explicit_inhibitory_policy_is_preserved(tmp_path):
    recipe = fixture(tmp_path)
    recipe["weights"]["type_signs"]["LPLC2"] = -1
    path = import_malecns(tmp_path, recipe, tmp_path / "signed.json")
    assert [e["weight"] for e in json.loads(path.read_text())["model"]["synapses"]] == [7, -7]
