import importlib.util
from pathlib import Path

import numpy as np
import pytest

from flybrain import Connectome, LIFConfig, Synapse
from flybrain.backends.cpu import CPUBackend
from flybrain.olfaction import OlfactoryDrive, sample_odor

spec = importlib.util.spec_from_file_location(
    "olfactory_map", Path(__file__).parents[1] / "scripts/build_olfactory_map.py"
)
mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapping)


@pytest.fixture
def model():
    return Connectome(
        "fixture",
        ("101", "102", "103", "104"),
        (Synapse(0, 2, 30.0), Synapse(1, 3, 30.0)),
        {},
        {},
        {},
    )


def test_local_antenna_stimulus_only_drives_annotated_side(model):
    odor = OlfactoryDrive(model, left_ids=["101"], right_ids=["102"])
    left = odor.current(left=1, right=0)
    np.testing.assert_array_equal(np.flatnonzero(left), [0])
    brain = CPUBackend(model, LIFConfig())
    for _ in range(100):
        brain.step(left)
    # A fixture tests transport/isolation, not a biological odor pathway.
    state = brain.observe_selected((0, 1, 2, 3), ("rates_hz",))["rates_hz"]
    assert state[0] > 0 and state[2] > 0
    assert state[1] == 0 and state[3] == 0
    brain.set_silenced((0,), True)
    for _ in range(1000):
        brain.step(left)
    assert brain.observe_selected((2,), ("rates_hz",))["rates_hz"][0] < 1e-5


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True])
def test_invalid_concentration_rejected(model, bad):
    odor = OlfactoryDrive(model, left_ids=["101"], right_ids=["102"])
    with pytest.raises(ValueError):
        odor.current(left=bad, right=0)


def test_response_bounded_and_monotone(model):
    odor = OlfactoryDrive(model, left_ids=["101"], right_ids=["102"])
    values = [odor.amplitude(v) for v in [0, 1e-6, 0.1, 1, 10, 1e308]]
    assert values == sorted(values)
    assert values[0] == 0 and values[-1] <= odor.max_current
    with pytest.raises(ValueError, match="overlap"):
        OlfactoryDrive(model, left_ids=["101"], right_ids=["101"])
    with pytest.raises(ValueError, match="known"):
        OlfactoryDrive(model, left_ids=["unknown"], right_ids=["102"])


def test_world_field_uses_local_sample_and_is_mirror_symmetric():
    source = (0, 2, 0)
    assert sample_odor(source, source) == 1
    assert sample_odor((-1, 0, 0), source) == sample_odor((1, 0, 0), source)
    assert sample_odor((0, 1, 0), source) > sample_odor((0, -1, 0), source)
    assert sample_odor((0, 0, 0), source, strength=0) == 0
    with pytest.raises(ValueError):
        sample_odor((float("nan"), 0, 0), source)


def rows():
    selections = [
        ("ORN_DM1", "left", "olfactory", "sensory"),
        ("ORN_DM1", "right", "olfactory", "sensory"),
        ("DNa02", "left", "", "descending"),
        ("DNa02", "right", "", "descending"),
        ("PN", "left", "ALPN", "central"),
        ("MBON", "right", "MBON", "central"),
    ]
    return [
        dict(root_id=str(720575940600000000 + i), cell_type=t, side=s, cell_class=c, super_class=sc)
        for i, (t, s, c, sc) in enumerate(selections)
    ]


def test_annotation_ids_preserved_and_unknown_side_never_guessed():
    data = rows()
    ids = [r["root_id"] for r in data]
    groups = mapping.resolve_populations(data, ids)
    assert groups["ORN_DM1_right"] == ["720575940600000001"]
    data[1]["side"] = ""
    with pytest.raises(ValueError, match="ambiguous"):
        mapping.resolve_populations(data, ids)


def test_incompatible_annotation_graph_fails_closed():
    data = rows()
    with pytest.raises(ValueError, match="unknown neuron"):
        mapping.resolve_populations(data, [r["root_id"] for r in data[1:]])
    with pytest.raises(ValueError, match="duplicate"):
        mapping.resolve_populations(data + [data[0]], [r["root_id"] for r in data])


def test_source_drift_rejected_before_loading_ids(tmp_path):
    changed = tmp_path / "annotations.tsv"
    roots = tmp_path / "roots.npy"
    changed.write_text("changed")
    roots.write_text("not a numpy file")
    with pytest.raises(ValueError, match="checksum"):
        mapping.build(changed, roots)
