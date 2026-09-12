"""Regression for official uint64 neuron-list / int64 Feather endpoint IDs."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "fullbrain_runner", Path(__file__).parents[1] / "scripts/validate_fullbrain.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_mixed_integer_dtypes_preserve_neighboring_large_ids():
    base = 720575940600000000
    ids = np.array([base, base + 1, base + 2, base + 100], dtype=np.uint64)
    endpoints = np.array([base + 2, base, base + 100, base + 1], dtype=np.int64)
    assert runner.map_source_ids(ids, endpoints).tolist() == [2, 0, 3, 1]


@pytest.mark.parametrize("unknown", [720575940600000003, 720575940600000101])
def test_unknown_large_id_is_not_rounded_to_an_existing_id(unknown):
    ids = np.array([720575940600000002, 720575940600000100], dtype=np.uint64)
    with pytest.raises(ValueError, match="unknown neuron"):
        runner.map_source_ids(ids, np.array([unknown], dtype=np.int64))


def test_float_ids_are_rejected():
    with pytest.raises(ValueError, match="integer storage"):
        runner.map_source_ids(np.array([1], dtype=np.uint64), np.array([1.0]))
