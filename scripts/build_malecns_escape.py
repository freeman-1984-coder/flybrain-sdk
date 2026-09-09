"""Reproduce the distributed escape model from verified local source files."""

import argparse
from pathlib import Path

from flybrain.importers import import_malecns

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "data_dir", type=Path, help="directory containing the three MaleCNS source files"
)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
folder = root / "models" / "male-cns-escape-v1"
result = import_malecns(args.data_dir, folder / "recipe.json", args.output or folder / "model.json")
print(result)
