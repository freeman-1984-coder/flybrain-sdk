"""Run python -m flybrain models {list,info,download}."""

import argparse
import json

from .registry import fetch_model, list_models, model_info


def main() -> None:
    parser = argparse.ArgumentParser(description="flybrain-sdk: No CUDA required")
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models")
    actions = models.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    info = actions.add_parser("info")
    info.add_argument("model")
    download = actions.add_parser("download")
    download.add_argument("model")
    download.add_argument("--asset", action="append", dest="assets")
    download.add_argument("--cache-dir")
    download.add_argument("--max-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    if args.action == "list":
        for entry in list_models():
            size = sum(a["size_bytes"] for a in entry["assets"].values())
            print(f"{entry['id']:20} {entry['status']:10} {size:>12,} bytes  {entry['title']}")
    elif args.action == "info":
        print(json.dumps(model_info(args.model), indent=2))
    else:
        paths = fetch_model(
            args.model, assets=args.assets, cache_dir=args.cache_dir, max_bytes=args.max_bytes
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        if model_info(args.model)["status"] == "raw-data":
            print("Downloaded raw data. Conversion/calibration is required before simulation.")


if __name__ == "__main__":
    main()
