"""Model catalog, editable projects and offline demos: python -m flybrain."""

import argparse
import json
from pathlib import Path

from .projects import TEMPLATES, create_project, doctor, run_preset
from .registry import fetch_model, list_models, model_info


def main(argv=None) -> None:
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
    commands.add_parser("doctor", help="Check this installation without network access")
    demos = commands.add_parser("demos", help="List or run bundled offline demos")
    demo_actions = demos.add_subparsers(dest="demo_action", required=True)
    demo_actions.add_parser("list")
    run = demo_actions.add_parser("run")
    run.add_argument("template", choices=TEMPLATES)
    run.add_argument("--frames", type=int, default=300)
    run.add_argument("--output", type=Path, required=True)
    init = commands.add_parser("init", help="Generate an editable source project")
    init.add_argument("destination", type=Path)
    init.add_argument("--template", choices=TEMPLATES, default="dodge")
    for command in (run, init):
        command.add_argument("--model", default="toy")
        command.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(), indent=2))
            return
        if args.command == "init":
            path = create_project(
                args.destination, args.template, model=args.model, download=args.download
            )
            print(f"Created {path}\nEdit recipe.json, then run app.py in that directory.")
            return
        if args.command == "demos":
            if args.demo_action == "list":
                for name, description in TEMPLATES.items():
                    print(f"{name:10} {description} (CPU; no training)")
            else:
                print(
                    json.dumps(
                        run_preset(
                            args.template,
                            args.output,
                            args.frames,
                            model=args.model,
                            download=args.download,
                        ),
                        indent=2,
                    )
                )
            return
    except (ValueError, OSError) as error:
        parser.error(str(error))
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
