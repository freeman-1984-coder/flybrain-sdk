"""Replay and verify a recording downloaded from the browser circuit lab."""

import argparse
import json
from pathlib import Path

from flybrain.experiment import replay_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--download", action="store_true", help="Allow catalog model download")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.recording.stat().st_size > 20_000_000:
        parser.error("recording exceeds 20 MB")
    try:
        brain = replay_experiment(
            json.loads(args.recording.read_text(encoding="utf-8")),
            download=args.download,
            cache_dir=args.cache_dir,
            max_steps=args.max_steps,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Replay failed: {error}\n")
    print(
        json.dumps(
            {
                "verified": True,
                "tick": brain.state.tick,
                "time_ms": brain.state.time_ms,
                "action": brain.action().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
