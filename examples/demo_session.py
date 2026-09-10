"""Run, record, restore and replay a game or tone demo without CUDA."""

import argparse
import json
from pathlib import Path

from flybrain.demos import DodgeArena, PulseScore, make_demo, render_wav
from flybrain.session import replay, write_json
from flybrain.viewer import write_replay_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=["dodge", "tones"])
    parser.add_argument("--model", default="toy")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000:
        parser.error("frames must be in 1..10000")
    args.output.mkdir(parents=True, exist_ok=True)
    session = make_demo(args.demo, model=args.model, download=args.download)
    recording = session.record(args.frames)
    write_json(args.output / "recording.json", recording)
    session.save(args.output / "session.json")
    write_replay_html(recording, args.output / "replay.html")
    factory = DodgeArena.from_snapshot if args.demo == "dodge" else PulseScore.from_snapshot
    restored = replay(recording, environment_factory=factory)
    if args.demo == "tones":
        render_wav(recording, args.output / "tones.wav")
    print(
        json.dumps(
            {
                "verified_replay": True,
                "frames": restored.frame_index,
                "brain_tick": restored.brain.progress.tick,
                "environment": restored.environment.snapshot(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
