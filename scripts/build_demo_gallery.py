"""Build inspectable recorded gallery examples from the checked-in real model."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from flybrain import __version__
from flybrain.demos import DodgeArena, PulseScore, make_demo, render_wav
from flybrain.session import replay
from flybrain.viewer import write_replay_html

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "site/demos"
folder.mkdir(exist_ok=True)
model = ROOT / "models/male-cns-escape-v1/model.json"
report = {
    "format": "flybrain-demo-gallery",
    "schema_version": 1,
    "sdk_version": __version__,
    "source_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src/flybrain").rglob("*.py"))
    },
    "python": platform.python_version(),
    "numpy": np.__version__,
    "model_asset_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    "frames": 300,
    "period_ms": 20,
    "runs": [],
    "files": {},
}
for demo, silence in [("dodge", False), ("dodge", True), ("tones", False)]:
    session = make_demo(demo, model=model)
    if silence:
        session.brain.intervene.silence(["10001", "10010"])
    recording = session.record(300)
    factory = DodgeArena.from_snapshot if demo == "dodge" else PulseScore.from_snapshot
    replay(recording, environment_factory=factory)
    name = "dodge-silenced" if silence else demo + "-replay"
    html = write_replay_html(recording, folder / (name + ".html"))
    report["files"][html.name] = hashlib.sha256(html.read_bytes()).hexdigest()
    if demo == "tones":
        wav = render_wav(recording, folder / "tones.wav")
        report["files"][wav.name] = hashlib.sha256(wav.read_bytes()).hexdigest()
    report["runs"].append(
        {
            "name": name,
            "verified_replay": True,
            "initial_silence": silence,
            "final_environment": session.environment.snapshot(),
        }
    )
(folder / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report["runs"], indent=2))
