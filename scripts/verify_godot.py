"""Run the actual Godot client against Python, including whole-session restore.

Usage: python scripts/verify_godot.py /path/to/godot
No network download is needed: toy and the repository's real subgraph are tested.
"""

import argparse
import importlib.util
import json
import math
import subprocess
import tempfile
import threading
from pathlib import Path

from flybrain.demos import make_demo

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("godot_bridge", ROOT / "examples/godot/bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def equal(actual, expected):
    if isinstance(expected, bool):
        assert actual is expected
    elif isinstance(expected, (float, int)):
        assert not isinstance(actual, bool)
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9), (actual, expected)
        if isinstance(expected, int):
            assert actual == expected
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal(a, b)
    else:
        assert actual == expected


def verify(binary, model, directory):
    with bridge.make_server(0, model=model) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            first, second = directory / "first.json", directory / "resumed.json"
            command = [
                binary,
                "--headless",
                "--path",
                str(ROOT / "examples/godot"),
                "--",
                f"--endpoint=http://127.0.0.1:{server.server_port}",
            ]
            for frames, output, restore in [(120, first, None), (200, second, first)]:
                args = [*command, f"--frames={frames}", f"--trace={output}"]
                if restore:
                    args.append(f"--restore={restore}")
                run = subprocess.run(args, capture_output=True, text=True, timeout=90)
                assert run.returncode == 0 and "GODOT_OK" in run.stdout, run.stdout + run.stderr
            a, b = (json.loads(p.read_text()) for p in (first, second))
            assert len(a["frames"]) == 120 and len(b["frames"]) == 80
            reference = make_demo("dodge", model=model)
            for seq, actual in enumerate(a["frames"] + b["frames"]):
                expected = reference.step()
                equal(actual["observation"], expected.observation)
                equal(
                    actual["response"],
                    {
                        "seq": seq,
                        "duration_ms": 20.0,
                        "brain_tick": expected.brain_tick,
                        "requested": expected.requested,
                    },
                )
                equal(actual["applied"], expected.applied)
                equal(actual["environment"], expected.environment)
            equal(json.loads(b["controller_json"])["brain"], reference.brain.snapshot())
            equal(b["environment"], reference.environment.snapshot())
            assert any(abs(f["applied"]["steer"]) > 0 for f in a["frames"])
            # Corrupt only the transport sequence; the engine must not apply it.
            original_dispatch = server.bridge.dispatch

            def stale(path, data):
                response = original_dispatch(path, data)
                if path == "/offer":
                    response["seq"] += 0.5
                return response

            server.bridge.dispatch = stale
            fault_file = directory / "fault.json"
            failed = subprocess.run(
                [*command, "--frames=1", f"--trace={fault_file}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert failed.returncode == 1 and "Outdated action rejected" in failed.stderr
            diagnostic = json.loads(fault_file.read_text())
            assert diagnostic["environment"]["frame"] == 0 and diagnostic["applied"] == 0
            server.bridge.dispatch = original_dispatch
            # The world clock must match the opaque, validated brain checkpoint.
            bad = dict(a)
            bad["environment"] = {**a["environment"], "frame": 121}
            bad["next_seq"] = 121
            bad_file = directory / "mismatch.json"
            bad_file.write_text(json.dumps(bad))
            failed = subprocess.run(
                [*command, "--frames=200", f"--restore={bad_file}", f"--trace={fault_file}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert failed.returncode == 1 and "clocks differ" in failed.stderr
            print(
                f"PASS {model}: 200 Godot frames, restore at 120, full parity; "
                "stale action and mismatched save rejected"
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("godot")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        for model in ("toy", ROOT / "models/male-cns-escape-v1/model.json"):
            verify(str(Path(args.godot).resolve()), model, Path(temp))
