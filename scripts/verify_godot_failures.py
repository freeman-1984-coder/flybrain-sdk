"""Actual Godot transport fault checks; runs separately from numerical conformance."""

import argparse
import json
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from verify_godot import ROOT, bridge

CASES = {
    "missing_start": (0, "Invalid start response"),
    "fractional_clock": (0, "Invalid start response"),
    "missing_offer": (0, "Outdated action rejected"),
    "invalid_steering": (0, "Invalid steering response"),
    "missing_ack": (1, "Acknowledgement mismatch"),
    "missing_checkpoint": (1, "Invalid checkpoint response"),
    "disconnect": (1, "Bridge rejected/unavailable"),
    "timeout": (1, "Bridge rejected/unavailable"),
}


def verify(binary, mode, directory):
    controller = bridge.Bridge()

    class FaultHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = controller.dispatch(self.path, data)
            if self.path == "/start":
                if mode == "missing_start":
                    reply = {"session": reply["session"]}
                elif mode == "fractional_clock":
                    reply["base_tick"] = 0.5
            elif self.path == "/offer":
                if mode == "missing_offer":
                    reply = {}
                elif mode == "invalid_steering":
                    reply["requested"] = {"steer": "not a number"}
                elif data["seq"] == 1:
                    # Neural integration already happened, but Godot never receives it.
                    if mode == "disconnect":
                        self.connection.shutdown(socket.SHUT_RDWR)
                        self.connection.close()
                        self.close_connection = True
                        return
                    if mode == "timeout":
                        time.sleep(2.6)
            elif self.path == "/ack" and mode == "missing_ack":
                reply = {}
            elif self.path == "/checkpoint" and mode == "missing_checkpoint":
                reply = {}
            raw = json.dumps(reply).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the client rejects a late response.

    expected_frame, message = CASES[mode]
    with HTTPServer(("127.0.0.1", 0), FaultHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            output = directory / f"{mode}.json"
            command = [
                binary,
                "--headless",
                "--path",
                str(ROOT / "examples/godot"),
                "--",
                f"--endpoint=http://127.0.0.1:{server.server_port}",
                f"--frames={2 if mode in ('disconnect', 'timeout') else 1}",
                f"--trace={output}",
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            assert result.returncode == 1 and message in result.stderr, (
                result.stdout + result.stderr
            )
            state = json.loads(output.read_text())
            assert state["failed"] is True
            assert state["environment"]["frame"] == expected_frame
            assert state["applied"] == 0
            if mode in ("disconnect", "timeout"):
                assert controller.controller.brain.progress.tick == 40
                assert controller.controller.pending["seq"] == 1
                assert controller.controller.next_seq == 1
            print(f"PASS {mode}: Godot stopped at frame {expected_frame}, controls cleared")
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("godot")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        for name in CASES:
            verify(str(Path(args.godot).resolve()), name, Path(temporary))
