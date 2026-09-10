"""Loopback-only Godot development bridge. Run from an installed SDK environment."""

import argparse
import json
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from flybrain.demos import make_demo
from flybrain.external import ExternalController


class Bridge:
    def __init__(self, model="toy", download=False):
        demo = make_demo("dodge", model=model, download=download)
        self.initial = ExternalController(demo.brain, demo.encoder, demo.readout).snapshot()
        self.controller = None
        self.session_id = None

    def dispatch(self, path, data):
        if path == "/start":
            if "checkpoint_json" in data:
                controller = ExternalController.from_snapshot(json.loads(data["checkpoint_json"]))
            else:
                controller = ExternalController.from_snapshot(self.initial)
            if controller.period_ms != 20:
                raise ValueError("Godot reference arena requires 20 ms")
            self.controller, self.session_id = controller, uuid.uuid4().hex
            return {
                "session": self.session_id,
                "next_seq": controller.next_seq,
                "base_tick": controller.base_tick,
                "period_ms": controller.period_ms,
                "model": controller.brain.model.name,
            }
        if not self.controller or data.get("session") != self.session_id:
            raise ValueError("stale or missing session; reset both participants")
        if path == "/offer":
            return self.controller.offer(data["seq"], data["observation"])
        if path == "/ack":
            self.controller.acknowledge(data["seq"], data["applied"])
            return {"ack": data["seq"]}
        if path == "/checkpoint":
            return {"checkpoint_json": json.dumps(self.controller.snapshot(), allow_nan=False)}
        raise ValueError("unknown endpoint")


def make_server(port=8766, *, model="toy", download=False):
    bridge = Bridge(model, download)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                # Browsers receive no cross-origin access to this local dev service.
                if (
                    self.headers.get("Origin")
                    or self.headers.get_content_type() != "application/json"
                ):
                    raise ValueError("native JSON client required")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_000_000:
                    raise ValueError("request limit: 16 MB")
                self.connection.settimeout(5)
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("JSON object required")
                result = bridge.dispatch(self.path, data)
                code = 200
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
                result, code = {"error": str(error)}, 400
            raw = json.dumps(result, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.bridge = bridge
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", default="toy")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print("Preparing model before accepting game connections…", flush=True)
    with make_server(args.port, model=args.model, download=args.download) as server:
        print(f"Godot bridge: http://127.0.0.1:{server.server_port} · {args.model}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
