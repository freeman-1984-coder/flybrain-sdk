import importlib.util
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

spec = importlib.util.spec_from_file_location(
    "bridge", Path(__file__).resolve().parents[1] / "examples/godot/bridge.py"
)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def test_transport_ownership_retries_and_opaque_checkpoint():
    with bridge.make_server(0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def post(path, data, origin=None):
            headers = {"Content-Type": "application/json"}
            if origin:
                headers["Origin"] = origin
            request = Request(
                f"http://127.0.0.1:{server.server_port}{path}",
                data=json.dumps(data).encode(),
                headers=headers,
            )
            with urlopen(request, timeout=3) as response:
                return json.load(response)

        try:
            sid = post("/start", {})["session"]
            offer = {"session": sid, "seq": 0, "observation": {"danger_left": 1, "danger_right": 0}}
            response = post("/offer", offer)
            assert post("/offer", offer) == response
            assert server.bridge.controller.brain.progress.tick == 20
            ack = {"session": sid, "seq": 0, "applied": {"steer": 0}}
            assert post("/ack", ack) == post("/ack", ack)
            checkpoint = post("/checkpoint", {"session": sid})["checkpoint_json"]
            assert type(json.loads(checkpoint)["next_seq"]) is int
            restored = post("/start", {"checkpoint_json": checkpoint})
            assert restored["next_seq"] == 1
            assert restored["session"] != sid
            with pytest.raises(HTTPError) as error:
                post("/offer", offer)
            assert error.value.code == 400
            with pytest.raises(HTTPError):
                post("/start", {}, origin="https://example.com")
            assert server.bridge.session_id == restored["session"]
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_bridge_pins_selected_backend_when_restoring():
    service = bridge.Bridge(backend="cpu")
    saved = dict(service.initial)
    saved["brain"] = dict(saved["brain"], backend="cuda")
    # Selection is caller-owned, even when saved metadata names another device.
    # This checks routing only; it does not pretend this fixture ran on CUDA.
    service.dispatch("/start", {"checkpoint_json": json.dumps(saved)})
    assert service.controller.brain.snapshot()["backend"] == "cpu"
    assert service.controller.offer(0, {"danger_left": 1, "danger_right": 0})["brain_tick"] == 20
