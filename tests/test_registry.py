import hashlib
import importlib
import io
import json

import pytest

from flybrain import Connectome, FlyBrain, fetch_model, list_models, model_info, registry


class Response(io.BytesIO):
    headers = {"Content-Type": "application/octet-stream"}

    def geturl(self):
        return "https://example.org/model.bin"


def fake_catalog(monkeypatch, payload=b"model data", checksum=None):
    asset = {
        "filename": "model.bin",
        "size_bytes": len(payload),
        "url": "https://example.org/model.bin",
    }
    if checksum is not None:
        asset["checksum"] = checksum
    monkeypatch.setattr(
        registry, "model_info", lambda _: {"status": "raw-data", "assets": {"connections": asset}}
    )
    return asset


def test_listing_never_downloads(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("unexpected network request")

    monkeypatch.setattr(registry, "urlopen", blocked)
    assert {m["id"] for m in list_models()} == {
        "toy",
        "male-cns-escape-v1",
        "male-cns-v1.0",
        "flywire-v783",
    }
    assert model_info("flywire-v783")["status"] == "raw-data"
    assert model_info("male-cns-v1.0")["assets"]["connections"]["size_bytes"] > 1_000_000_000


def test_builtin_fetch_and_load_offline(tmp_path):
    path = fetch_model("toy", cache_dir=tmp_path)["model"]
    assert FlyBrain.load(path).model.name == "toy-v1"


def test_download_checksum_cache_and_corruption_recovery(tmp_path, monkeypatch):
    payload = b"model data"
    fake_catalog(monkeypatch, payload, "sha256:" + hashlib.sha256(payload).hexdigest())
    calls = []

    def opener(*args, **kwargs):
        calls.append(1)
        return Response(payload)

    monkeypatch.setattr(registry, "urlopen", opener)
    path = fetch_model("test", cache_dir=tmp_path)["connections"]
    assert path.read_bytes() == payload
    receipt = json.loads(path.with_name(path.name + ".receipt.json").read_text())
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert fetch_model("test", cache_dir=tmp_path)["connections"] == path
    assert len(calls) == 1
    path.write_bytes(b"bad cache!")
    fetch_model("test", cache_dir=tmp_path)
    assert len(calls) == 2
    assert path.read_bytes() == payload


@pytest.mark.parametrize(
    "payload,checksum",
    [
        (b"short", None),
        (b"too much model data", None),
        (b"model data", "sha256:" + "0" * 64),
    ],
)
def test_bad_download_never_becomes_cached_asset(tmp_path, monkeypatch, payload, checksum):
    fake_catalog(monkeypatch, checksum=checksum)
    monkeypatch.setattr(registry, "urlopen", lambda *a, **k: Response(payload))
    with pytest.raises(ValueError):
        fetch_model("test", cache_dir=tmp_path)
    assert not list(tmp_path.rglob("model.bin"))
    assert not list(tmp_path.rglob(".download-*"))


def test_limit_and_asset_validation_before_network(tmp_path, monkeypatch):
    fake_catalog(monkeypatch)

    def blocked(*args, **kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr(registry, "urlopen", blocked)
    with pytest.raises(ValueError, match="max_bytes"):
        fetch_model("test", cache_dir=tmp_path, max_bytes=1)
    with pytest.raises(ValueError):
        fetch_model("test", cache_dir=tmp_path, assets=["unknown"])
    with pytest.raises(ValueError):
        fetch_model("test", cache_dir=tmp_path, assets="connections")


def test_model_catalog_is_detached():
    entries = list_models()
    entries[0]["id"] = "changed"
    assert list_models()[0]["id"] == "toy"


def test_ready_model_opt_in_load_cache_and_integrity(tmp_path, monkeypatch):
    payload = json.dumps(Connectome.load().to_dict()).encode()
    asset = {
        "filename": "model.json",
        "url": "https://example.org/model.json",
        "size_bytes": len(payload),
        "checksum": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }
    entry = {"id": "test-ready", "status": "ready", "assets": {"model": asset}}
    monkeypatch.setattr(registry, "list_models", lambda: [entry])
    monkeypatch.setattr(importlib.import_module("flybrain.brain"), "list_models", lambda: [entry])
    calls = []

    def opener(*args, **kwargs):
        calls.append(1)
        return Response(payload)

    monkeypatch.setattr(registry, "urlopen", opener)
    with pytest.raises(FileNotFoundError):
        FlyBrain.load("test-ready", cache_dir=tmp_path)
    assert calls == []
    brain = FlyBrain.load("test-ready", cache_dir=tmp_path, download=True)
    assert brain.model.name == "toy-v1"
    assert len(calls) == 1
    assert FlyBrain.load("test-ready", cache_dir=tmp_path).state == brain.state
    assert len(calls) == 1
    path = tmp_path / "test-ready" / "model.json"
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(FileNotFoundError):
        FlyBrain.load("test-ready", cache_dir=tmp_path)
    assert len(calls) == 1
    FlyBrain.load("test-ready", cache_dir=tmp_path, download=True)
    assert len(calls) == 2
    assert path.read_bytes() == payload
    del asset["checksum"]
    with pytest.raises(ValueError, match="SHA256"):
        FlyBrain.load("test-ready", cache_dir=tmp_path, download=True)
    assert len(calls) == 2
