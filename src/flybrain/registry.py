"""Curated model sources with explicit, streamed, cached downloads.

Raw connectomes are data, not automatically calibrated runnable neural models.
Importing this module or listing sources never accesses the network.
"""

import hashlib
import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Dict, Optional, Sequence, Union
from urllib.request import Request, urlopen

from .config import positive_int


def list_models() -> list:
    """Return detached catalog entries; inspect status and asset sizes before downloading."""
    return json.loads(files("flybrain.data").joinpath("registry.json").read_text(encoding="utf-8"))


def model_info(model_id: str) -> dict:
    for entry in list_models():
        if entry["id"] == model_id:
            return entry
    raise ValueError(f"unknown model {model_id!r}; see list_models()")


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_asset(asset: dict, folder: Path, max_bytes: int) -> Path:
    filename = asset["filename"]
    if Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError("invalid asset filename")
    if not asset["url"].startswith("https://"):
        raise ValueError("model downloads require HTTPS")
    size = positive_int(asset["size_bytes"], "size_bytes")
    if size > max_bytes:
        raise ValueError(f"{filename} needs {size} bytes, exceeding max_bytes={max_bytes}")
    target = folder / filename
    receipt = folder / (filename + ".receipt.json")
    checksum = asset.get("checksum")
    algorithm, expected = checksum.split(":", 1) if checksum else ("sha256", None)
    if algorithm not in ("sha256", "md5"):
        raise ValueError("unsupported source checksum")
    if target.exists() and receipt.exists():
        try:
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            digest = _digest(target)
            if (
                saved["url"] == asset["url"]
                and saved["sha256"] == digest
                and target.stat().st_size == size
                and (expected is None or _digest(target, algorithm) == expected)
            ):
                return target
        except (ValueError, KeyError):
            pass
    temporary = None
    try:
        request = Request(asset["url"], headers={"User-Agent": "flybrain-sdk/0.1.0a1"})
        with urlopen(request, timeout=60) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("download redirected to an insecure URL")
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("source returned HTML (possibly a login page), not model data")
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".download-", delete=False) as out:
                temporary = Path(out.name)
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > min(max_bytes, size):
                        raise ValueError("download exceeds declared size or max_bytes")
                    out.write(chunk)
        if count != size:
            raise ValueError(f"truncated or changed asset: expected {size} bytes, got {count}")
        if expected is not None and _digest(temporary, algorithm) != expected:
            raise ValueError("source checksum mismatch")
        sha256 = _digest(temporary)
        os.replace(temporary, target)
        receipt.write_text(
            json.dumps(
                {
                    "url": asset["url"],
                    "sha256": sha256,
                    "size_bytes": size,
                    "source_checksum": checksum,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_model(
    model_id: str,
    *,
    assets: Optional[Sequence[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    max_bytes: int = 2_000_000_000,
) -> Dict[str, Path]:
    """Download selected assets; default all, max_bytes is a per-asset limit.

    Returns asset names mapped to local paths. Raw entries still need conversion.
    No archive extraction, remote code execution, credentials, or pickle loading.
    Source SHA/MD5 pins are verified where supplied; all caches get a SHA256 receipt.
    """
    positive_int(max_bytes, "max_bytes")
    entry = model_info(model_id)
    root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "flybrain-sdk"
    if entry["status"] == "builtin":
        if assets is not None and tuple(assets) != ("model",):
            raise ValueError("toy has only the 'model' asset")
        folder = root / model_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "model.json"
        path.write_text(
            files("flybrain.data").joinpath("toy.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return {"model": path}
    available = entry["assets"]
    if isinstance(assets, str):
        raise ValueError("assets must be a sequence of names, not a string")
    selected = list(available) if assets is None else list(assets)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(available):
        raise ValueError(f"assets must be unique names from {list(available)}")
    # Validate all sizes before starting a multi-file download.
    for name in selected:
        if available[name]["size_bytes"] > max_bytes:
            raise ValueError(f"asset {name!r} exceeds max_bytes={max_bytes}")
    folder = root / model_id
    folder.mkdir(parents=True, exist_ok=True)
    return {name: _fetch_asset(available[name], folder, max_bytes) for name in selected}
