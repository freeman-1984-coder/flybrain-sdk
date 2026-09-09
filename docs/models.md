# Model catalog and on-demand downloads

The SDK ships URLs and metadata, not bulk biological datasets. Listing models is
offline. Downloads happen only through `fetch_model()` or the `models download`
command. This makes a small installation useful immediately, while allowing
contributors to add sources over time.

| Catalog ID | Runnable now? | Files |
| --- | --- | --- |
| `toy` | Yes, builtin | 12 synthetic neurons / 12 edges |
| `male-cns-v1.0` | No; raw data | Connections 1,051,241,946 B; annotations 14,483,314 B; transmitter predictions 43,282,834 B |
| `flywire-v783` | No; raw data | Proofread connections 852,022,274 B; neuron IDs 1,114,168 B |

Sources: [MaleCNS official downloads](https://male-cns.janelia.org/download/),
[FlyWire pinned archive](https://zenodo.org/records/10676866). Both catalog entries
retain CC-BY-4.0 attribution requirements. Metadata and file sizes were checked
on 2026-09-09. Upstream files can change; report a failed size/checksum verification.

```python
from flybrain import fetch_model

paths = fetch_model(
    "male-cns-v1.0",
    assets=["annotations"],
    cache_dir="./data-cache",
    max_bytes=20_000_000,
)
```

Default cache: `~/.cache/flybrain-sdk/<catalog-id>/`. Set `cache_dir` to keep it
elsewhere. `assets=None` downloads every catalog asset for the selected entry;
select assets to avoid unneeded downloads. The default 2 GB limit is **per asset**.
Inspect total sizes using `model_info()` first. Downloads stream to temporary files,
verify declared size and any available pinned checksum, then replace the target.
Interrupted/invalid transfers leave no partial target. There is no resume support
or multi-process cache locking in this alpha.

FlyWire archive MD5 values are pinned as published upstream; they detect transfer
corruption, not adversarial tampering. MaleCNS currently has no catalog SHA256 pin;
HTTPS and the declared size are checked, then a SHA256 receipt is recorded on first
fetch. Subsequent cache reads re-hash against that receipt. A local receipt is not
independent source authentication. Adding publisher-verified SHA256 pins is welcome.
The downloader rejects HTML login pages and never unpickles or executes a download.

To add a model, edit `src/flybrain/data/registry.json` through a pull request.
Include immutable/versioned HTTPS URLs, byte sizes, license, source/citation links,
checksums where possible and status (`builtin` or `raw-data` in this release).
Do not mark raw wiring as a runnable physiological model. A future `ready` status
needs a reviewed importer, explicit parameter/mapping files and behavioral tests.
