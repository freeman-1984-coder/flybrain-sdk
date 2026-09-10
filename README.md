# flybrain-sdk

**No CUDA required.** A Python SDK for connecting small connectome simulations to games and experiments, starting with a working NumPy CPU backend.

**Try without installing:** [Circuit lab](https://freeman-1984-coder.github.io/flybrain-sdk/lab.html) — stimulate toy or real cells, silence outputs, inspect rates, download an editable HTML demo, or [replay an exported experiment in Python](docs/browser-lab.md).

[![CI](https://github.com/freeman-1984-coder/flybrain-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/freeman-1984-coder/flybrain-sdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Project website](https://freeman-1984-coder.github.io/flybrain-sdk/) · [API](docs/api.md) · [Model catalog](docs/models.md) · [Contribute](CONTRIBUTING.md) · [中文](docs/README.zh-CN.md)

```python
from flybrain import FlyBrain

brain = FlyBrain.load("toy", backend="cpu")
brain.stimulate("food", strength=0.9, duration_ms=100)
brain.step(100)
print(brain.action().to_dict())
```

**0.4 alpha:** the bundled offline demo is a hand-authored 12-neuron circuit. A separate 3.8 MB MaleCNS model now runs 313 real source neurons and 20,607 anatomical edges with explicitly assumed LIF parameters. [Model card and reproducible recipe](models/male-cns-escape-v1/README.md). CUDA and WASM are reserved interfaces, not implemented runtimes. No GPU, credentials, or network access are needed to run the toy demo after installation.

## Game and audio demo kits

The same `Session` loop now supports replaceable feature encoders, rate readouts and environments, with full checkpoints and verified feedback replay. Try [the demo gallery and original circuit audio](https://freeman-1984-coder.github.io/flybrain-sdk/demos.html), then generate your own:

```sh
python examples/demo_session.py dodge --output work/dodge
python examples/demo_session.py tones --output work/tones
```

These produce recordings, checkpoints and standalone playback HTML; tones also exports WAV. Gallery viewers play recorded Python runs, while the circuit lab runs live. See the [session and adapter guide](docs/demo-kits.md) for code and limitations.

## Install and run

Python 3.9+ on a NumPy-supported platform. Development installation:

```sh
git clone https://github.com/freeman-1984-coder/flybrain-sdk.git
cd flybrain-sdk
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python examples/quickstart.py
python -m pip install -e ".[dev]"
pytest
```

The package is **not yet published to PyPI**. Install directly from GitHub without cloning:

```sh
python -m pip install "flybrain-sdk @ git+https://github.com/freeman-1984-coder/flybrain-sdk.git"
```

Normal installation needs only NumPy at runtime. Offline operation means the demo makes no network requests; initial dependency installation needs an existing wheel cache or internet access.

## One API, explicit backend support

| API | Behavior |
| --- | --- |
| `FlyBrain.load("toy", backend="cpu")` | Load bundled data; a local model JSON path or `Connectome` also works |
| `brain.stimulate("looming_left", strength=1, duration_ms=100)` | Start a finite current pulse on the next tick |
| `brain.step(20)` | Advance 20 fixed simulation ticks, return final immutable state |
| `brain.action()` | Read independent motor intensities without advancing time |
| `brain.save("brain.checkpoint.json")` | Atomically save model, config, dynamics, and pending stimuli |
| `FlyBrain.restore("brain.checkpoint.json")` | Create a brain that continues the saved trajectory |

The toy sensory channels are `food`, `looming_left`, `looming_right`, `touch`.
Motor channels are `walk`, `turn_left`, `turn_right`, `jump`, each between 0 and 1.
They are heuristic control values, not probabilities or measured physical velocities.
Game movement and rendering stay in your application; see [game_loop.py](examples/game_loop.py).

| Backend | Current status |
| --- | --- |
| `cpu` | Working reference implementation, NumPy edge lists, float64 |
| `wasm` | Reserved name; raises `BackendUnavailableError` |
| `cuda` | Reserved name; raises `BackendUnavailableError`; no CUDA dependencies |

See [dynamics and backend contract](docs/architecture.md) for equations, spike timing and limitations.

## Models are downloaded only when requested

Run the real anatomical subgraph with an explicit first download:

```python
brain = FlyBrain.load("male-cns-escape-v1", download=True)
brain.stimulate("looming_left", duration_ms=100)
brain.advance(duration_ms=100)
gf = brain.neurons.select(cell_type="DNp01")
print(brain.observe(gf, fields=["rates_hz"]).to_dict())
```

The model is experimental: wiring is measured, while its dynamics and mappings
are assumptions. No Arrow/CUDA dependency is needed to use the converted bundle.

Use your own output names and directly drive or silence selected cells:

```python
brain.bind_readout({"flash": gf})
brain.drive.current(gf, amplitude=2, duration_ms=20, units="normalized")
brain.advance(duration_ms=20)
print(brain.action()["flash"])
brain.intervene.silence(gf)  # enabled=False releases the intervention
```

See [the offline open-circuit example](examples/open_circuit.py) and
[real-model reproduction](models/male-cns-escape-v1/README.md).

```python
from flybrain import list_models, model_info, fetch_model

print([(m["id"], m["status"]) for m in list_models()])  # offline catalog
print(model_info("male-cns-v1.0"))  # includes URLs, sizes, license

# Explicit network request: download only this ~1.1 MB asset, then reuse local cache.
paths = fetch_model("flywire-v783", assets=["neuron_ids"])
print(paths["neuron_ids"])
```

```sh
python -m flybrain models list
python -m flybrain models info male-cns-v1.0
python -m flybrain models download male-cns-v1.0 --asset annotations
```

`fetch_model()` returns file paths. Raw Feather/NPY data cannot yet be passed straight to `FlyBrain.load()`.
The wheel contains no large model. Biological model downloads require explicit `download=True`; importing the package and loading the toy remain offline. See the [catalog and cache contract](docs/models.md) and [real-data integration plan](docs/real-data.md).

## Scope and scientific honesty

The SDK offers plumbing for experiments, not a validated emulation of a fly. Wiring alone does not specify physiological parameters, receptor effects, sensory encoding, motor decoding, a body, or learning. The demo has no plasticity, morphology, realistic vision, or inferred biological behavior. Full-brain CPU real-time performance has not been established.

Code and original toy data are MIT. Third-party datasets retain their own licenses and required citations; they are not relicensed by this repository. This is an independent project, unaffiliated with Janelia, FlyWire, Google, or the dataset authors.

## Help build it

Issues and pull requests are welcome. Good starting areas:

- Add a versioned model source with license, attribution, sizes and integrity metadata.
- Add a FlyWire importer or another MaleCNS recipe with explicit mappings and validation.
- Add a Godot/Unity adapter or a headless game example.
- Port the reference dynamics to WASM and match Python reference traces.

Read [CONTRIBUTING.md](CONTRIBUTING.md), [roadmap](docs/roadmap.md), and the [TypeScript CPU runtime](packages/js/README.md). New contributors can use the model request, bug report, or feature request templates. Pull requests run CI before review.

## Build a distribution

```sh
python -m build
python -m twine check dist/*
```

The wheel includes the toy connectome and model catalog. See [release instructions](docs/releasing.md) before publishing.
