# Circuit lab: try, export, build

**No CUDA required.** [Open the circuit lab](https://freeman-1984-coder.github.io/flybrain-sdk/lab.html), apply a stimulus, and press **Run** or **+20 ms**. The browser runs the SDK's float64 JavaScript CPU implementation in a Web Worker.

The initial 12-cell toy is artificial. **Load real circuit** explicitly downloads a 3.8 MB pinned MaleCNS anatomical subgraph (313 cells, 20,607 edges). The loader verifies size and SHA256 before parsing, and rechecks cached bytes on reuse. The page needs a connection to load its code; an exported standalone demo needs no connection. No simulation or recording is uploaded.

## A useful first experiment

1. Load the real circuit and apply `looming_left` for 100 ms at strength 1. Advance several steps and watch LC4, LPLC2 and GF rates.
2. Reset. Silence **All output neurons**, apply the same stimulus, and advance again. Input activity can continue while new GF spikes are suppressed.
3. Release the cells, inspect an individual source ID, or export the recording.

The input is an artificial current pulse into annotated visual populations, not a retinal image. `jump` is a normalized GF mean rate, not a probability or measured escape behavior. Curves connect groups with retained edges; positions are schematic, not anatomy. Rate history samples the end of each 20 ms batch and is not a spike raster. Previously emitted spikes still propagate after silencing; rate history decays naturally.

## Build a demo without setup

**Download editable demo** creates one HTML file containing the selected model, runtime, license and input/output example. Open it locally, then edit the section marked `EDIT HERE` in a text editor. Connect the returned channel values to your game's objects, audio parameters or controls. The exported demo starts at rest, rather than copying the current experiment state. Its small reference loop runs on the main thread; use a worker for sustained interactive applications.

Code is MIT. The real model's derived anatomy is CC BY 4.0; its source attribution travels with the demo. Preserve both when sharing. Read the [model card](../models/male-cns-escape-v1/README.md) for source versions, signs, normalization and limits.

## Replay the same inputs in Python

From an installed repository checkout:

```sh
pip install -e .
python examples/replay_experiment.py flybrain-experiment.json --download
```

`--download` permits the pinned catalog model to download when needed. Once cached, omit it to replay offline. The recording contains model fingerprint, asset checksum, LIF configuration, ordered stimulus/intervention commands and the expected final observation. Replay uses only the installed catalog: recorded URLs or executable code are never loaded. The CLI limits input to 20 MB and one million ticks by default.

```python
import json
from flybrain.experiment import replay_experiment

with open("flybrain-experiment.json") as stream:
    brain = replay_experiment(json.load(stream), download=True)
brain.save("continued.checkpoint.json")
brain.advance(duration_ms=100)
```

Replay checks exact final spikes and clock, and voltage, rate and action values at absolute tolerance `1e-9`. It verifies the recorded endpoint, not every intermediate tick. Returned Python state retains pending stimuli and silencing and can continue. Arbitrary custom readouts, custom graphs and learning updates are not part of this lab recording schema yet.

## Runtime parity and development

The JavaScript core supports real and toy graphs, normalized LIF parameters, named sensory current, direct current injection, annotation selection, selected observations, custom readouts, reversible silencing and JSON save/restore. CPU works; CUDA and WASM still fail explicitly. This is source-available TypeScript, not a package published on NPM.

```sh
pip install -e '.[dev]'
python scripts/browser_reference.py
cd packages/js
npm ci
npm test
cd ../..
node scripts/build-browser-site.mjs
pytest
python -m http.server 8765 --directory site
```

Tests compare Python and JS at every cell of 500 toy/real ticks, including inhibition, silencing, refractory state and checkpoint continuation. Spikes must match exactly; rates and voltages use `1e-10` absolute tolerance. The website's actual worker generates committed recording fixtures which Python replays offline. Exported HTML is executed in an isolated JS context with no network globals. These fixtures demonstrate the supported cases, not universal numerical equivalence on arbitrary graphs/hardware.

JavaScript `advance(ticks)` takes integer ticks, while Python `advance(duration_ms=...)` takes aligned milliseconds; both `step(ticks)` methods take ticks. JavaScript `load` takes parsed model data, with fetching owned by the caller. JS checkpoints use `flybrain-js-checkpoint`; Python checkpoints use schema 2. Do not exchange checkpoint files directly: use experiment replay for the supported cross-language workflow. Model IDs stay strings in both languages.
