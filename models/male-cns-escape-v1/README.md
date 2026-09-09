# MaleCNS escape-circuit model card

**Experimental real anatomical subgraph. No CUDA required.** This bundle contains
313 neurons selected by the exact source types LC4 (126), LPLC2 (185), and DNp01
(2), with 20,607 directed edges representing 79,112 reconstructed synaptic
contacts. The graph is real; its LIF dynamics, effective weights, artificial
stimuli and action scaling are modeling choices, not recovered fly physiology.

## Data and attribution

Source: [MaleCNS v1.0, minimum-confidence 0.5 flat connectome](https://male-cns.janelia.org/download/),
FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology
and Google Research. Original anatomy and this selected/normalized derivative
are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Selection, normalization, metadata packaging and software adaptation by
flybrain-sdk contributors. SDK code remains MIT, independently of data licensing.

The choice of LC4/LPLC2 and giant-fiber readout is informed by the
[giant-fiber looming study](https://doi.org/10.1016/j.cub.2019.01.079).
This reference motivates the experiment; it does not validate our parameterization.

`recipe.json` freezes all source IDs, source URLs, file sizes, SHA256 values,
selection and normalization. Source hashes were computed by this project from
downloads at official URLs; they are not publisher-issued signatures.
`model.json` retains raw contact counts, source tracing status and transmitter
predictions, and embeds the recipe. Both GF IDs are preserved: `10001` and `10010`.
Source labels include roughly/preliminarily traced cells; do not call this a
uniformly proofread model.

Bundle SHA256:

```text
ff38cfff0c76345cc2f250e992f95b0bf19863b434d8f97f926da6975cb93a35
```

Bundle size: 3,813,814 bytes. Model fingerprint (canonical neural graph and
annotations, distinct from the complete file checksum):

```text
47a0c91b3eba9c606c29faef58fe15c8846fc049c4314a5e3b6912ab22c5e68a
```

## Exact transformation

The importer keeps all source connections with both endpoints in the frozen
313-cell selection, aggregates repeated pairs, and requires at least one contact
per retained edge. It never invents connections. The original connection file
also contains unselected cells and reconstruction fragments; its 151,856,684
rows are not a count of connections in this runnable subgraph or of synaptic
contacts in a complete biological CNS.

For a retained connection from i to j:

```text
w[i,j] = explicit_type_sign[i] * 14 * raw_count[i,j]
         / sum(raw_count[k,j] for retained k)
```

All three selected types are assigned +1 by this recipe. The source transmitter
annotations are retained separately; no universal transmitter-to-sign rule is
claimed. The total retained incoming weight magnitude is 14 per nonempty target.
Normalizing only retained inputs changes the effective gain of this incomplete
circuit. The gain was selected to yield an observable transient in the assumed
LIF model, not fitted to animal recordings.

Dynamics: `lif-exact-v1`, normalized voltage/current units, dt 1 ms, membrane time
constant 10 ms, rest/reset 0, threshold 1, refractory 2 ms, one-tick synaptic
delivery, sensory input gain 2, firing-rate filter 50 ms. See
[reference equations](../../docs/architecture.md). No additional synaptic decay,
gap junctions, plasticity, retinal optics, biomechanics or muscles are simulated.

## Inputs and output

`looming_left` and `looming_right` inject uniform current directly into same-side
LC4 and LPLC2 cells. `lc4_left/right` and `lplc2_left/right` allow separate tests.
These names describe pre-encoded experimental inputs; they do not accept images
or encode measured angular size/velocity tuning.

`jump` is the clipped mean filtered rate of the two GF cells divided by 100 Hz.
It is an experimental intensity, not jump probability or a physical movement.
A game must supply its own action mapping or bind a custom readout.

With the supplied parameters, a 100 ms combined left-side pulse produces a
transient GF response which decays after input ends. LC4-only and LPLC2-only
responses differ; notably LPLC2-only stimulation need not trigger the GF under
these parameters. The model does not claim to reproduce the full measured looming
response or natural escape behavior.

## Run and modify (SDK 0.2 alpha)

From the repository after editable installation:

```sh
python examples/real_connectome.py
```

The small model ships in this repository/source distribution, separately from
the Python wheel. The SDK catalog supports opt-in download and verified caching:

```python
from flybrain import FlyBrain

brain = FlyBrain.load("male-cns-escape-v1", download=True)
lc4_left = brain.neurons.select(cell_type="LC4", side="L")
gf = brain.neurons.select(cell_type="DNp01")
brain.bind_readout({"flash": gf})
brain.drive.current(lc4_left, amplitude=2, duration_ms=100)
brain.advance(duration_ms=100)
print(brain.observe(gf, fields=["rates_hz"]).to_dict())
print(brain.action().to_dict())
```

Silence GF with `brain.intervene.silence(gf)`, or edit a copy of the graph to remove
its incoming edges. Historical rate estimates decay rather than resetting when
cells are silenced. Save/restore retains custom readouts, active currents,
annotations and intervention state.

## Reproduce from original files

This is optional: normal use downloads only the 3.8 MB model, not the 1.1 GB sources.

```sh
python -m pip install -e ".[datasets]"
python -m flybrain models download male-cns-v1.0 --cache-dir data-cache
python scripts/build_malecns_escape.py data-cache/male-cns-v1.0 --output reproduced.json
```

The converter verifies all three source SHA256 values before conversion and
streams the large connection file in Arrow batches. Compare the complete output
file SHA256 with the value above. It requires the exact files in the recipe;
future source revisions must receive a new recipe and model version.

## Software validation and performance

Tests cover source checksum rejection, duplicate-pair aggregation, missing IDs,
explicit positive/negative signs, exact counts, stimulus response and decay,
removal of GF incoming edges, and checkpoint continuation. These are software and
within-model causal tests, not comparisons against animal recordings.

`benchmark.json` records five end-to-end CPU runs of 1,000 ticks with `step()` plus
`action()` each tick. On the recorded macOS arm64 / Python 3.9.6 / NumPy 2.0.2
environment, median time was 0.1043 s for 1 s of simulated time (~9.59× real time).
Loading is excluded; this applies only to this 313-cell model, not a full brain.

```sh
python scripts/benchmark.py --model models/male-cns-escape-v1/model.json
```

Full-brain runtime, physiological accuracy, learned behavior and CUDA/WASM
execution are not established by this bundle.
