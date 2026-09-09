# From MaleCNS / FlyWire data to a runnable model

## Implemented in 0.2 alpha

A frozen-selection MaleCNS importer now lives in `flybrain.importers.malecns`,
behind the optional `datasets` extra. The first [published model and recipe](../models/male-cns-escape-v1/README.md)
retain original IDs, counts, annotations, source SHA256 pins and assumed dynamics.
It runs without Arrow or CUDA after conversion. See that model card for exact
reproduction commands and validation. Full-brain calibration and the FlyWire
importer remain future work. The original broader integration plan follows.


The first release solves installation, CPU dynamics, a shared API and explicit
downloads. The initial milestone was a **small, reproducible real-connectome subgraph**,
not a claim that the complete brain is already simulated accurately.

## 1. Pin the source

MaleCNS: start from the catalog's versioned Feather connections, annotations and
transmitter tables. The official alternative is `neuprint-python` against dataset
`male-cns:v1.0`; use a user-supplied token for that route. Keep authentication out
of the core package. [Official download and API instructions](https://male-cns.janelia.org/download/).

FlyWire: start with the pinned v783 Zenodo archive in the catalog. Preserve source
root IDs as strings. Treat newer Codex snapshots as separate dataset versions;
never silently mix identifiers or annotations across releases.
[Publication archive](https://zenodo.org/records/10676866),
[Codex source/version guidance](https://codex.flywire.ai/faq).

## 2. Add isolated importers

MaleCNS now implements this design; add `flybrain.importers.flywire` behind the
`[datasets]` extra containing Arrow tooling. Inspect actual file schemas and record
a tested column mapping. Keep core install dependencies unchanged. Each importer
should accept downloaded paths, a frozen list of neuron IDs and a conversion policy.
Select 100–1,000 neurons initially, retain only edges whose endpoints are in the
selection, aggregate duplicates, and emit a report of removed endpoints/edges.
A source filename is not enough to infer its columns or biological scope.

Deliver `model.json`, `mapping.json`, and `provenance.json` with original IDs,
source URL/version/license/citations, source and output SHA256, importer version,
selection criteria, confidence filters and all weight/parameter choices. Use a
compact Arrow/NPZ array format for later large networks; load NPZ with pickle disabled.

## 3. Make physiological assumptions explicit

Synapse counts are not calibrated currents. Preserve raw counts and transmitter
predictions separately from simulation weights. Review receptor context and
uncertainty; do not hardcode one excitatory/inhibitory sign for every transmitter.
Require an explicit unknown-transmitter policy. Document gain/normalization,
thresholds, time constants, refractory periods and delays as modeling assumptions.
Do not reuse the toy's weight=16 as a biological estimate.

## 4. Map inputs and outputs with evidence

Choose a documented circuit and reviewed neuron list. Encode stimuli into its
sensory population, and expose a small descending/motor readout. Keep mappings in
versioned data files with literature/source references. FlyWire brain-only data
needs a deliberate boundary/readout rather than pretending it contains a complete
body controller. Keep the game adapter and body physics outside the neural core.

## 5. Validate before adding a runnable catalog entry

- Structural checks: unique IDs, endpoint coverage, counts, edge orientation,
  transmitter policy, exclusions, and source fingerprints.
- Numerical checks: small analytically solvable circuits, inhibition, refractory
  timing, deterministic continuation and stable activity across input strengths.
- Behavioral checks: baseline vs stimulation, left/right or circuit-specific
  responses, and agreement with selected experimental evidence where available.
- Performance checks: measured N/E, peak memory, ms of simulation per wall second,
  hardware, timestep, dependency versions and precision. Do not extrapolate a toy
  timing into full-brain real-time claims.

Release a model card and a CPU quickstart for the selected subgraph. Only then add
`FlyBrain.load(<ready-model-id>)` with explicit opt-in download, pinned model assets,
and CPU/WASM/CUDA conformance fixtures. Contributions can target each step independently.
