# Full FlyWire CUDA validation

This opt-in runner loads all **139,255 proofread FlyWire v783 neurons** and every
row in the official proofread connections table. It uses the SDK's actual
`Connectome`, `CPUBackend`, and `CUDABackend`, without replacing their execution
paths. No neuron selection, minimum-count pruning, or synthetic graph is used.

## Reproduce on an NVIDIA host

Use the experimental CUDA branch. A 32 GB system-memory host gives headroom for
this initial Python object-based importer. GPU memory requirements are measured
by the runner; the raw file size is not the required GPU memory.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[datasets]' 'cupy-cuda12x[ctk]==14.2.0' pytest
mkdir -p data results
curl -fL 'https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1' -o data/proofread_root_ids_783.npy
curl -fL 'https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1' -o data/proofread_connections_783.feather
python scripts/validate_fullbrain.py --data-dir data --output results/fullbrain.json
```

Download source: [FlyWire Consortium, v783.0, Zenodo](https://zenodo.org/records/10676866).
The runner checks both published MD5 values, then records SHA-256 values and byte
sizes. A missing or different file fails the test. The raw data are not bundled
in the package or redistributed by this repository.

## What is checked

- All official neuron IDs are unique; all source edge endpoints resolve.
- Every published proofread connection row is retained, including weak connections,
  self-connections, and parallel rows for different neuropils.
- The complete network rests without input, then generates spikes under seeded input.
- CPU and CUDA states across every neuron are compared every 25 ticks over 200
  active ticks: spikes and refractory counters exactly, voltage/rates within 1e-10.
- A JSON checkpoint reproduces 30 subsequent CUDA ticks exactly.
- Three 1,000 ms runs per backend follow a 100-tick warmup. Compilation/loading are
  reported separately; timed steps include host input upload and error synchronization.
- 10,000 additional CUDA ticks check finite, bounded states and record population
  activity, device allocations, and peak host memory. These observation-heavy
  timings are separate from the throughput benchmark.

## What this means biologically

The whole published proofread **brain** graph is used; this is not the whole
nervous system or the raw EM volume, and not every unproofread segment is a neuron
in this graph. Each source row aggregates anatomical contacts between a neuron
pair in a neuropil. A connection row is not one anatomical synapse.

Dynamics are dimensionless, simplified LIF assumptions. Contact counts are scaled
to incoming absolute weight sum 2.5. Each neuron's sign uses its outgoing
contact-weighted transmitter scores: GABA/glutamate negative, other labels positive.
Unknown transmitter evidence uses positive sign and is counted. This deliberately
simple policy does not model receptor-specific effects or neuromodulation.

The benchmark uses artificial currents, not reconstructed vision, behavior,
physiological calibration, or biological learning. Passing establishes full-graph
software execution and numerical checks for the reported configuration. It does
not establish that a living fly's complete brain function has been reproduced.
