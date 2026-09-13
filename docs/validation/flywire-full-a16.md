# Full FlyWire v783 on NVIDIA A16-8Q

Actual-device run, 2026-09-12 UTC. Tested source commit:
`5cb825507a05a50c0a0c7200e3c130d547f77663`.
The later Windows import fix only makes Linux RSS collection optional outside Linux;
no simulation, importer mapping, or benchmark algorithm was changed by that fix.

- All 139,255 proofread neurons, including 616 isolated neurons.
- All 16,847,997 published neuron-pair/neuropil rows retained (0 discarded).
- 54,492,922 anatomical synaptic contacts represented by those rows.
- Both official MD5 checks passed; SHA-256 values are in the report.
- 12/12 actual CUDA hardware checks passed; full suite **171 passed, 0 skipped**.
- 200 active CPU/CUDA ticks, comparing every neuron every 25 ticks: exact spikes
  and refractory state; voltage and rate maximum difference was 0 at all checkpoints.
- JSON backend-state roundtrip reproduced 30 subsequent CUDA ticks exactly.
- 10,000 additional CUDA ticks (10 seconds simulated), finite/bounded full states
  checked every 100 ticks: 93,609,984 spikes, 139,254 neurons fired at least once.

| SDK backend | Three wall times for 1,000 ms simulation | Median |
| --- | --- | --- |
| NumPy CPU | 110.955, 110.815, 113.702 s | 110.955 s |
| CuPy CUDA | 11.552, 11.558, 11.549 s | 11.552 s |

CUDA was **9.61x faster than this SDK CPU reference on this host**, still about
11.55 wall seconds per simulated second (not real time). Timings include per-tick
host input uploads and scalar error synchronization. This is not a comparison
against an optimized third-party CPU simulator. Both use float64 and dt=1 ms.

Model import: 63.95 s. CPU backend initialization: 3.27 s. CUDA initialization and
compilation: 6.77 s. Stability pass including activity collection: 122.03 s.
Final CuPy pool in-use allocation: 210,532,864 bytes (~200.8 MiB); reserved:
214,013,952 bytes. These are not peak total GPU memory. Host process peak RSS:
7,122,432 KiB (~6.79 GiB). The Python object graph is costly in host memory.

Environment: A16-8Q, 8 GiB VRAM; 3 vCPU / 32 GiB system RAM; Ubuntu 24.04;
Python 3.12.3, NumPy 2.5.3, CuPy 14.2.0, NVIDIA driver 550.90.07,
CUDA runtime API 12090. See frozen dependencies for the complete installation.

## Scope and assumptions

This is the complete published **proofread brain graph**, not the entire CNS,
not raw unproofread EM fragments, and not a physiological or behavioral model.
The original 313-cell Reflex School remains a separate experiment.
Artificial currents drive this benchmark. No learning or body/game integration
is claimed here. GABA/glutamate signs, other/unknown positive signs, homogeneous
LIF parameters, and incoming normalization to 2.5 are explicit assumptions.
1,250 neurons have no outgoing transmitter-score evidence; the policy counts them
and assumes positive sign. No edges are dropped to hide missing annotations.

The first attempt failed closed because NumPy mixed int64/uint64 lookups lost
precision on long neuron IDs. Normalizing both source columns to uint64 fixed
this; four regression tests protect neighboring large IDs and unknown IDs.

The report's checkpoint is the backend state with the same model, not a benchmark
of serializing the entire graph through FlyBrain.save(). The whole-brain entry
point is currently a validation script rather than a catalog-ready game model.

- [Full JSON report](flywire-full-a16.json)
- [Full pytest XML](flywire-full-pytest.xml)
- [44 source SHA-256 values](flywire-full-source-sha256.json)
- [Frozen requirements](flywire-full-requirements.txt)
- [Reproduction guide](../fullbrain-validation.md)
- [Official dataset](https://zenodo.org/records/10676866)
