# Complete FlyWire synaptic CUDA odor experiment

**Sensory propagation observed, 2026-09-13 UTC. Navigation not established.**
The separate mV synaptic engine ran all **139,255 neurons**, retaining all
**16,847,997 aggregate edges / 54,492,922 anatomical contacts**, without pruning.
The historical normalized benchmark and its negative result remain unchanged.

| Condition | ORN left / right spikes | ALPN | MBON | Descending | DNa02 left / right |
|---|---:|---:|---:|---:|---:|
| No odor | 0 / 0 | 0 | 0 | 0 | 0 / 0 |
| Left odor | 2,545 / 0 | 23,298 | 3,286 | 2,778 | 29 / 6 |
| Right odor | 0 / 2,466 | 23,305 | 3,175 | 2,799 | 32 / 8 |
| Bilateral odor | 2,545 / 2,466 | 23,503 | 3,276 | 2,883 | 32 / 9 |
| Bilateral, ORNs silenced | 0 / 0 | 0 | 0 | 0 | 0 / 0 |

Only annotated DM1 sensory cells received external input. Each condition started
from the same resting checkpoint: 100 ms rest, 500 ms stimulus, 200 ms recovery;
dt=0.1 ms, 8,000 steps per condition. NumPy PCG64 seed 20260913 draws are matched
across conditions. Input uses the discrete PoissonGroup approximation p=rate*dt,
150 Hz nominal rate and 68.75 mV jumps. Input events can be cleared by same-tick
reset, so injected events and observed ORN spikes are intentionally distinct.

## What the results establish

- Five actual-GPU conformance cases executed: **zero skipped, failed or errored**.
  These cover CPU parity, delays, inhibition, equal decay constants, stream
  ownership, cross-backend checkpoints, silencing and overflow rollback.
- The complete graph passed **200 active CPU/GPU ticks** with 201 sensory input
  events. At five checkpoints, every neuron's voltage, synaptic state and rate
  had maximum absolute difference **0**. Spike/history/refractory timestamps
  were exactly equal. This covers the measured trajectory, not all possible inputs.
- Full-graph JSON state plus input RNG replayed **20 future ticks exactly**.
- The same hardware environment passed **204 software tests, zero skips**.
- Sensory-only stimulation now drives ALPN, MBON and descending spikes. The
  no-odor and input-silenced controls remain silent in these recorded populations.

## What remains unresolved

DNa02 activity is left-biased for both left and right stimulation. The observed
pair therefore does not simply encode stimulus side. Counts in aggregate output
populations are not proof of correct body commands, attraction or reliable
navigation. This protocol has one seed, nominal parameters and an 800 ms horizon;
it does not establish robust physiology across doses, seeds or behavioral states.

The graph uses assumed transmitter signs, 0.275 mV/contact weights and no incoming
normalization. These fitted research parameters do not measure every synapse.
The [independent Brian2 validation](synaptic-brian2-reference.json) establishes
the implemented equations/schedule, not biological validity of this particular
graph/parameter combination. The stimulus is one Or42b/DM1 channel, not a complete
ethyl-acetate response, banana mixture or visual-recognition model.

Each 800 ms condition took 102.1–104.1 seconds, including observations. These
are experiment wall times, not an isolated kernel benchmark or real-time claim.
The raw report's `mapping.stimulus.current_units` describes the historical
dimensionless adapter bundled with the annotation map; it is not used here.
The actual mV input units, dynamics and weights are defined in `protocol`,
`config` and `model.weight_units`.

## Evidence and reproduction

- [Raw five-condition report](flywire-synaptic-odor-a16.json)
- [Five-case actual CUDA gate](synaptic-cuda-a16.json)
- [Software test XML](synaptic-gpu-pytest.xml)
- [64 tested Python file SHA-256 values](synaptic-gpu-source-sha256.txt)
- [Reproduction commands and physical assumptions](../synaptic-dynamics.md)
- Tested source: `592738a3a963961e81a6cc948d0b5d3dfca3471a`.
- Graph arrays SHA-256:
  `f6dc0d6348cc7db3be7514323a57142549327e679f9aee709b2343351685f5b5`.
- Retrieved evidence archive SHA-256:
  `cfeaabc6fb28b90f76b2e4cd0ac013c7cf3a87d44d1e813141d5474e181d9a47`.
- NVIDIA A16-8Q, 8,394,047,488 bytes device memory; CuPy 14.2.0;
  CUDA driver API 12040 / runtime API 12090; Python 3.12.3, NumPy 2.5.3.
- Started `2026-09-13T02:51:35Z`; finished `2026-09-13T03:01:55Z`.

The archive checksum and all 64 recorded source file hashes were verified after
download. The source graph files also passed the pinned official checksums.
