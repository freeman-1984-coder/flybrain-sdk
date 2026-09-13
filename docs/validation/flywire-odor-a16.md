# Full FlyWire odor probe — input works, downstream spiking absent

**Negative functional result, 2026-09-13 UTC.** All 139,255 neurons and 16,847,997
source rows ran on NVIDIA A16-8Q. This is not a subgraph. CUDA's 12 hardware cases
passed; the full suite passed **181 tests, zero skips**. The five-condition
experiment completed, but this preset did **not** demonstrate odor transmission
through the olfactory projection neurons or motor output.

| Condition | Input ORN spikes | ALPN spikes | MBON spikes | Descending spikes |
|---|---:|---:|---:|---:|
| no_odor | 0 | 0 | 0 | 0 |
| left_odor | 3,500 | 0 | 0 | 0 |
| right_odor | 3,300 | 0 | 0 | 0 |
| bilateral_odor | 6,800 | 0 | 0 | 0 |
| bilateral_odor_ORNs_silenced | 0 | 0 | 0 | 0 |

Each condition: 100 ms rest + 500 ms stimulus + 200 ms recovery, from the same
resting checkpoint. No global background current. Only annotated ORN_DM1 cells
receive external input. Left/right cell counts are 35/33. A single responsive
Or42b channel is approximated; no complete banana mixture is claimed.

The maximum observed ALPN membrane potential was 0.2516453 (threshold 1).
Maximum descending membrane potential was 0.00129747. Zero spikes therefore
means no threshold crossing; it does not mean absolutely no subthreshold signal.
The GPU executed 4,000 total ticks. Per-condition wall times were 9.62–9.69 s,
including selected-neuron observations; this was not a throughput benchmark.

## Why this preset cannot support the requested behavior

The historical numerical benchmark normalized every neuron's incoming absolute
weight sum to 2.5. With its 1 ms step, 10 ms membrane time constant and two-tick
refractory hold, an input cell can spike at most once every three ticks. From
rest and with zero external postsynaptic current, the largest possible voltage
is bounded by the all-excitatory, perfectly synchronized case:

```
leak = exp(-1/10)
V_max <= 2.5 * (1 - leak) / (1 - leak**3) = 0.9179135 < threshold 1
```

Inhibition only lowers this upper bound. Thus increasing odor intensity alone
cannot fix transmission in this preset. A separate CPU two-cell diagnostic with
maximal 3-tick firing spacing reached 0.9179135027773137 and zero downstream
spikes over 2,000 ticks, matching the independent bound. That small diagnostic
explains the parameter issue; it is not a substitute for the full-brain GPU run.
The original full-brain benchmark remains valid for numerical conformance,
checkpointing and timing under its artificial global input, **not physiology**.

## Next implementation gate

Use an explicitly separate dynamics configuration based on the published
[Shiu et al. model](https://www.nature.com/articles/s41586-024-07763-9), rather than
silently changing the historical benchmark. Its membrane and synaptic states,
synaptic decay, transmission delay, refractory behavior and Poisson sensory
stimulation must be checked against an independent reference before making
behavior claims. Preserve the full graph, source/weight provenance and the same
negative controls. Only after propagation is demonstrated should a voxel-world
body consume neural outputs. Any learned or handcrafted readout must be labeled
as such; target coordinates must not bypass the neural path.

## Evidence and reproduction

- [Protocol, annotation download and commands](../olfactory-experiment.md).
- [Raw report with all 400 sampled frames](flywire-odor-a16.json).
- [Actual GPU test suite XML](flywire-odor-pytest.xml).
- [Tested Python source SHA-256 values](flywire-odor-source-sha256.txt).
- [Analytical-bound CPU diagnostic](benchmark-voltage-bound.json).
- Tested source: `357e6546cad18b45b7033c9725c140d0b8901dec`. The subsequent
  `930b5f9` commit only formats an annotation metadata string.
- Downloaded evidence archive SHA-256:
  `e41c10b70bf46a51569afb3e404213f5ee01924adcc8e2d324380d7dd82881aa`.
  All source hashes were matched to the published tested commit.
- CUDA backend/kernel unchanged from the earlier numerical benchmark. CuPy
  14.2.0, CUDA runtime API 12090, driver API 12040, 8,394,047,488 device bytes.

The temporary GPU instance was terminated after results were retrieved and
verified. No persistent GPU endpoint or live browser inference is provided by
this report.
