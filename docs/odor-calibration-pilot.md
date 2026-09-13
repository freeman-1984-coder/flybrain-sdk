# Full-brain odor gain pilot: protocol before results

**Executed on A16 on 2026-09-13:** [all eight results and hardware evidence](validation/odor-gain-pilot-a16.md).
The protocol below was published in commit
`433c77f6c6da3b25c10996847b788d0bca871b4d` before execution and was not changed
in response to the results. No gain or behavioral default was selected.

This protocol tests whether changing recurrent contact strength alters transient
side responses and post-stimulus spiking in the experimental synaptic engine.
It does **not** test banana identification, flight, learning or food seeking.
There is no environment, fitted action decoder or target position in this pilot.
The source commit should be published before the GPU run; publish every result,
including non-responsive, persistently active or failed conditions.

## Why this experiment

The [previous full-brain odor experiment](validation/flywire-synaptic-odor-a16.md)
showed a cumulative left DNa02 bias under either stimulus side. Its time series
also contains an early right DNa02 response to right input. The
[post-stimulus counts](validation/synaptic-post-stimulus.json) show actual new
downstream spikes during the observed recovery window. Cumulative counts alone
hide timing; an exponentially smoothed rate alone cannot prove continued firing.

## Fixed design

- All 139,255 FlyWire v783 neurons and all 16,847,997 source aggregate edges.
  The official source files and annotation mapping must pass their pinned checksums.
  No edge pruning, incoming normalization, added background drive or rescue reflex.
- Four contact strengths: **0.05, 0.10, 0.175, 0.275 mV/contact**. Excitatory and
  inhibitory contacts scale together. These are exploratory values, not measured
  biological parameters. All other [synaptic settings](synaptic-dynamics.md) stay fixed.
- Each strength uses both **left → right** and **right → left** input order,
  making eight runs. JSON state is restored only between runs.
- Each run: 50 ms silent baseline → 150 ms first odor → 200 ms recovery →
  150 ms other-side odor → 250 ms recovery. No state reset between these phases.
- Fixed `dt = 0.1 ms`, seed `20260914`, NumPy PCG64. Both sides' random draws are
  consumed at every tick, even if their input is disabled, preserving matched
  per-neuron input sequences across gains and orders. Left/right populations have
  different sizes and independently sampled events; they are not identical copies.
- Only the pinned 35 left and 33 right DM1 ORNs receive input. Bernoulli probability
  is `150 Hz × dt / 1000`; each event supplies a 68.75 mV input jump. This jump
  **does not scale with recurrent contact strength**. The physiological accuracy
  of this single-channel odor adapter has not been established.

## Prespecified measurements

Each phase records actual per-neuron spike totals, first-50-ms counts, group totals,
and group spike counts in its last 100 ms (the initial rest is only 50 ms).
An additional 10 ms trace records smoothed mean rates for visualization.

For each stimulus, report DNa02 **ipsilateral minus contralateral** firing rate
separately for the first 50 ms and the remaining 100 ms, plus cumulative
right-minus-left spike count. The pinned mapping contains one DNa02 cell per side.
Positive ipsilateral difference is descriptive; it is not an established action
label. Compare the same side when presented first versus second.

For each recovery, report **new spikes** in its final 100 ms in DM1 ORNs, ALPNs,
MBONs, descending neurons and DNa02. A quiet finite window does not establish
long-term stability; persistent activity alone does not establish memory.

Report all four strengths and both orders without choosing a winner automatically.
A candidate from this one-seed pilot requires independently specified multiple-seed
and held-out stimulus validation before changing a public behavioral preset.
Changing the gain must not be described as restoring validated biological behavior.

## Reproduce on an actual GPU

Install the repository with dataset support, pytest and a compatible CuPy CUDA
installation, and download the same pinned data files used by the
[full-brain odor experiment](validation/flywire-synaptic-odor-a16.md).

```sh
python scripts/validate_synaptic_cuda.py --output results/hardware-gate.json
python scripts/calibrate_fullbrain_odor.py \
  --data-dir data --annotations data/annotations.tsv \
  --output results/odor-gain-pilot.json
FLYBRAIN_REQUIRE_CUDA=1 python -m pytest -q
```

The hardware gate must report five executed passing GPU cases and zero skips.
There is no CPU fallback in this experiment. Retain the full output JSON, hardware
report, test results, source hashes and dependency versions. Incomplete JSON is
marked `running` or `failed`; only all eight finished runs may be `completed`.
The ordinary CPU SDK remains usable without CUDA.
