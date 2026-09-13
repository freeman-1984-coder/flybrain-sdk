# Full FlyWire odor gain pilot — NVIDIA A16

**All eight full-graph CUDA runs completed. This pilot did not identify a preset
that establishes reliable odor-directed navigation. No behavioral default changed.**
[Interactive comparison](https://freeman-1984-coder.github.io/flybrain-sdk/odor-calibration.html)
· [Unmodified raw report](../../site/experiments/odor-gain-pilot.json)
· [Prespecified protocol](../odor-calibration-pilot.md).

## Execution and provenance

- Public pre-run commit: `433c77f6c6da3b25c10996847b788d0bca871b4d`.
- Ran 2026-09-13 03:49:16–04:07:47 UTC, including graph construction between gains.
- Four strengths × both stimulus orders; 800 ms per run, 0.1 ms ticks, seed 20260914.
- All 139,255 neurons, 16,847,997 source rows, 54,492,922 anatomical contacts retained
  at every strength. No additional pruning or incoming normalization. Source file
  hashes and a separate graph-array hash for each strength are in the raw report.
- NVIDIA A16-8Q, 8 GB virtual GPU; Python 3.12.3, CuPy 14.2.0,
  driver API 12040 and CUDA runtime 12090. Full dependencies are
  [recorded here](odor-gain-requirements.txt).
- [Hardware gate](odor-gain-cuda-a16.json): five GPU cases passed, zero skipped.
- [Full hardware-host test suite](odor-gain-pytest.xml): 217 passed, zero skipped.
- [69 Python source-file hashes](odor-gain-source-sha256.txt) verified against the
  pre-run commit after retrieval. Group totals and onset metrics were independently
  recomputed from the per-neuron counts; all eight planned conditions were checked.
- [During-run GPU observation](odor-gain-gpu-active.csv): 87% GPU utilization and
  289 MiB GPU memory at that one instant. It is not a peak-memory measurement.
- Individual 800 ms runs took 101.9–103.7 seconds, including observations. This is
  not real-time simulation and is not the older dimensionless benchmark's speedup.

The retrieved evidence archive SHA256 was
`5028ab681c2f634ceeafb2de81c78cd07f501f29c1b7f75504ca1a69c188f081`.
The unmodified published raw report SHA256 is
`e72b30c19b76efb3a7f92227519a65ee68629690f216b35dfd6fdbe7170c011f`.
The archive contains setup logs as well as the individually published evidence;
its digest is a transfer check, not a URL to an uploaded archive.

## Prespecified results

Onset means the first 50 ms of each 150 ms stimulus; later means its remaining
100 ms. DNa02 values are **ipsilateral minus contralateral Hz**, with one cell on
each side. Recovery values are **new descending-neuron spikes in the final 100 ms**
of each recovery, not residual smoothed rates. Raw data retains all other groups.

| mV/contact | Order | Onset 1 Hz | Onset 2 Hz | Later 1 Hz | Later 2 Hz | Recovery 1 spikes | Recovery 2 spikes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | left → right | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| 0.05 | right → left | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| 0.1 | left → right | 0.0 | 0.0 | 0.0 | 0.0 | 17 | 17 |
| 0.1 | right → left | 0.0 | 0.0 | 0.0 | 0.0 | 16 | 17 |
| 0.175 | left → right | 0.0 | -20.0 | 10.0 | -10.0 | 152 | 148 |
| 0.175 | right → left | 0.0 | 0.0 | -10.0 | 0.0 | 151 | 154 |
| 0.275 | left → right | 60.0 | -40.0 | 60.0 | -30.0 | 422 | 430 |
| 0.275 | right → left | 20.0 | 20.0 | -40.0 | 40.0 | 388 | 436 |

## Interpretation and limits

At 0.05 mV/contact, the selected sensory neurons and a few ALPNs fired, but no
MBONs or descending neurons did. Both measured recovery tails were quiet.
At 0.10, activity reached some downstream cells but neither DNa02 fired, and
new downstream spikes continued after input ended. At 0.175 and 0.275, selected
DNa02 cells fired, with no consistent ipsilateral response across both orders
and both time windows. Downstream spiking persisted into every measured recovery
tail at these strengths.

The 0.275 run with right input first has a positive early right response (+20 Hz)
and a later left bias (-40 Hz ipsilateral difference). Right input presented second
in the other run instead has a negative early difference (-40 Hz). This describes
an observed order/timing difference; **it does not isolate a causal memory effect**.
The first and second phases use different draws from the same absolute-time random
stream as well as different preceding neural states. Multiple seeds and matched
within-stimulus event patterns would be required to separate those effects.

The pilot preserves fixed 68.75 mV sensory jumps while changing all recurrent
contact weights together. These are assumed homogeneous LIF dynamics and a
single-channel DM1 food-odor approximation, not measured banana chemistry,
receptor dose-response curves or calibrated cell-specific physiology. It provides
no learned decoder, flight model or closed-loop foraging validation. A positive
DNa02 difference is a measurement, not a certified motor command.

Lowering all connection weights alone did not yield a demonstrated responsive
navigation preset in these four values and this seed. Do not promote a selected
gain or reverse a body controller based on these data. Follow-up hypotheses need
independent stimulus/seed validation and declared input/readout assumptions.
The experiment code and reproduction commands remain in the prespecified protocol.
