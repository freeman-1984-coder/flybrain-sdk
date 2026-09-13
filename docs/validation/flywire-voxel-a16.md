# Full-brain CUDA voxel experiment: movement, no successful foraging

**Recorded on NVIDIA A16-8Q, 2026-09-13.** All 139,255 FlyWire v783 neurons and
16,847,997 aggregate edges were simulated on the GPU in both runs. A fixed,
untrained readout converted DNa02 rates to a kinematic ground body's speed and
turn. There was no target-bearing controller or direct motor stimulation.

| Recorded condition | Simulated time | Path length | Initial / final food distance | Food contact |
|---|---:|---:|---:|---|
| Neural control | 2 s | 3.840 | 2.778 / 4.380 | No |
| ORN inputs silenced | 2 s | 0 | 2.778 / 2.778 | No |

Distances use arbitrary game units. The active run initially approached to
2.266 units, then curved left and moved away. It did not reach the 0.6-unit
contact radius. Neither run encountered a block; collision mechanics are tested
separately in the CPU environment tests, not established by these trajectories.
This is evidence of an implemented neural-to-body closed loop, **not successful
odor-directed navigation, training or biological locomotion**.

## Exact control boundary

- World observations contain only left and right local antenna concentrations.
- Input rates are `180*c/(0.2+c)` Hz; per-tick Bernoulli events generate 68.75 mV
  jumps on the 35 left / 33 right annotated DM1 input cells.
- All graph cells update at 0.1 ms. After 200 neural steps, the 20 ms body frame
  consumes DNa02 left/right exponentially smoothed rates, with a 50 ms time constant.
- Speed = `clip(0.03*(left+right), 0, 3)` game units/s.
- Turn = `2*tanh((right-left)/40)` radians/s. Positive turns toward world +z.
- Food coordinates reach the odor field, rendering and scoring only, never the
  motor readout. Input silencing operates on the brain, not on renderer animation.

The speed mapping is an engineering hypothesis with no biological validation.
The ground body, Gaussian field, gains, initial scene and seed are explicit in
the recording. The model covers one food-odor channel and no vision. DNa02's
side bias and post-stimulus persistence in the [sensory controls](flywire-synaptic-odor-a16.md)
remain reasons to investigate calibration and a more informative readout.

## Validation and reproduction

- **207 software tests passed on the GPU host, zero skips**:
  [test XML](voxel-gpu-pytest.xml).
- At frame 20 in each run, brain state (including all delayed events), world and
  input RNG were serialized through JSON. The next complete 20 ms frame was
  replayed and compared: exact full-brain state, world record and RNG equality.
- All 100 frames per run are preserved. Wall times: active 243.654 s; silenced
  248.907 s. These include observations and the extra replay checks; not real time.
- Tested source: `e89b8feaf7f2381eeaad625de0cc0897bb6ac773`.
- [Tested Python source hashes](voxel-gpu-source-sha256.txt) and the hashes in the
  raw recording were verified after download.
- Evidence archive SHA-256:
  `4e3dc17aa64c878c3447320ad9098f94a93a79c05655fb710f21f8b64af64455`.
- Started `2026-09-13T03:04:58Z`; finished `2026-09-13T03:14:25Z`.
- NVIDIA A16-8Q / CuPy 14.2.0 / CUDA driver API 12040 / runtime API 12090.

```bash
python scripts/run_fullbrain_voxel.py --data-dir data \
  --annotations data/annotations.tsv --output fullbrain-voxel.json
```

Use the source files and dependencies from the [full-brain odor protocol](../olfactory-experiment.md).
[The browser page](https://freeman-1984-coder.github.io/flybrain-sdk/foraging.html)
replays the [complete JSON recording](../../site/experiments/fullbrain-voxel.json).
It supports selecting the control, scrubbing time and rotating the view; it does
not perform live GPU inference. It refuses to substitute generated trajectories
when the recording is absent or incomplete.

The temporary GPU instance was destroyed after evidence retrieval and checksum
verification. The provider instance list was confirmed empty; the task SSH key
was deleted. The shared temporary rental for these experiments cost $0.24.
