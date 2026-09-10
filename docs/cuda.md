# Experimental CUDA backend — hardware validation pending

This development branch implements a CuPy/CUDA reference backend. **It has not yet passed an actual NVIDIA-device run.** The published v0.3 alpha remains CPU-first. CPU tests and successful import of a cloud runner do not demonstrate GPU conformance or speedup. Do not advertise production CUDA support from this branch yet.

## Optional installation

The basic `pip install -e .` path still installs only NumPy. Importing `flybrain` or creating a CPU brain does not import CuPy or initialize CUDA. On an NVIDIA host with a suitable driver, use a separate Python 3.12 environment:

```sh
pip install -e '.[dev]' 'cupy-cuda12x[ctk]==14.2.0'
python scripts/validate_cuda.py --output cuda-validation.json
```

CuPy 14 requires Python 3.10+; the CPU SDK still supports Python 3.9. Match the CuPy wheel to the host driver/toolkit and install only one CuPy distribution. The `ctk` extra installs compatible CUDA components; it cannot supply the host NVIDIA driver. See [CuPy installation](https://docs.cupy.dev/en/stable/install.html).

After hardware validation:

```python
from flybrain import FlyBrain, available_backends

print(available_backends())  # dependency/device probe, not a conformance certificate
brain = FlyBrain.load("male-cns-escape-v1", backend="cuda", download=True)
brain.stimulate("looming_left", duration_ms=100)
brain.advance(duration_ms=100)
print(brain.action().to_dict())
brain.save("gpu.checkpoint.json")
cpu = FlyBrain.restore("gpu.checkpoint.json", backend="cpu")
```

Unavailable dependencies, devices or kernel compilation raise `BackendUnavailableError`. There is no silent CPU fallback. Python-only `backend="cuda"` does not add CUDA to the browser runtime; WASM is still unimplemented.

## What the implementation does

- One CUDA thread owns one postsynaptic row. Stable ordering keeps original parallel-edge accumulation order. It uses float64 and disables fused multiply-add to preserve separate reference operations; conformance must still be measured.
- The graph, voltage, spikes, refractory counters, filtered rates and silencing mask reside on one GPU. Each instance owns a stream so external CuPy stream contexts do not reorder its operations.
- One kernel advances one tick using previous-tick spikes. It writes separate output buffers. A scalar error flag is checked before committing the tick, preserving state on nonfinite voltage errors.
- Selected observations transfer only requested cells/fields. Full observations and checkpoints explicitly transfer full state. Restore uses the existing CPU validation rules, with temporary CPU graph/state allocation.
- Current vectors are currently prepared on the host and uploaded every tick. Each tick also synchronizes an error flag. This first implementation prioritizes compatibility; it is **not** the future device-resident batched schedule and may be slower than CPU on small models.

The [CuPy RawKernel API](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html) provides runtime compilation. Larger graphs, batching, mixed precision and trainable dynamics need separate work and evidence.

## The hardware gate

`validate_cuda.py` requires an actual CuPy-visible NVIDIA device, runs the hardware suite with `FLYBRAIN_REQUIRE_CUDA=1`, rejects skipped tests, then times toy and real circuits on CPU and GPU. Without a device it writes a failed report and exits nonzero. Ordinary CPU CI skips those hardware tests explicitly.

The suite compares every cell over 180 ticks for toy, real and signed/parallel-edge synthetic graphs at two timesteps (1,080 comparison ticks), plus cross-device checkpoints with pending stimuli, custom readout, interventions, selected observations, invalid-state handling and stream isolation. Spikes/clock must match exactly; voltage/rates use absolute tolerance `1e-10`. A report includes environment versions, device, source hashes, passed test counts, configuration, warmup and per-run timings. This covers the reported hardware and cases only.

Acceptance before merging/advertising CUDA:

1. No skipped or failed hardware tests on at least one NVIDIA GPU.
2. Checkpoint continuation and direct/named current semantics match CPU.
3. Honest CPU/GPU wall times with compilation/loading separated and synchronization included.
4. Store the report with the exact tested source identity and inspect any failures.
5. Re-run normal CPU packaging/CI; missing CUDA never breaks CPU installation.

## Optional rented T4 run

A prepared runner is available at `scripts/validate_cuda_modal.py`. Its configuration is one T4, one CPU core, 4 GiB RAM, at most one container, a 600-second function timeout, no retries, and scale-to-zero. There is no persistent volume, web endpoint or schedule. Only source code, the hardware test, validator and the small model are uploaded; no credentials or local home directory are included.

After the account and spending authorization are resolved, a maintainer can run:

```sh
# Separate Python 3.10+ environment on the controlling computer.
pip install 'modal==1.5.5'
modal setup
modal run scripts/validate_cuda_modal.py --output cuda-validation.json
```

This command starts remote compute and can incur charges. As checked on 2026-09-10, [Modal's published pricing](https://modal.com/pricing) lists T4 at $0.000164/second, CPU at $0.0000131/core/second, and RAM at $0.00000222/GiB/second. Ten minutes at the configured allocations is about $0.112 in execution charges, before build/startup/other charges. This estimate is not an enforced account-wide dollar cap. Review current pricing and available account credits before running; no credits are assumed. Use a $1 trial budget and verify the ephemeral app stops in the provider console when the run finishes or is canceled.

The runner definition was checked against the local Modal SDK without invoking any remote function. It remains untested remotely. [Modal GPU documentation](https://modal.com/docs/guide/gpu) describes device selection. Recheck container termination and actual billed usage before considering the rental step finished.
