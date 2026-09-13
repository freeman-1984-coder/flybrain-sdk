# flybrain-sdk 0.5.0a1

**No CUDA required.** This alpha integrates optional, actual-device-tested CUDA
with the CPU SDK, real MaleCNS model, browser demos and Godot adapter.
Install from the GitHub release wheel or the `v0.5.0a1` tag; there is no PyPI/NPM
publication. CUDA additionally needs a compatible NVIDIA driver and CuPy.

## Included

- CPU-default `FlyBrain` API with on-demand, checksum-verified real model downloads,
  direct cell input/observation, custom readouts, silencing and complete checkpoints.
- Optional CUDA backend with 12 passing actual A16 hardware cases. The 313-cell
  workload was slower on GPU; small models should generally stay on CPU.
- Experimental synaptic mV CPU/CUDA engines, checked against Brian2 and five actual
  GPU cases. These explicitly require mV weights; existing dimensionless models
  and checkpoints keep their meaning.
- Full FlyWire numerical and sensory experiments, a GPU voxel recording, and an
  eight-run odor gain pilot. All source neurons and aggregate edges are retained.
  Original records, source hashes, assumptions and negative results are public.
- Editable demo generation, verified session replay and the Godot CPU/CUDA bridge.
  Generated project requirements pin this public Git tag.
- Source distributions include the JSON/XML/CSV/text validation evidence referenced
  by the documentation, in addition to reports, examples and model recipes.

## Limits

The packaged ready real model is still the 313-neuron MaleCNS subgraph with assumed
LIF dynamics. Full FlyWire loading is an explicit research-script/data-download
workflow, not a ready catalog model or a `FlyBrain.load()` preset. Its complete
network was run on GPU, but biological physiology, reliable foraging and real-time
performance have not been established. The voxel and reflex web pages replay
recorded experiments; the circuit lab and game sandbox execute the JS CPU runtime.
WASM and biological synaptic plasticity remain unimplemented.

The odor pilot changed no behavior default. Its one-seed comparison could not
establish a responsive navigation preset by scaling all contact weights alone.
A temporary early side response must not be described as banana identification.

## Evidence

- [Actual CUDA validation](cuda.md) and [full FlyWire benchmark](fullbrain-validation.md).
- [Synaptic dynamics and independent reference](synaptic-dynamics.md).
- [Full-brain voxel experiment](validation/flywire-voxel-a16.md).
- [All eight odor gain conditions](validation/odor-gain-pilot-a16.md).
- [Release procedure](releasing.md), including clean-wheel installation,
  source-archive completeness and post-tag generated-project installation.
