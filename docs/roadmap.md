# Roadmap

## Version 0.5.0a1

- Optional CUDA backend passed actual NVIDIA A16 conformance checks. CPU remains
  the default and does not require CuPy. See [CUDA evidence](cuda.md).
- Full FlyWire v783 numerical benchmark: all 139,255 neurons and source edges,
  CPU/GPU comparison, checkpoint replay and stability measurement. Not real time.
- Separate synaptic mV engine matches an independent Brian2 oracle and actual
  GPU tests. Sensory-only input reaches downstream groups; biological behavior
  is not established. See [synaptic dynamics](synaptic-dynamics.md).
- Full-brain GPU voxel recording with an input-silenced control and paired
  brain/world/input-RNG replay. The untrained readout moves but does not reach food.
- [Odor gain pilot protocol](odor-calibration-pilot.md) separates transient side
  responses from persistent activity before further behavioral calibration.

The older v0.4.0a4 release is CPU-only. Version 0.5 integrates the verified CUDA
engines; see the [release checklist](releasing.md) for distribution checks. Raw full-brain
loading remains a research-script workflow, not a `FlyBrain.load()` catalog entry.

## Available in 0.4.0a4

- Actual Godot 4 CPU scene, offline toy and optional real MaleCNS model.
- Reusable ExternalController with acknowledged controls and complete checkpoints.
- Engine feedback parity, new-process continuation and eight fault cases in CI.
- Native macOS run/pause, save/restore and disconnect recovery verified.
- [Godot setup and contract](../examples/godot/README.md).

## Available in 0.4.0a3

- Live JavaScript CPU game sandbox using toy or the pinned real circuit.
- Scene edits, input/output gains, silencing, whole-session checkpoints and verified replay.
- 300 Python feedback frames check the reference game adapter against Session/DodgeArena.
- [Live adapter and format guide](live-sandbox.md). Godot follows in 0.4.0a4.

## Available in 0.4.0a2

- Installed `flybrain` command: doctor, demo list/run, editable project generation.
- Editable input/output recipes, pinned model identity, source and attribution export.
- [Fresh X/source research and priorities](research-2026-09-10.zh-CN.md).

## Available in 0.4.0a1

- Composable Python sessions with current encoders, rate readouts and JSON environments.
- Exact fractional clock allocation and complete session checkpoints/feedback replay.
- Deterministic dodge and tone templates, recorded viewers, and resumable WAV export.
- [Integration guide](demo-kits.md) and a gallery with a silenced-output comparison.

Still open: additional engine adapters, streaming audio and learned readouts.
CUDA subsequently passed actual-device validation; see the current-source section above.

## Available in 0.3.0a1

- JavaScript float64 CPU runtime with toy/real cross-language trace tests.
- Interactive circuit lab: explicit model download, stimuli, silencing and inspection.
- Standalone editable HTML export and browser command recordings replayed in Python.
- [Browser guide](browser-lab.md) with exact semantics and current limitations.

Later versions added Godot and audio demo kits; current source includes hardware-tested
CUDA. Browser WASM remains a separate, unimplemented backend.

## Available in 0.2.0a1

- Reproducible, opt-in MaleCNS escape subgraph with a model card and CPU measurements.
- Exact source-ID/annotation selection, direct currents and selected observations.
- Arbitrary named rate readouts, reversible spike silencing and schema-2 checkpoints.
- The previous alpha API and schema-1 checkpoint reader remain supported.

The circuit lab shipped in 0.3; basic game/audio sessions, the live game and Godot
adapter shipped in 0.4. Streaming audio and trained rhythm templates remain open.
CUDA has since passed hardware checks. See [RFC 0001](rfcs/0001-open-runtime-and-demo-kits.zh-CN.md) for the
proposed architecture; proposal-only APIs are not current API documentation.


## Available in 0.1.0a1

- Python src-layout package, NumPy CPU LIF simulation and offline toy circuit.
- Sensory/motor interfaces, fixed-step loop, self-contained JSON checkpoints.
- Official MaleCNS/FlyWire source catalog and on-demand verified/cacheable downloads.
- Tests, packaging, CI, public project documentation and contribution templates.
- Draft TypeScript contract; no NPM/WASM runtime yet.

## Completed by 0.2: first real-data slice

- Pin source SHA256 checksums where missing.
- Add tested importers and one reviewed 100–1,000-neuron subgraph model.
- Publish a model card with physiological assumptions and response measurements.
- Benchmark CPU runtime and memory before choosing acceleration work.

## Then: integrations and scale

- Extend Godot, add Unity and richer game examples.
- Compact sparse-array model/checkpoint format for large graphs.
- WASM reference implementation and TypeScript package.
- Improve the optional CUDA backend with separately benchmarked batching.
- Explore learning/plasticity separately from the fixed-connectome MVP.

Open issues and propose focused milestones; these are directions, not release-date promises.
