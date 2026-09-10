# Roadmap

## Available in 0.4.0a2

- Installed `flybrain` command: doctor, demo list/run, editable project generation.
- Editable input/output recipes, pinned model identity, source and attribution export.
- [Fresh X/source research and priorities](research-2026-09-10.zh-CN.md).

## Available in 0.4.0a1

- Composable Python sessions with current encoders, rate readouts and JSON environments.
- Exact fractional clock allocation and complete session checkpoints/feedback replay.
- Deterministic dodge and tone templates, recorded viewers, and resumable WAV export.
- [Integration guide](demo-kits.md) and a gallery with a silenced-output comparison.

Still open: live game integrations, engine adapters, streaming audio and learned readouts.
CUDA implementation is in a separate draft PR pending actual-device validation.

## Available in 0.3.0a1

- JavaScript float64 CPU runtime with toy/real cross-language trace tests.
- Interactive circuit lab: explicit model download, stimuli, silencing and inspection.
- Standalone editable HTML export and browser command recordings replayed in Python.
- [Browser guide](browser-lab.md) with exact semantics and current limitations.

Next: optional CUDA tested on hardware, engine/game adapters, then audio and
rhythm demo kits sharing the open core. Browser WASM remains a separate backend.

## Available in 0.2.0a1

- Reproducible, opt-in MaleCNS escape subgraph with a model card and CPU measurements.
- Exact source-ID/annotation selection, direct currents and selected observations.
- Arbitrary named rate readouts, reversible spike silencing and schema-2 checkpoints.
- The previous alpha API and schema-1 checkpoint reader remain supported.

The circuit lab shipped in 0.3 and basic game/audio sessions in 0.4. Live game,
streaming audio, trained rhythm templates and validated CUDA remain open. See [RFC 0001](rfcs/0001-open-runtime-and-demo-kits.zh-CN.md) for the
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

- Godot/Unity adapters and richer game examples.
- Compact sparse-array model/checkpoint format for large graphs.
- WASM reference implementation and TypeScript package.
- Optional CUDA backend, only after cross-backend conformance tests.
- Explore learning/plasticity separately from the fixed-connectome MVP.

Open issues and propose focused milestones; these are directions, not release-date promises.
