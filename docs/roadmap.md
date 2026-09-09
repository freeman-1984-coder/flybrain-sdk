# Roadmap

## Available in 0.2.0a1

- Reproducible, opt-in MaleCNS escape subgraph with a model card and CPU measurements.
- Exact source-ID/annotation selection, direct currents and selected observations.
- Arbitrary named rate readouts, reversible spike silencing and schema-2 checkpoints.
- The previous alpha API and schema-1 checkpoint reader remain supported.

Next: an interactive circuit explorer and game sandbox sharing these interfaces,
exportable demo kits, optional CUDA tested on hardware, then audio and rhythm
examples. See [RFC 0001](rfcs/0001-open-runtime-and-demo-kits.zh-CN.md) for the
proposed architecture; proposal-only APIs are not current API documentation.


## Available in 0.1.0a1

- Python src-layout package, NumPy CPU LIF simulation and offline toy circuit.
- Sensory/motor interfaces, fixed-step loop, self-contained JSON checkpoints.
- Official MaleCNS/FlyWire source catalog and on-demand verified/cacheable downloads.
- Tests, packaging, CI, public project documentation and contribution templates.
- Draft TypeScript contract; no NPM/WASM runtime yet.

## Next: real-data slice

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
