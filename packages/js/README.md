# TypeScript / WASM contract (design only)

This directory deliberately has `private: true`. It is not an NPM runtime and
cannot yet simulate a brain. Python is the working reference implementation.

```sh
npm install
npm run typecheck
```

Future implementation: port the fixed-step LIF core to Rust/WASM, reuse the model
and checkpoint schema, and add a small TypeScript wrapper. Run Python-generated
spike, rate, refractory and checkpoint fixtures against WASM before publishing.
Use float64 first; require explicit tolerances for cross-backend comparisons.
Transport neuron IDs as strings, since FlyWire IDs can exceed JavaScript's safe
integer range. Keep simulation off the main UI thread for large models.

The factory has async `load`/`restore` for asset fetching and WASM initialization;
`stimulate`, `step`, `action` and `save` remain synchronous within a worker.
Python uses snake_case and milliseconds; TypeScript uses camelCase and the same
units. Convert checkpoint keys at the wrapper boundary, never fork the format.
