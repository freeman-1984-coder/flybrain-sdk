# Python API reference

Version: 0.2.0a1. CPU is implemented; CUDA and WASM remain unavailable.

## Open circuit API

`brain.neurons.select(ids=None, **attributes)` returns immutable source-ID tuples
in model order, intersecting exact attribute matches. Available attribute names
are in `brain.neurons.attributes`. The real escape model supplies `cell_type`,
`side`, `status` and `consensus_nt`. Missing fields and empty selections are
errors; the SDK does not guess anatomy from IDs. Direct methods also accept lists
of known string IDs, including large numeric IDs.

`brain.drive.current(ids, *, amplitude, duration_ms, units="normalized")` queues
a signed current for the next tick. Unlike named stimuli, amplitude is not
multiplied by `input_gain` or constrained to [0,1]. It must be finite. Only
normalized units are supported by this dynamics revision. Overlapping direct
and named stimuli add. `brain.drive.clear()` cancels direct currents;
`clear_stimuli()` cancels named stimuli only.

`brain.advance(duration_ms=100)` advances without exporting the full neural state
and returns `Progress(tick, time_ms)`. Both direct-current durations and `advance`
require an integer multiple of dt (within a 1e-9 tick conversion tolerance).
Legacy named-stimulus duration rounding remains unchanged for compatibility.

`brain.observe(ids=None, fields=("voltage", "spikes", "rates_hz"))` returns a
detached `Observation` with IDs, tick, time and selected values. Fields must be
unique supported names. This transfers only selected cells/fields. `spikes`
means events at the final tick, not a count over an `advance` window.

`brain.bind_readout({"volume": ids, "flash": other_ids}, scale_hz=100)` replaces
the output mapping without changing graph connections. `action()` then returns a
`ChannelAction`; use `action["volume"]` or `action.to_dict()`. Each value is
`clip(mean(rates_hz[ids]) / scale_hz, 0, 1)`. This is a configurable rate readout,
not a trained decoder. Binding `{}` permits an experiment with no action outputs.
Invalid replacement mappings leave the previous mapping intact.

`brain.intervene.silence(ids, enabled=True)` suppresses new spikes beginning with
the next tick and holds voltage at reset. Spikes already emitted still propagate;
historical firing rates decay normally. `enabled=False` releases selected cells.
This operation does not remove edges or alter the model fingerprint. The mask is
included in checkpoints.

Examples: [offline custom output](../examples/open_circuit.py),
[real source data](../examples/real_connectome.py),
[model assumptions](../models/male-cns-escape-v1/README.md).

## Load and observe

```python
from flybrain import FlyBrain, LIFConfig, Stimulus

brain = FlyBrain.load("toy", config=LIFConfig(dt_ms=1.0))
brain.stimulate(Stimulus("food", strength=0.8, duration_ms=120))
state = brain.step(20)
print(state.tick, state.time_ms, state.voltage, state.spikes, state.rates_hz)
print(brain.action().walk)
```

`FlyBrain.load(model="toy", *, backend="cpu", config=None, download=False, cache_dir=None)`
accepts the builtin name, a local JSON model/bundle, a validated `Connectome`, or
a ready catalog entry. Downloading requires explicit `download=True`; subsequent
loads verify and reuse the local cache. Recognized raw
catalog IDs raise an explanatory error. `available_backends()` returns a mapping
of recognized names to runtime availability.

`state` is an immutable observation copied out of the backend; neuron positions
correspond to `brain.model.neuron_ids`. Spikes are from the **last tick**, not a
union across `step(steps)`. For every spike event, call `step()` once per tick.
The initial observation is tick/time zero, resting voltage, zero spikes/rates.

## Sensory input

`stimulate(channel, *, strength=1.0, duration_ms=20.0)` returns the same brain for
chaining. Alternatively pass one `Stimulus`; do not mix it with keyword overrides.
Strength must be finite and in [0, 1], duration finite and positive. Duration is
rounded up to `ceil(duration_ms / dt_ms)` ticks. Stimuli begin at the next tick,
and overlapping pulses add their currents, including on shared neurons.
Each neuron in a sensory port receives `strength * input_gain`.

`clear_stimuli()` cancels queued/active inputs but does not reset neural dynamics.
The four toy channel names describe **pre-encoded stimuli**; they do not accept
camera pixels, images, odor chemistry, or raw sensor measurements.

## Time and action

`step(steps=1)` takes a positive integer; it advances simulation time, not wall
clock time. The default timestep is 1 ms. The game loop chooses how many ticks to
run per frame. `action()` is a pure read. The legacy toy/escape readout returns
`MotorAction`, with missing legacy fields zero. Arbitrary model channels and
`bind_readout()` return `ChannelAction`, a mapping with only the configured names.
Both have `to_dict()`; all current readouts are clipped mean-rate intensities.

## Checkpoints

```python
brain.save("brain.checkpoint.json")
continued = FlyBrain.restore("brain.checkpoint.json", backend="cpu")
```

`save()` returns the destination `Path`; its parent directory must exist. It uses
atomic file replacement. `restore()` is a class method returning a **new** brain.
The checkpoint embeds the model and its SHA256 fingerprint, config, dynamics
revision, tick, voltages, previous spikes, refractory counters, filtered firing
rates, pending named stimuli, direct currents, silencing masks, annotations and
readout configuration. New files use checkpoint schema 2; schema 1 remains
readable. No pickle or code execution is used.

Unknown schema/dynamics versions, altered model fingerprints, invalid vector
shapes and nonfinite state are rejected with `CheckpointError`. File access
errors remain ordinary `OSError` subclasses. Unavailable backends raise
`BackendUnavailableError`. Exact continuation is tested on the same CPU runtime;
bit-for-bit reproducibility across different numerical libraries/hardware or
future backends is not promised. A fingerprint checks consistency, not authenticity.

## Custom models

Use `Connectome.from_dict()` or edit a copy of `Connectome.load().to_dict()`.
Schema version 1 contains `name`, string `neuron_ids`, `synapses` (pre ID, post ID,
signed `weight`), `sensory` and `motor` port maps, string-valued `provenance`,
and optional per-ID `annotations` whose attribute keys and values are strings.
IDs are kept as strings, including numeric source IDs. Duplicate neuron IDs,
unknown edge endpoints and nonfinite weights are invalid. Parallel edges add.
Models have immutable neuron/edge tuples and port maps. The normalized model
format is an MVP interchange format, not an efficient whole-brain storage format.
