# Python API reference

## Load and observe

```python
from flybrain import FlyBrain, LIFConfig, Stimulus

brain = FlyBrain.load("toy", config=LIFConfig(dt_ms=1.0))
brain.stimulate(Stimulus("food", strength=0.8, duration_ms=120))
state = brain.step(20)
print(state.tick, state.time_ms, state.voltage, state.spikes, state.rates_hz)
print(brain.action().walk)
```

`FlyBrain.load(model="toy", *, backend="cpu", config=None)` accepts the builtin name,
a local JSON path, or a validated `Connectome`. It never downloads. Recognized raw
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
run per frame. Calling `action()` is a pure read and returns `MotorAction`.
`to_dict()` produces ordinary JSON-compatible values. Motor channels missing from
a custom model output zero. Unsupported motor names are rejected in this MVP.

## Checkpoints

```python
brain.save("brain.checkpoint.json")
continued = FlyBrain.restore("brain.checkpoint.json", backend="cpu")
```

`save()` returns the destination `Path`; its parent directory must exist. It uses
atomic file replacement. `restore()` is a class method returning a **new** brain.
The checkpoint embeds the model and its SHA256 fingerprint, config, dynamics
revision, tick, voltages, previous spikes, refractory counters, filtered firing
rates, and all pending stimuli. No pickle or code execution is used.

Unknown schema/dynamics versions, altered model fingerprints, invalid vector
shapes and nonfinite state are rejected with `CheckpointError`. File access
errors remain ordinary `OSError` subclasses. Unavailable backends raise
`BackendUnavailableError`. Exact continuation is tested on the same CPU runtime;
bit-for-bit reproducibility across different numerical libraries/hardware or
future backends is not promised. A fingerprint checks consistency, not authenticity.

## Custom models

Use `Connectome.from_dict()` or edit a copy of `Connectome.load().to_dict()`.
Schema version 1 contains `name`, string `neuron_ids`, `synapses` (pre ID, post ID,
signed `weight`), `sensory` and `motor` port maps, and string-valued `provenance`.
IDs are kept as strings, including numeric source IDs. Duplicate neuron IDs,
unknown edge endpoints and nonfinite weights are invalid. Parallel edges add.
Models have immutable neuron/edge tuples and port maps. The normalized model
format is an MVP interchange format, not an efficient whole-brain storage format.
