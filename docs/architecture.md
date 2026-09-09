# Architecture and reference dynamics

```text
Game / simulator
      | pre-encoded sensory pulses
      v
FlyBrain -> SensoryEncoder -> Backend.step(current)
                                  | previous spikes -> signed edge list
                                  | LIF voltage / refractory / rates
                                  v
                              MotorDecoder -> MotorAction -> game adapter
```

The public facade owns orchestration, sensory scheduling and checkpoint schema.
`Backend` in `src/flybrain/backends/base.py` defines `step`, `observe`, `snapshot`,
and `restore`. Each backend advances one fixed timestep and exports detached
observations. `create_backend` is the single selection point. Add actual WASM/CUDA
implementations there after equivalence tests; never silently fall back or install
GPU libraries during CPU use. GPU work is optional and not needed for development.

## LIF model

Time is in milliseconds; voltage, weights and currents are normalized toy units,
not fitted membrane measurements. For tick t, previous spikes s[t-1] drive edges:

```text
I_syn[j] = sum(weight[i,j] * s[t-1,i])
a = exp(-dt_ms / tau_ms)
v_next[j] = rest + (v[j] - rest) * a + (I_external[j] + I_syn[j]) * (1 - a)
```

This is exact subthreshold integration for constant current during a tick, with
threshold crossings detected only at tick boundaries. Synaptic current is a
single-tick pulse; there is no conductance model or additional synaptic filter.
The edge orientation is pre -> post. Negative weights inhibit. No dense N×N matrix
is allocated. NumPy `bincount` sums duplicate postsynaptic targets and parallel edges.

Available neurons spike if `v_next >= threshold`, then reset. A spike sets a hold
counter to `ceil(refractory_ms / dt_ms)`. That many **subsequent** full ticks hold
voltage at reset. Every spike has one tick of transmission delay. There is no
noise, random initialization, stochastic release or plasticity.

Firing rates use an exponential filter, `b = exp(-dt_ms / rate_tau_ms)`:

```text
rate_hz = b * rate_hz + (1 - b) * spike * 1000 / dt_ms
action[channel] = clip(mean(rate_hz[port]) / action_rate_hz, 0, 1)
```

The motor decoder is an explicit heuristic. Its nominal saturation rate defaults
to 100 Hz. Different motor outputs can all be 1; they do not sum to one.
Default dynamics: dt=1 ms, tau=10 ms, rest/reset=0, threshold=1, refractory=2 ms,
input gain=2, rate time constant=50 ms. Sensory pulses are additive.

## Performance and portability

CPU time is O(N + E) per tick; arrays use O(N + E) memory. The current Python model
objects, JSON checkpoints and tuple observations have additional overhead. This
implementation is for toy networks and modest subgraphs. Full connectomes need
compact integer arrays, chunked imports and sparse kernels before serious
whole-brain CPU benchmarks. No real-time full-brain claim is made.

Backends must agree on edge summation semantics, threshold/reset order, fixed dt,
refractory counting and pulse boundaries. Begin WASM with float64 and deterministic
reference traces; document tolerances if a future accelerator uses float32.

## Open interfaces added in 0.2 alpha

The facade also owns an ID/annotation index, direct-current scheduler and
replaceable mean-rate readout. `advance()` returns only clock progress;
`observe(ids, fields=...)` and `action()` read only the required backend arrays.
CPU stepping is still a Python loop; GPU batch execution remains planned.

Silencing is a runtime mask: new spikes are suppressed and voltage is held at
reset from the next tick; previous spikes still propagate and rate history decays.
It does not mutate the anatomical graph. Current schedules are consumed only
after a successful backend step. Schema-2 checkpoints include direct currents,
readout configuration and masks; schema 1 remains readable.
