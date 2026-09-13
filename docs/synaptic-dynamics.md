# Synaptic LIF research engine: independent CPU reference

**CPU reference verified; CUDA implementation and full-brain behavior not yet
validated for this engine.** This is the next implementation step after the
[full-brain odor probe](validation/flywire-odor-a16.md) showed that the numerical
benchmark's incoming normalization cannot support sensory propagation.

The existing dimensionless `FlyBrain` API, models and checkpoints keep their
original meaning. The new interface lives in `flybrain.experimental.synaptic`;
it is not yet exposed by `FlyBrain.load()` or a model catalog entry. It requires
explicit millivolt weights and rejects dimensionless weight units. The original
benchmark is retained as a reproducible historical numerical test.

## Model and source

Nominal equations and parameters follow [Shiu et al., Nature (2024)](https://www.nature.com/articles/s41586-024-07763-9)
and the author's [pinned Brian2 implementation](https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py):

```
dv/dt = (v_rest - v + g) / tau_mem
dg/dt = -g / tau_syn
```

`v` and `g` have mV units. Here `g` is a voltage-valued synaptic drive, not a
conductance in siemens. A source spike schedules a signed increment to `g` after
the transmission delay. The new engine uses the exact solution of the two
coupled linear equations between events, including the equal-time-constant limit.

| Parameter | Nominal value |
|---|---:|
| Step | 0.1 ms |
| Rest / reset | −52 mV |
| Strict spike threshold | greater than −45 mV |
| Membrane time constant | 20 ms |
| Synaptic decay time constant | 5 ms |
| Refractory interval | 2.2 ms |
| Transmission delay | 1.8 ms |

The paper uses **0.275 mV per anatomical contact**, with transmitter-dependent
signs. This is a fitted model parameter, not a measurement of every synapse.
Callers must supply already-converted edge weights. The engine never normalizes
them by incoming degree. Receptor/neuromodulator mechanisms, membrane compartments,
gap junctions, body mechanics, learning and model calibration are not implemented.
Applying these nominal parameters to another graph/version is a new experiment,
not automatic replication of the paper's biological results.

## Event schedule and checkpoints

One step represents the schedule at `t = previous_tick * dt`:

1. Update refractory eligibility and integrate `v` and `g` exactly.
2. Detect threshold crossings.
3. Deliver delayed synaptic events and declared external voltage jumps.
4. Reset spiking cells' `v` and `g`, then update recorded rates.

Both states are frozen and protected from synaptic writes during refractoriness,
matching Brian2's `(unless refractory)` semantics. Eligibility uses integer tick
differences `>= refractory_ticks`, verified against Brian2 2.9.0. Source neurons
listed in `input_ids` have zero refractory duration, as in the paper's Poisson
targets. An externally injected jump can trigger a spike on the next step; an
input coinciding with a reset can be cleared. The oracle exercises this ordering.
Delay and refractory durations must be exact multiples of the timestep; there is
no hidden rounding to a different biological duration.

Snapshots contain `g`, membrane voltage, rates, silencing, last-spike times and
the complete circular buffer of delayed spikes. A fingerprint binds the graph,
weights, dynamics and declared input neurons. Cross-model restores are rejected.
Restore validates state before replacing it. Silencing suppresses future spikes;
events already queued before silencing retain their scheduled delivery.

`step(voltage_jumps_mv)` accepts a finite vector, with nonzero entries only at
declared inputs. This engine has no internal RNG. A later Poisson encoder must
save its own RNG state with the experiment. The original paper uses stochastic
inputs with jump amplitude `0.275 * 250 = 68.75 mV`; validation here uses known
event times, separating solver correctness from random-number generator choice.

## Reproduce independent verification

In an isolated Python 3.12 environment:

```sh
pip install -e . 'Brian2==2.9.0' 'numpy==1.26.4'
python scripts/validate_synaptic_reference.py --output synaptic-reference.json
```

Brian2 2.9.0 accesses an API removed in NumPy 2, so this **oracle environment**
pins NumPy 1.26.4. The SDK's ordinary CPU installation still supports NumPy 2;
Brian2 is not added to runtime dependencies.

The oracle independently constructs the same equations in Brian2 and compares
every cell at every step, with explicit scheduled input events. Four cases cover
0.1/0.2 ms steps, positive/negative and parallel edges, recurrent edges, an
autapse, zero/positive delays, simultaneous input/reset, isolated cells and equal
time constants. Each case includes real downstream spikes in the synthetic test
network. These small fixtures validate the solver; they are not fly models.

Local result: **all four cases passed**, identical spike ticks, membrane errors
at most approximately `5.05e-13 mV`, synaptic-state errors at most approximately
`3.56e-14 mV`. JSON checkpoints taken while an event is in flight replay exactly.
[Machine-readable oracle report](validation/synaptic-brian2-reference.json).
Eleven dependency-free unit cases additionally cover delayed delivery, downstream
activation, invalid units/input selections, incompatible or malformed checkpoints,
silencing, and the near-equal time-constant limit. CI runs the Brian2 oracle in its own environment.

For equal time constants, the oracle uses the same symbolic time constant in both
equations. Brian2's generic expression for distinct symbolic constants divides
by their difference when they happen to have equal numeric values. Using the
equivalent single-symbol equations permits its independent exact solver to
derive the degenerate case, rather than evaluating that removable singularity.

## Required next gates

Port the verified schedule to an optional CUDA engine. Compare all neural state
fields and delayed checkpoint replay on real NVIDIA hardware. Import the complete
FlyWire graph with explicit signed contact-scaled weights, then repeat the
published five-condition odor protocol with reproducible Poisson inputs. Preserve
negative results and compare response stability before any motor readout or
three-dimensional foraging claim. The intended game environment consumes neural
readouts; target coordinates must not reach a hidden seek-food controller.
