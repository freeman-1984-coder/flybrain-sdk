# Reusable sessions: game feedback and circuit sonification

**No CUDA required.** These Python examples use the same `Session` loop with replaceable input projection, rate readout and environment adapters. The toy and pinned real MaleCNS escape circuit are supported by the presets. The real graph is anatomical; every input projection, LIF parameter and game/audio mapping here is an engineered assumption.

[Live game sandbox](https://freeman-1984-coder.github.io/flybrain-sdk/live.html) · [Gallery and audio](https://freeman-1984-coder.github.io/flybrain-sdk/demos.html) · [Live circuit lab](https://freeman-1984-coder.github.io/flybrain-sdk/lab.html)

## Generate your own demo

From a repository checkout:

```sh
pip install -e .
python examples/demo_session.py dodge --output work/dodge
python examples/demo_session.py tones --output work/tones
# A real model downloads only with your explicit opt-in:
python examples/demo_session.py tones --model male-cns-escape-v1 --download --output work/real-tones
```

Each command writes `recording.json`, `session.json` and `replay.html`, then verifies the full recorded feedback loop. The tone demo also writes `tones.wav`. Open the HTML to inspect playback; its slider and speed controls change the view, not the simulation. Modify the Python recipe and rerun it to generate different behavior. The website's existing circuit lab runs a live brain; the new gallery viewers are clearly labeled recordings.

## Create an editable project (0.4.0a2)

After installing the SDK, these commands also work without a repository checkout:

```sh
flybrain doctor
flybrain demos list
flybrain init my-fly --template dodge
cd my-fly
python app.py --output runs/first
```

`python -m flybrain` is an equivalent entry point. `init` writes ordinary source,
editable `recipe.json`, a pinned Git-tag requirement, README, LICENSE and model
attribution. Existing project/output directories are never overwritten. Change
`encoder.weights` or `readout.channels`, then run to a new output directory;
replace components in `build_session()` for more extensive changes. The model
fingerprint rejects accidental mismatches between a model and its mappings.

To start with the real model, use
`flybrain init my-real --template tones --model male-cns-escape-v1 --download`.
The download is explicit; subsequent runs use the cache. A fresh machine must
pass `--download` when running that generated app. A raw full dataset is not a
ready-made demo model. All presets remain CPU based and have no learning.

For an immediate run without project generation:
`flybrain demos run tones --frames 300 --output work/tones`.
`doctor` runs a small local CPU check, reports backend availability, and uses no
network. Availability is not CUDA conformance. No PyPI publication is assumed.

## One loop, three replaceable parts

```python
from flybrain import FlyBrain
from flybrain.demos import DodgeArena
from flybrain.session import LinearEncoder, RateReadout, Session

brain = FlyBrain.load("toy")
encoder = LinearEncoder(
    {
        "danger_left": {"sense.looming_left": 2.0},
        "danger_right": {"sense.looming_right": 2.0},
    }
)
readout = RateReadout(
    {
        "steer": {
            "weights": {"motor.turn_right": 1.0, "motor.turn_left": -1.0},
            "min": -1.0,
            "max": 1.0,
        }
    },
    scale_hz=100,
)
session = Session(brain, encoder, readout, DodgeArena(seed=7), period_ms=20)
frame = session.step()
print(frame.observation, frame.requested, frame.applied)
```

The encoder maps observed features to normalized per-cell currents. Multiple features targeting the same cell add together. The readout is a weighted sum of filtered firing rates divided by its declared scale, then clipped to declared bounds. It supports signed controls without changing the brain graph or its existing `action()` binding.

`Environment.observe()` supplies current, JSON-serializable observations. `Environment.apply(action, duration_ms)` advances the world and returns the controls actually applied. `snapshot()` must capture everything required for continuation. Custom encoder/readout/environment objects can implement these small protocols. Raw high-bandwidth video/audio transport is not provided; extract features or supply a suitable serializable representation in your environment adapter.

For the real demo, both LC4/LPLC2 input populations are explicitly driven and the two GF cells (`10010`, `10001`) feed named outputs. Assigning GF activity to steering or oscillator volume is an application mapping, not a biological claim about the cells' natural meaning.

## Timing and records

The session owns the neural clock. One step observes the environment, projects currents, advances the assigned integer neural ticks, reads outputs, then advances the environment. A period such as 16.6666667 ms alternates 16/17 ticks at a 1 ms neural step; an exact decimal accumulator keeps the discrepancy below one neural tick. Periods shorter than one neural tick are rejected. `brain.progress` reads the clock without copying neural arrays.

Each recorded frame includes input observations, projected currents, neural tick, requested controls, applied controls and the resulting environment state. The dodge example reports reduced applied steering at walls. Only current visible obstacle features reach its encoder; random generator state and detailed environment snapshots are recorded for inspection/restore, not passed to the brain as future observations.

Do not call `brain.step()` independently while a session owns it. Stimuli and interventions may be added at session boundaries, but an additional change made after recording starts needs its own recorded adapter state/event mechanism; the current `record()` helper runs uninterrupted frames. If a custom adapter throws after neural integration, recover from a saved checkpoint rather than retrying the partly completed frame.

## Save, continue, and replay

```python
from flybrain.demos import DodgeArena
from flybrain.session import Session, replay

session.save("session.json")
restored = Session.restore("session.json", environment_factory=DodgeArena.from_snapshot)
recording = restored.record(300)
verified = replay(recording, environment_factory=DodgeArena.from_snapshot)
```

Session checkpoints contain the entire brain (including pending inputs and interventions), component configurations/state and clock. `FlyBrain.snapshot()` and `FlyBrain.from_snapshot()` expose the existing JSON checkpoint format without temporary files. Environment/adapter factories come from caller code; a recording never names a module to import or code to execute. Use `PulseScore.from_snapshot` for tone sessions and your own trusted factories for custom components.

Replay recomputes the environment feedback and compares every frame plus final brain/component state. Floats use absolute tolerance `1e-9`; integers, booleans, spikes, keys and sequence lengths must match exactly. Default limits are 10,000 frames and one million neural ticks. This is a deterministic small-environment protocol, not a guarantee that an external commercial game can restore arbitrary live state. Custom stochastic adapters must include random state in their snapshots.

## What the two demos show

- **Dodge:** a normalized one-dimensional player avoids descending obstacles. A deterministic generator supplies positions; neural rate differences determine steering. There is no rescue controller or learned policy. The gallery compares the same seed/run length with both output neurons active and silenced; this small comparison is not a general performance benchmark.
- **Tones:** alternating synthetic pulses stimulate two pathways. Their neural output rates control 440/660 Hz oscillator gains. The WAV is original synthesized audio, not a song or a demonstration of musical comprehension. Silencing the output cells from the start produces silence.

Dodge uses coordinates in `[0, 1]`, player speed 1 unit/second at full steering, obstacle descent 0.8 units/second, a collision distance of 0.13, and one spawn every 45 environment frames. Collisions are checked at frame ends. Changing the environment period changes the spawn frequency. The score emits 100 ms pulses at 0/250/500/750 ms of a repeating one-second cycle. These rules are versioned in each environment snapshot.

`render_wav(recording, path)` uses applied output gains and absolute session time. It streams mono 16-bit PCM with gain ramps and an optional ending fade. It can render a resumed recording; use `fade_out=False` when joining segments. Tests verify that resumed clips concatenate sample-for-sample with an uninterrupted render. The renderer caps exports at ten million samples and supports sample rates from 8 to 192 kHz.

## Reproduce the public gallery

```sh
pip install -e '.[dev]'
python scripts/build_demo_gallery.py
pytest tests/test_session.py
```

The script loads the checked-in, pinned real model, regenerates three six-second replays and the WAV, and verifies each feedback trace before writing `site/demos/manifest.json`. The manifest records environment versions, model checksum, scenario outcomes and artifact checksums. HTML viewers omit the full graph and need no network to play. Keep the code license and model attribution when sharing exports.

The [live browser sandbox](live-sandbox.md) now implements the dodge reference
with Python parity checks. Still planned: external engine adapters, streaming audio/features, trained readouts with held-out evaluation, and validated CUDA. The current templates provide runnable, inspectable starting points for that work.
