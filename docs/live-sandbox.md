# Live browser game adapter

**No CUDA required.** [Open the live sandbox](https://freeman-1984-coder.github.io/flybrain-sdk/live.html).
The scene and its neural model are computed on your device. This differs from the
clearly labeled recorded viewers in the demo gallery.

## Try the complete loop

1. Choose the artificial toy or explicitly load the pinned 3.8 MB real MaleCNS circuit.
2. Add an obstacle near the player and step 20 ms. Observe input, output-cell rates,
   requested steering and steering actually applied by the environment.
3. Run continuously. Add obstacles by clicking the canvas or using the accessible
   position slider/button. Changes apply at the next worker command boundary.
4. Change input/output gains or silence outputs. Silencing blocks new spikes;
   existing filtered rates decay, so a previously active player need not stop instantly.
5. Save a checkpoint or export a recording. Select that JSON with Restore / verify.
   A checkpoint continues its world and neural state; a recording recomputes all
   frames and checks them before replacing the current session.

Exports retain a download link and an expandable, read-only JSON text field so
the data can be copied when a browser does not permit automatic downloads.
Imported experiments are labeled as self-described models: replay checks
consistency, not source authenticity. The embedded fingerprint is metadata, not
a cryptographic verification of imported graph content. The built-in real-model
download is independently checked against the catalog's pinned asset checksum.

The page is paused initially and pauses when hidden. Reset creates seed 7 with
resting neural state and default mappings. Each fresh/restored recording can collect
up to 5,000 frames (100 simulated seconds) and 10,000 commands; export and reset
at the limit. File imports are local, JSON-only and limited to 32 MB. A malformed
import does not replace the active worker session. No remote code is loaded from
checkpoint metadata.

## Reuse the adapter

`packages/js/src/dodge.ts` is a reference application adapter built on the existing
`FlyBrain` runtime. It does not add fixed game controls to the neural core. The
compiled package exposes it at `@flybrain-sdk/core/dodge`; the package remains a
private workspace package, not an NPM publication.

```typescript
import { DodgeSession } from "@flybrain-sdk/core/dodge";

const session = new DodgeSession(bundle, 7); // JSON ModelBundle, already loaded
session.command({ op: "add", x: 0.3, y: 0.7 });
const frame = session.step();
console.log(frame.observation, frame.requested, frame.applied);
const checkpoint = session.save();
const resumed = DodgeSession.restore(checkpoint);
const verified = DodgeSession.replay(session.recording());
```

Other commands are `clear`, `configure` (input gain 0..4, output gain 0..2), and
`silence` with a boolean `enabled`. Every command carries its actual environment
frame in recordings. Commands at the final frame are included and replayed, even
when no subsequent integration occurred. Restoring a checkpoint begins a new
recording from that checkpoint; an imported complete recording retains its history.

The Python equivalent remains `flybrain init my-fly --template dodge`. Python's
generic `Session` supports replaceable encoder/readout/environment objects and
fractional periods. This small JS reference adapter currently supports only the
two demo models, `dt_ms=1`, and a fixed 20 ms environment period. Implement other
adapters using `FlyBrain.inject/observe/advance` without changing the neural core.

## Timing, scope and reproducibility

Each frame samples present obstacle features, injects normalized currents for 20
neural ticks, reads the two filtered output rates, and advances the world by 20 ms.
Rendering is outside the simulation worker. The displayed speed is simulated time
divided by elapsed wall time for the current run. Slow devices slow the scene;
the scheduler does not drop neural steps to catch up. This is not a full-brain
real-time benchmark. Graphs larger than 1,000 cells / 100,000 edges are rejected
by this adapter; the generic runtime has a separate scope.

Default environment rules match Python `DodgeArena`: normalized positions, player
speed 1 unit/second, obstacles descend at 0.8 units/second, collision distance 0.13,
and deterministic LCG spawning every 45 frames. The interactive adapter adds a cap
of 64 obstacles; automatic spawns are skipped while at that cap. Added obstacles
must have x in 0..1 and y in 0..0.95. Collisions are checked at frame ends.

`scripts/dodge_reference.py` generates 150 frames each for toy and real models
using Python Session, including obstacle edits, gain changes and silencing.
JavaScript tests compare every recorded environment/input/control frame and final
neural state at absolute tolerance 1e-9, with boolean spikes exact. The core runtime
also has its existing every-cell/per-tick Python comparisons. Worker tests cover
save, restore, complete replay and failed-import state preservation. These checks
do not establish biological fidelity, task learning or superiority of real wiring.

The new formats are `flybrain-dodge-checkpoint` and `flybrain-dodge-recording`,
schema 1. They embed the JS neural checkpoint plus environment, settings and model
identity; recordings also store edits and full per-frame feedback. They are not
Python Session checkpoints. Replay verifies integral frame/tick timestamps exactly
and floating state to 1e-9. Endpoints and recordings are deterministic within this
implementation; no external commercial game's save state is promised.

The toy is artificial. The real model contains anatomical MaleCNS wiring with
assumed LIF dynamics, LC4/LPLC2 input projection and engineered GF-to-steering
readout. Its [model card](../models/male-cns-escape-v1/README.md) records source,
license, transformation, fingerprint and measurements. This demo has no learned
policy, rescue controller or biological claim about steering from GF activity.

## Build locally

```sh
pip install -e '.[dev]'
python scripts/dodge_reference.py
cd packages/js
npm ci
npm test
cd ../..
node scripts/build-browser-site.mjs
python -m http.server 8765 --directory site
```

Open `http://localhost:8765/live.html`. The shared loader verifies and caches the
real model using the catalog's pinned URL, size and SHA256. The toy loads with the
page. After assets are loaded the simulation needs no network; this is not a
service-worker installation or a promise of offline page reload.
