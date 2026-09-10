# Godot CPU adapter

**No CUDA required.** A real Godot 4 scene owns the world while the installed
Python SDK owns the neural simulation. It runs offline with the artificial toy;
the existing 313-cell MaleCNS bundle is an explicit optional download.

Available in v0.4.0a4. Verified with actual Godot 4.7.2 processes on macOS and
Linux CI. Native macOS run/pause, save/restore, file-dialog cancellation and
service-disconnect recovery were checked on 2026-09-11.

## Run from this checkout

Install [Godot 4.7.2](https://godotengine.org/download/archive/4.7.2-stable/).
The verification engine is the standard edition; .NET and export templates are
not needed. From the repository root:

```sh
python -m pip install -e .
python examples/godot/bridge.py
```

Import `examples/godot/project.godot` in Godot and run the project (F5), or in
another terminal run `godot --path examples/godot`. Click **Run / pause**.
The initial scene is paused. Wait for the bridge to print its ready address before
starting Godot; optional model download happens before the server starts listening.

For the real subgraph, restart the bridge with:

```sh
python examples/godot/bridge.py --model male-cns-escape-v1 --download
```

The toy is hand-designed. Real connectivity is anatomical MaleCNS data, but LIF
parameters, input currents and GF-to-steering assignment are engineered demo
choices. No learned policy, complete fly, or validated natural behavior is claimed.
The same model attribution and CC BY 4.0 terms apply; see the
[model card](../../models/male-cns-escape-v1/README.md).

## Save and continue

**Save session** pauses after the outstanding frame is acknowledged and writes a
new `flybrain-session-*.json` file in Godot's user-data folder. The on-screen message
shows its location. **Restore saved session** opens that folder; select a saved
file and press Run to continue. **Restart** resets both participants to a fresh
session. Only one game instance should use one bridge process/port.

The file contains the whole engine arena (including spawn RNG), brain checkpoint,
input/output mapping, and matching sequence clock. The brain JSON is stored as an
opaque string: Godot's JSON numbers are floating-point, and parsing/re-serializing
the neural checkpoint would lose its integer types and potentially precision.
The server parses the original string with the SDK's checkpoint validation.

Checkpoints are taken at acknowledged boundaries. They are not interchangeable
with the browser sandbox's or generic Python Session's files. An import is a
self-described experiment; successful restore does not authenticate its source.
The included frame trace is diagnostic history, not an independently verified
replay merely because a save file opens.

## Replace the pieces

- `arena.gd`: engine observations, obstacle/world update and actual applied control.
- `bridge.py`: compose a model, `LinearEncoder` and `RateReadout` in Python. These
  can be replaced without editing Godot's HTTP loop or the neural runtime.
- `main.gd`: transport, fixed simulated-time handshake, rendering and whole-session
  save/restore. Change its scene/rendering code to build your own game.
- `flybrain.external.ExternalController`: reusable integration boundary independent
  of Godot and HTTP; arbitrary named numeric observations and output channels.

The default encoder maps current left/right obstacle intensity to the corresponding
sensory cells with gain 2. The readout subtracts left steering output rate from
right steering output rate, divides by 100 Hz and clamps to [-1, 1]. The engine
moves the player by `steer * 0.02` normalized units and clips at walls, then reports
the actual displacement divided by 0.02. No fallback controller intervenes.

This first adapter supports a continuous steering axis. Short button presses,
pose streams, learning, remote hosting and multiple simultaneous clients remain
future work. The core does not hard-code those future game actions.

## Timing and transport contract

All requests are native-client POST JSON to `127.0.0.1:8766`. The example server
is a serial, loopback-only development process, with no browser CORS support.
Use separate ports for separate clients. It is not a public production API.

1. `/start` returns a fresh session identity and clock; optional `checkpoint_json`
   restores the controller. Starting a new session invalidates the previous one.
2. `/offer` supplies session, sequence and current observation. Python integrates
   exactly 20 neural ticks (1 ms each) and returns requested controls for 20 ms.
3. Godot applies that action once, advances its world 20 ms and sends `/ack` with
   the actual applied controls. Only then can the next observation be offered.
4. `/checkpoint` is allowed only at an acknowledged boundary.

Identical pending offers and last acknowledgements are idempotent in Python.
Conflicting retries and out-of-order requests are rejected. The reference Godot
client has one request in flight and pauses on failure; it does not automatically
reapply an uncertain response or retry a partially executed world update.

Render speed does not set neural time. Transport and rendering add wall-clock
latency, so this debug-oriented example may run slower than real time. No neural
steps are skipped. Pause completes an already issued frame, then stops. Timeout
or rejected/expired action stops the scene and clears active control; reset or
restore both participants before continuing. Responses older than two wall-clock
seconds are rejected. This is not a claim of network-wide exactly-once delivery.

The generic controller accepts integer-multiple neural periods from 1..1000 ms;
this Godot scene is fixed at 20 ms. Recordings cap at 10,000 acknowledged frames
per fresh/restored controller. JSON requests, responses and imports cap at 16 MB.
A decoder failure after neural integration faults the controller and requires
restore/reset rather than repeating the same integration.

## Verify with the actual engine

```sh
python scripts/verify_godot.py /absolute/path/to/godot
python scripts/verify_godot_failures.py /absolute/path/to/godot
```

The script starts a temporary loopback bridge, runs **Godot itself** for 120 frames,
saves, starts a new Godot process from that checkpoint, and continues to frame 200.
For toy and real circuits it compares every observation, requested/applied control,
world state and final neural state against Python `Session` / `DodgeArena` at
absolute tolerance 1e-9 (integer clocks exact). It uses repository model data;
there is no model download during verification. It also checks that stale actions
and mismatched world/brain saves are rejected before movement. CI pins the engine archive and
checks its SHA256 before executing it.

The fault suite runs the real engine against malformed start/action/ack/checkpoint
responses, a dropped connection and a response delayed beyond the timeout. The
last two cases drop a response after neural integration: the engine remains at
its last applied frame, clears active control and requires recovery. These checks
exercise transport failure; they do not replace visual interaction checks.

Useful manual checks before release: run/pause at several points, save/restore
through the file dialog, stop the bridge while running and confirm the scene stops,
restart both components, and inspect text/layout at the default window size.

Godot API references: [HTTPRequest](https://docs.godotengine.org/en/stable/classes/class_httprequest.html)
and [JSON](https://docs.godotengine.org/en/stable/classes/class_json.html).
Code in this example uses the repository's MIT license. Godot is separately MIT
licensed and is not redistributed in the Python wheel.
