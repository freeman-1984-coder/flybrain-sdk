# Full-brain voxel foraging experiment — implementation in progress

The requested experiment couples the **entire FlyWire graph on an NVIDIA GPU**
to a small voxel world. A fruit mesh will represent a simplified food-odor
source. It does not imply banana vision, a complete odor blend, flight mechanics
or naturally validated food-seeking behavior.

## Implemented environment

`flybrain.experimental.voxel_arena.VoxelArena` provides floor motion, solid cubic
obstacles, collision detection, two antenna concentration samples, food-contact
scoring and JSON world-state replay. It is a kinematic ground body in arbitrary
game units. Odor uses a Gaussian field without turbulence or obstacle occlusion.

The control boundary is explicit:

1. The arena exposes only `odor_left` and `odor_right` to the sensory adapter.
2. The adapter schedules events on annotated ORNs in the full network.
3. A separately declared readout converts neural activity to speed and turn.
4. `arena.apply({"speed": ..., "turn": ...}, duration_ms=20)` moves the body.
5. `arena.scene()` supplies positions/food/obstacles to rendering and evaluation;
   it must not be passed to the neural controller or readout.

Tests verify that relocating food changes the sensed odor, but cannot change
the trajectory under identical commands. Zero movement commands keep the body
stationary. Obstacles stop motion, and JSON restore reproduces future pose,
concentration and collision results.

## Remaining integration and acceptance

The new synaptic full-graph odor controls must first establish what activity
actually reaches candidate output populations. DNa02 is a candidate steering
readout based on [descending control experiments](https://www.nature.com/articles/s41586-024-07523-9),
but this does not establish an appropriate forward-speed mapping, odor response
or behavior in our graph/dynamics. Any gain, baseline gait, trained readout or
external reflex must be disclosed in the recording and reproduction recipe.

The final recording must identify graph/configuration/source hashes and actual
GPU, contain paired controls with matched initial state and sensory RNG, and
record body pose, input events, neural output and applied commands. Replaying
it in a browser is **recorded GPU output**, not live browser GPU inference.
World, brain, delayed events and sensory RNG must restore together. Report
failures to move or reach food; do not choose successful seeds and hide others.

The renderer and full-brain closed loop are not delivered by this environment
module alone. No successful foraging or training result is claimed yet.
