# Full-brain voxel odor experiment

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

## Implemented closed loop and remaining behavior work

The [full-graph odor controls](validation/flywire-synaptic-odor-a16.md) establish
that sensory-only input reaches downstream populations. DNa02 is a candidate steering
readout based on [descending control experiments](https://www.nature.com/articles/s41586-024-07523-9),
but this does not establish an appropriate forward-speed mapping, odor response
or behavior in our graph/dynamics. Any gain, baseline gait, trained readout or
external reflex must be disclosed in the recording and reproduction recipe.

The [actual GPU recording](validation/flywire-voxel-a16.md) identifies graph,
configuration, source hashes and GPU. It contains an active and input-silenced
control with the same initial state and RNG seed, recording body pose, input
events, neural output and applied commands. Replaying
it in a browser is **recorded GPU output**, not live browser GPU inference.
World, brain, delayed events and sensory RNG restore together; the next complete
frame was replayed exactly in each run. Report
failures to move or reach food; do not choose successful seeds and hide others.

Run `scripts/run_fullbrain_voxel.py` with the pinned dataset files to produce
the full-brain recording. `site/foraging.html` renders its saved frames. The active
run moved, curved left and ended farther from food; the silenced run remained
stationary. Neither reached food. Calibration and a useful motor readout remain
unfinished. No successful foraging or training result is claimed.
