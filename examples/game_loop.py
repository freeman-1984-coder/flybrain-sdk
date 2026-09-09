"""Minimal headless game adapter; velocities here are game-design choices."""

from flybrain import FlyBrain

brain = FlyBrain.load()
brain.stimulate("food", duration_ms=1000)
x = 0.0
# One game frame = 20 neural ticks = 20 ms at the default dt (50 FPS).
for frame in range(50):
    brain.step(20)
    action = brain.action()
    x += action.walk * 3.0 * 0.020
    if frame % 10 == 0:
        print(f"frame={frame:02d} time={brain.state.time_ms:.0f}ms x={x:.3f}")
