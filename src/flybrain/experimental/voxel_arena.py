"""Deterministic ground arena for neural-controller experiments.

The body is an engineered kinematic approximation, not a fly biomechanics
model. Food affects antenna observations and contact scoring only. Movement
accepts explicit speed/turn commands and never follows a target automatically.
"""

import math

from ..config import finite_number, positive_int
from ..olfaction import sample_odor


class VoxelArena:
    """A floor, solid cubes and one odor source, in arbitrary game units.

    x/z span the floor; y is up. Positive yaw turns toward +z. The fly's body
    remains at y=0.5. Obstacles affect collisions but do not occlude the toy
    Gaussian odor field. A neural adapter receives observe(), not scene().
    """

    def __init__(self, *, food=(4.0, 0.5, 2.0), blocks=((0.0, 0.0), (0.0, 1.0))):
        food = tuple(food)
        if len(food) != 3:
            raise ValueError("food requires x/y/z")
        self.food = tuple(finite_number(v, "food") for v in food)
        self.blocks = tuple(tuple(finite_number(v, "block") for v in b) for b in blocks)
        if any(len(b) != 2 for b in self.blocks):
            raise ValueError("block requires floor x/z")
        self.x, self.z, self.yaw = -4.0, -2.0, 0.0
        self.frame, self.contacts = 0, 0
        self.time_ms = 0.0
        self.reached_food = False
        self.radius = 0.18

    def observe(self):
        # Antennae extend forward and symmetrically to each side of the body.
        forward = (math.cos(self.yaw), math.sin(self.yaw))
        left = (math.sin(self.yaw), -math.cos(self.yaw))
        return {
            f"odor_{side}": sample_odor(
                (
                    self.x + 0.25 * forward[0] + sign * 0.16 * left[0],
                    0.5,
                    self.z + 0.25 * forward[1] + sign * 0.16 * left[1],
                ),
                self.food,
                spread=4.0,
            )
            for side, sign in (("left", 1), ("right", -1))
        }

    def _blocked(self, x, z):
        if abs(x) > 7 - self.radius or abs(z) > 7 - self.radius:
            return True
        for bx, bz in self.blocks:
            dx, dz = max(abs(x - bx) - 0.5, 0), max(abs(z - bz) - 0.5, 0)
            if dx * dx + dz * dz < self.radius**2:
                return True
        return False

    def apply(self, action, duration_ms=20.0):
        dt = finite_number(duration_ms, "duration_ms") / 1000
        if not 0 < dt <= 0.1:
            raise ValueError("body step must be positive and at most 100 ms")
        speed = finite_number(action["speed"], "speed")
        turn = finite_number(action["turn"], "turn")
        if not 0 <= speed <= 3 or abs(turn) > 4:
            raise ValueError("speed must be 0..3 units/s; turn must be -4..4 radians/s")
        self.yaw = (self.yaw + turn * dt + math.pi) % (2 * math.pi) - math.pi
        # Substeps prevent crossing a cube at the maximum supported speed/dt.
        steps = max(1, math.ceil(speed * dt / (self.radius / 2)))
        collision = False
        for _ in range(steps):
            nx = self.x + math.cos(self.yaw) * speed * dt / steps
            nz = self.z + math.sin(self.yaw) * speed * dt / steps
            if self._blocked(nx, nz):
                collision = True
                break
            self.x, self.z = nx, nz
        self.contacts += int(collision)
        self.frame += 1
        self.time_ms += dt * 1000
        self.reached_food |= math.dist((self.x, 0.5, self.z), self.food) <= 0.6
        return {"collision": collision, "reached_food": self.reached_food}

    def scene(self):
        """Renderer/evaluator state, never a neural sensory input."""
        return {
            "position": [self.x, 0.5, self.z],
            "yaw": self.yaw,
            "food": list(self.food),
            "blocks": [list(b) for b in self.blocks],
            "time_ms": self.time_ms,
            "frame": self.frame,
            "contacts": self.contacts,
            "reached_food": self.reached_food,
        }

    def snapshot(self):
        return {"type": "voxel-odor-arena-v1", **self.scene()}

    @classmethod
    def from_snapshot(cls, state):
        if state["type"] != "voxel-odor-arena-v1":
            raise ValueError("unsupported world state")
        arena = cls(food=state["food"], blocks=state["blocks"])
        position = tuple(state["position"])
        if len(position) != 3 or position[1] != 0.5:
            raise ValueError("ground body position requires y=0.5")
        arena.x, arena.z = (finite_number(position[i], "position") for i in (0, 2))
        arena.yaw = finite_number(state["yaw"], "yaw")
        arena.time_ms = finite_number(state["time_ms"], "time_ms")
        if arena._blocked(arena.x, arena.z) or arena.time_ms < 0:
            raise ValueError("invalid body placement or time")
        arena.frame = positive_int(state["frame"], "frame", allow_zero=True)
        arena.contacts = positive_int(state["contacts"], "contacts", allow_zero=True)
        if type(state["reached_food"]) is not bool:
            raise ValueError("reached_food must be boolean")
        arena.reached_food = state["reached_food"]
        return arena
