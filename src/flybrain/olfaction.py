"""Explicit bilateral odor-to-current adapter; no banana recognition or navigation.

Concentrations and currents are dimensionless engineering quantities. Receptor
identity can be evidence-based while the transfer function remains an assumption.
The adapter accepts only local antenna samples, never a target position/bearing.
"""

from math import exp

import numpy as np

from .config import finite_number
from .neurons import NeuronIndex


class OlfactoryDrive:
    """Inject a saturating stimulus into two explicitly selected ORN populations.

    ``left_ids``/``right_ids`` must come from annotations, not an arbitrary split.
    This stateless adapter has no hidden adaptation or random generator to save.
    Use one adapter per receptor channel and add their current vectors if needed.
    """

    def __init__(self, model, *, left_ids, right_ids, max_current=5.0, half_concentration=0.2):
        index = NeuronIndex(model)
        self.left = index.resolve(left_ids)
        self.right = index.resolve(right_ids)
        if set(self.left) & set(self.right):
            raise ValueError("left and right ORN populations must not overlap")
        self.n = len(model.neuron_ids)
        self.max_current = finite_number(max_current, "max_current")
        self.half_concentration = finite_number(half_concentration, "half_concentration")
        if self.max_current <= 0 or self.half_concentration <= 0:
            raise ValueError("max_current and half_concentration must be positive")

    def amplitude(self, concentration):
        concentration = finite_number(concentration, "concentration")
        if concentration < 0:
            raise ValueError("concentration must be nonnegative")
        # This form avoids overflow when concentration is close to float max.
        fraction = 0.0 if concentration == 0 else 1 / (1 + self.half_concentration / concentration)
        return self.max_current * fraction

    def current(self, *, left, right):
        """A fresh host vector; only the selected ORNs receive external current."""
        amplitudes = self.amplitude(left), self.amplitude(right)
        result = np.zeros(self.n, dtype=np.float64)
        result[list(self.left)] = amplitudes[0]
        result[list(self.right)] = amplitudes[1]
        return result


def sample_odor(position, source, *, spread=4.0, strength=1.0):
    """Isotropic Gaussian concentration field in arbitrary world units.

    A game-side approximation, not a fluid/chemistry model: no wind, turbulence,
    obstacle occlusion or measured banana emissions. Evaluate at each antenna.
    """
    position, source = tuple(position), tuple(source)
    if len(position) != 3 or len(source) != 3:
        raise ValueError("position and source must be three-dimensional")
    position = [finite_number(v, "position") for v in position]
    source = [finite_number(v, "source") for v in source]
    spread = finite_number(spread, "spread")
    strength = finite_number(strength, "strength")
    if spread <= 0 or strength < 0:
        raise ValueError("spread must be positive and strength nonnegative")
    delta = (np.asarray(position) - np.asarray(source)) / spread
    with np.errstate(over="ignore"):
        distance_squared = float(delta @ delta)
    return strength * exp(-0.5 * distance_squared)
