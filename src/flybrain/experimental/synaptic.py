"""CPU reference for a delayed, exponentially decaying synaptic LIF model.

Equations/nominal defaults follow Shiu et al. (Nature 2024), with an explicit
Brian2-style schedule. Voltages, synaptic state and edge weights use millivolts.
No incoming-weight normalization. This is not wired into FlyBrain.load yet:
dimensionless checkpoints/weights must never be silently reinterpreted as mV.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import exp, expm1, isclose

import numpy as np

from ..config import finite_number, positive_int
from ..neurons import NeuronIndex


@dataclass(frozen=True)
class SynapticLIFConfig:
    dt_ms: float = 0.1
    rest_mv: float = -52.0
    reset_mv: float = -52.0
    threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    rate_tau_ms: float = 50.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            object.__setattr__(self, name, finite_number(value, name))
        for name in ("dt_ms", "membrane_tau_ms", "synapse_tau_ms", "rate_tau_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rest_mv >= self.threshold_mv or self.reset_mv >= self.threshold_mv:
            raise ValueError("rest and reset must be below threshold")
        for name in ("delay_ms", "refractory_ms"):
            value = getattr(self, name)
            if value < 0 or not isclose(
                value / self.dt_ms, round(value / self.dt_ms), rel_tol=0, abs_tol=1e-9
            ):
                raise ValueError(f"{name} must be a nonnegative integer multiple of dt_ms")

    def to_dict(self):
        return asdict(self)


class SynapticCPU:
    """Sparse CPU research engine; edge weights must explicitly declare mV units.

    A step receives externally scheduled voltage jumps (mV) for input_ids only.
    These cells have zero refractory duration, matching the paper's Poisson
    targets. Input jumps happen after threshold detection, before resets. They
    can therefore cause a spike on the next step, or be cleared by a same-step
    reset. No RNG is hidden in this object; callers checkpoint their input RNG.
    """

    name = "cpu"
    engine = "lif-synaptic-mv-v1"

    def __init__(self, model, config=None, *, input_ids=(), weight_units):
        if weight_units != "mV":
            raise ValueError("synaptic weights must explicitly use mV")
        self.config = config or SynapticLIFConfig()
        if not isinstance(self.config, SynapticLIFConfig):
            raise ValueError("SynapticLIFConfig required; dimensionless config is incompatible")
        c = self.config
        self.n = len(model.neuron_ids)
        self._pre = np.array([e.pre for e in model.synapses], dtype="<i8")
        self._post = np.array([e.post for e in model.synapses], dtype="<i8")
        self._weights = np.array([e.weight for e in model.synapses], dtype="<f8")
        self._inputs = np.zeros(self.n, dtype=bool)
        if input_ids:
            self._inputs[list(NeuronIndex(model).resolve(input_ids))] = True
        self._ref_steps = np.full(self.n, round(c.refractory_ms / c.dt_ms), dtype=np.int64)
        self._ref_steps[self._inputs] = 0
        self._delay = round(c.delay_ms / c.dt_ms)
        self._em = exp(-c.dt_ms / c.membrane_tau_ms)
        self._es = exp(-c.dt_ms / c.synapse_tau_ms)
        decay_difference = 1 / c.synapse_tau_ms - 1 / c.membrane_tau_ms
        if decay_difference == 0:
            self._coupling = self._em * c.dt_ms / c.membrane_tau_ms
        elif decay_difference > 0:
            self._coupling = (
                self._em
                * -expm1(-c.dt_ms * decay_difference)
                / (c.membrane_tau_ms * decay_difference)
            )
        else:
            self._coupling = (
                self._es
                * expm1(c.dt_ms * decay_difference)
                / (c.membrane_tau_ms * decay_difference)
            )
        self._rate_decay = exp(-c.dt_ms / c.rate_tau_ms)
        self.tick = 0
        self._voltage = np.full(self.n, c.rest_mv)
        self._g = np.zeros(self.n)
        self._spikes = np.zeros(self.n, dtype=bool)
        self._last_spike = np.full(self.n, -round(c.refractory_ms / c.dt_ms) - 1, dtype=np.int64)
        self._rates = np.zeros(self.n)
        self._silenced = np.zeros(self.n, dtype=bool)
        self._history = np.zeros((self._delay + 1, self.n), dtype=bool)
        h = hashlib.sha256()
        h.update(json.dumps([model.neuron_ids, c.to_dict()], sort_keys=True).encode())
        for array in (self._pre, self._post, self._weights, self._inputs):
            h.update(array.tobytes())
        self._fingerprint = h.hexdigest()

    def step(self, voltage_jumps_mv):
        jumps = np.asarray(voltage_jumps_mv, dtype=np.float64)
        if jumps.shape != (self.n,) or not np.isfinite(jumps).all():
            raise ValueError("input jumps must be a finite vector matching the graph")
        if np.any(jumps[~self._inputs] != 0):
            raise ValueError("voltage jumps are allowed only at declared input neurons")
        c = self.config
        available = (self.tick - self._last_spike >= self._ref_steps) & ~self._silenced
        # Exact coupled integration uses the OLD synaptic state in the voltage update.
        v = self._voltage.copy()
        g = self._g.copy()
        v[available] = (
            c.rest_mv + (v[available] - c.rest_mv) * self._em + g[available] * self._coupling
        )
        g[available] *= self._es
        spikes = available & (v > c.threshold_mv)
        # Thresholds -> delayed synaptic events -> external input -> reset.
        # Both v and g are read-only during refractory time, as in Brian2's
        # `(unless refractory)`, including writes from arriving synapses.
        history = self._history.copy()
        history[self.tick % len(history)] = spikes
        arriving = history[(self.tick - self._delay) % len(history)]
        synaptic = np.bincount(
            self._post, weights=self._weights * arriving[self._pre], minlength=self.n
        )
        g[available] += synaptic[available]
        v[available] += jumps[available]
        v[spikes | self._silenced] = c.reset_mv
        g[spikes | self._silenced] = 0
        if not np.isfinite(v).all() or not np.isfinite(g).all():
            raise ValueError("nonfinite synaptic state; check weights and inputs")
        self._last_spike[spikes] = self.tick
        self._voltage, self._g, self._spikes, self._history = v, g, spikes, history
        self._rates = self._rates * self._rate_decay + spikes * (1000 / c.dt_ms) * (
            1 - self._rate_decay
        )
        self.tick += 1

    def observe_selected(self, indices, fields=("voltage_mv", "synaptic_mv", "spikes", "rates_hz")):
        indices = self._checked_indices(indices)
        arrays = {
            "voltage_mv": self._voltage,
            "synaptic_mv": self._g,
            "spikes": self._spikes,
            "rates_hz": self._rates,
        }
        return {name: tuple(arrays[name][list(indices)].tolist()) for name in fields}

    def set_silenced(self, indices, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        self._silenced[list(self._checked_indices(indices))] = enabled

    def _checked_indices(self, indices):
        indices = tuple(indices)
        if any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer))
            or i < 0
            or i >= self.n
            for i in indices
        ):
            raise ValueError("selection requires valid integer neuron indices")
        return indices

    def snapshot(self):
        return {
            "engine": self.engine,
            "fingerprint": self._fingerprint,
            "tick": self.tick,
            "voltage_mv": self._voltage.tolist(),
            "synaptic_mv": self._g.tolist(),
            "spikes": self._spikes.tolist(),
            "rates_hz": self._rates.tolist(),
            "last_spike_tick": self._last_spike.tolist(),
            "history": self._history.tolist(),
            "silenced": self._silenced.tolist(),
        }

    def restore(self, state):
        if state.get("engine") != self.engine or state.get("fingerprint") != self._fingerprint:
            raise ValueError("checkpoint graph, input population or dynamics mismatch")
        tick = positive_int(state["tick"], "tick", allow_zero=True)
        arrays = {}
        for key, dtype in (
            ("voltage_mv", float),
            ("synaptic_mv", float),
            ("rates_hz", float),
            ("spikes", bool),
            ("silenced", bool),
            ("last_spike_tick", int),
        ):
            value = state[key]
            if not isinstance(value, list) or len(value) != self.n:
                raise ValueError(f"{key} must have exactly {self.n} entries")
            if dtype in (bool, int) and any(type(v) is not dtype for v in value):
                raise ValueError(f"{key} has invalid entry types")
            arrays[key] = np.array(
                [finite_number(v, key) for v in value] if dtype is float else value, dtype=dtype
            )
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{key} must be finite")
        history = state["history"]
        if not isinstance(history, list) or len(history) != self._delay + 1:
            raise ValueError("checkpoint has wrong delay buffer length")
        if any(
            not isinstance(row, list) or len(row) != self.n or any(type(v) is not bool for v in row)
            for row in history
        ):
            raise ValueError("delay buffer must contain one boolean per neuron per slot")
        last = arrays["last_spike_tick"]
        minimum = -round(self.config.refractory_ms / self.config.dt_ms) - 1
        if (last < minimum).any() or (last >= tick).any():
            raise ValueError("last spike timestamps are out of range")
        rates = arrays["rates_hz"]
        if (rates < 0).any() or (rates > 1000 / self.config.dt_ms).any():
            raise ValueError("rates are out of range")
        self.tick = tick
        self._voltage, self._g = arrays["voltage_mv"], arrays["synaptic_mv"]
        self._rates, self._spikes = rates, arrays["spikes"]
        self._silenced, self._last_spike = arrays["silenced"], last
        self._history = np.array(history, dtype=bool)
