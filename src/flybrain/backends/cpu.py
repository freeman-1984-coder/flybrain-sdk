"""NumPy edge-list LIF reference backend. No dense N x N matrix or GPU."""

from math import ceil, exp

import numpy as np

from ..config import LIFConfig, finite_number, positive_int
from ..model import Connectome
from .base import Backend, SimulationState


class CPUBackend(Backend):
    name = "cpu"

    def __init__(self, model: Connectome, config: LIFConfig) -> None:
        self.config = config
        self.n = len(model.neuron_ids)
        self._pre = np.array([e.pre for e in model.synapses], dtype=np.intp)
        self._post = np.array([e.post for e in model.synapses], dtype=np.intp)
        self._weights = np.array([e.weight for e in model.synapses], dtype=np.float64)
        self._leak = exp(-config.dt_ms / config.tau_ms)
        self._rate_decay = exp(-config.dt_ms / config.rate_tau_ms)
        self._hold_ticks = ceil(config.refractory_ms / config.dt_ms)
        self._tick = 0
        self._voltage = np.full(self.n, config.rest, dtype=np.float64)
        self._spikes = np.zeros(self.n, dtype=bool)
        self._refractory = np.zeros(self.n, dtype=np.int64)
        self._rates = np.zeros(self.n, dtype=np.float64)
        self._silenced = np.zeros(self.n, dtype=bool)

    def step(self, current: np.ndarray) -> None:
        current = np.asarray(current, dtype=np.float64)
        if current.shape != (self.n,) or not np.isfinite(current).all():
            raise ValueError("current must be a finite vector of length neuron_count")
        # bincount sums repeated postsynaptic indices (including parallel edges).
        synaptic = np.bincount(
            self._post, weights=self._weights * self._spikes[self._pre], minlength=self.n
        )
        available = (self._refractory == 0) & ~self._silenced
        voltage = (
            self.config.rest
            + (self._voltage - self.config.rest) * self._leak
            + (current + synaptic) * (1.0 - self._leak)
        )
        voltage[~available] = self.config.reset
        if not np.isfinite(voltage).all():
            raise ValueError("nonfinite voltage; reduce synapse weights or stimulus current")
        spikes = available & (voltage >= self.config.threshold)
        voltage[spikes] = self.config.reset
        refractory = np.maximum(self._refractory - 1, 0)
        refractory[spikes] = self._hold_ticks
        rates = self._rates * self._rate_decay + spikes * (1000.0 / self.config.dt_ms) * (
            1.0 - self._rate_decay
        )
        self._voltage, self._spikes = voltage, spikes
        self._refractory, self._rates = refractory, rates
        self._tick += 1

    @property
    def tick(self) -> int:
        return self._tick

    def observe_selected(self, indices: tuple, fields: tuple) -> dict:
        arrays = {"voltage": self._voltage, "spikes": self._spikes, "rates_hz": self._rates}
        return {name: tuple(arrays[name][list(indices)].tolist()) for name in fields}

    def set_silenced(self, indices: tuple, enabled: bool) -> None:
        self._silenced[list(indices)] = enabled

    def observe(self) -> SimulationState:
        return SimulationState(
            self._tick,
            self._tick * self.config.dt_ms,
            tuple(self._voltage.tolist()),
            tuple(self._spikes.tolist()),
            tuple(self._rates.tolist()),
        )

    def snapshot(self) -> dict:
        return {
            "tick": self._tick,
            "voltage": self._voltage.tolist(),
            "spikes": self._spikes.tolist(),
            "refractory": self._refractory.tolist(),
            "rates_hz": self._rates.tolist(),
            "silenced": self._silenced.tolist(),
        }

    def restore(self, state: dict) -> None:
        tick = positive_int(state["tick"], "tick", allow_zero=True)
        for key in ("voltage", "spikes", "refractory", "rates_hz"):
            if not isinstance(state[key], list) or len(state[key]) != self.n:
                raise ValueError(f"{key} must have exactly {self.n} elements")
        voltage = np.array([finite_number(v, "voltage") for v in state["voltage"]])
        rates = np.array([finite_number(v, "rate") for v in state["rates_hz"]])
        if np.any(voltage >= self.config.threshold):
            raise ValueError("checkpoint voltage must be below threshold after reset")
        if np.any(rates < 0) or np.any(rates > 1000.0 / self.config.dt_ms):
            raise ValueError("checkpoint rates are out of bounds")
        if any(type(v) is not bool for v in state["spikes"]):
            raise ValueError("checkpoint spikes must be booleans")
        refractory = [positive_int(v, "refractory", allow_zero=True) for v in state["refractory"]]
        if any(v > self._hold_ticks for v in refractory):
            raise ValueError("checkpoint refractory counter is out of bounds")
        silenced = state.get("silenced", [False] * self.n)
        if (
            not isinstance(silenced, list)
            or len(silenced) != self.n
            or any(type(v) is not bool for v in silenced)
        ):
            raise ValueError("silenced must contain one boolean per neuron")
        self._tick, self._voltage, self._rates = tick, voltage, rates
        self._spikes = np.array(state["spikes"], dtype=bool)
        self._refractory = np.array(refractory, dtype=np.int64)
        self._silenced = np.array(silenced, dtype=bool)
