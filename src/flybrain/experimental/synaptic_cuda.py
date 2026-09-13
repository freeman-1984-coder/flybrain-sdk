"""Optional CUDA port of the separately versioned mV synaptic research engine.

Two ordered kernels implement integrate/threshold, then synaptic/input/reset.
Only after a successful step are state buffers and delayed-spike history committed.
Importing this module does not import CuPy or initialize a device.
"""

import numpy as np

from ..backends.cuda import _load_cupy
from ..errors import BackendUnavailableError
from .synaptic import SynapticCPU

_KERNELS = r"""
extern "C" __global__ void integrate(
    int n, long long tick, const double* old_v, const double* old_g,
    const long long* last_spike, const long long* ref_steps,
    const unsigned char* silent, double* v, double* g,
    unsigned char* spikes, unsigned char* available,
    double rest, double threshold, double em, double es, double coupling) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool active = tick - last_spike[i] >= ref_steps[i] && !silent[i];
    available[i] = active;
    v[i] = active ? rest + (old_v[i] - rest) * em + old_g[i] * coupling : old_v[i];
    g[i] = active ? old_g[i] * es : old_g[i];
    spikes[i] = active && v[i] > threshold;
}
extern "C" __global__ void events_and_reset(
    int n, long long tick, const long long* row, const int* pre,
    const double* weights, const unsigned char* arriving, const double* jumps,
    const unsigned char* silent, const unsigned char* available,
    const unsigned char* spikes, const double* old_rates,
    const long long* old_last_spike, double* v, double* g,
    double* rates, long long* last_spike, int* error,
    double reset, double rate_decay, double rate_hz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (available[i]) {
        double synaptic = 0.0;
        for (long long e = row[i]; e < row[i+1]; ++e)
            synaptic += weights[e] * (double)arriving[pre[e]];
        g[i] += synaptic;
        v[i] += jumps[i];
    }
    if (spikes[i] || silent[i]) { v[i] = reset; g[i] = 0.0; }
    rates[i] = old_rates[i] * rate_decay + ((double)spikes[i] * rate_hz) * (1.0-rate_decay);
    last_spike[i] = spikes[i] ? tick : old_last_spike[i];
    if (!isfinite(v[i]) || !isfinite(g[i]) || !isfinite(rates[i])) atomicExch(error, 1);
}
"""


class SynapticCUDA:
    """Research CUDA engine with the same explicit units/API as SynapticCPU."""

    name = "cuda"
    engine = SynapticCPU.engine

    def __init__(self, model, config=None, *, input_ids=(), weight_units):
        self._reference = SynapticCPU(model, config, input_ids=input_ids, weight_units=weight_units)
        ref = self._reference
        self.config, self.n, self.tick = ref.config, ref.n, 0
        if self.n > np.iinfo(np.int32).max:
            raise ValueError("CUDA indices require neuron count within int32 range")
        cp = self._cp = _load_cupy()
        try:
            self._device = cp.cuda.Device()
            self._stream = cp.cuda.Stream(non_blocking=True)
            with self._device, self._stream:
                order = np.argsort(ref._post, kind="stable")
                rows = np.empty(self.n + 1, dtype=np.int64)
                rows[0] = 0
                np.cumsum(np.bincount(ref._post, minlength=self.n), out=rows[1:])
                self._row = cp.asarray(rows)
                self._pre = cp.asarray(ref._pre[order].astype(np.int32))
                self._weights = cp.asarray(ref._weights[order])
                self._refs = cp.asarray(ref._ref_steps)
                self._integrate = cp.RawKernel(_KERNELS, "integrate", options=("--fmad=false",))
                self._events = cp.RawKernel(_KERNELS, "events_and_reset", options=("--fmad=false",))
                self._integrate.compile()
                self._events.compile()
                self._state = self._upload(ref.snapshot())
                self._next = {
                    k: cp.empty_like(self._state[k])
                    for k in ("voltage_mv", "synaptic_mv", "spikes", "rates_hz", "last_spike_tick")
                }
                self._available = cp.empty(self.n, dtype=cp.bool_)
                self._error = cp.zeros(1, dtype=cp.int32)
                self._stream.synchronize()
        except Exception as error:
            raise BackendUnavailableError(
                f"Synaptic CUDA initialization failed: {error}"
            ) from error

    def _upload(self, snapshot):
        dtypes = {
            "voltage_mv": np.float64,
            "synaptic_mv": np.float64,
            "rates_hz": np.float64,
            "last_spike_tick": np.int64,
            "spikes": np.bool_,
            "silenced": np.bool_,
            "history": np.bool_,
        }
        return {key: self._cp.asarray(snapshot[key], dtype=dtype) for key, dtype in dtypes.items()}

    def step(self, voltage_jumps_mv):
        ref, cp, c = self._reference, self._cp, self.config
        jumps = np.asarray(voltage_jumps_mv, dtype=np.float64)
        if jumps.shape != (self.n,) or not np.isfinite(jumps).all():
            raise ValueError("input jumps must be a finite vector matching the graph")
        if np.any(jumps[~ref._inputs] != 0):
            raise ValueError("voltage jumps are allowed only at declared input neurons")
        with self._device, self._stream:
            old, new = self._state, self._next
            device_jumps = cp.asarray(jumps)
            self._error.fill(0)
            shape = ((self.n + 255) // 256,), (256,)
            self._integrate(
                *shape,
                (
                    np.int32(self.n),
                    np.int64(self.tick),
                    old["voltage_mv"],
                    old["synaptic_mv"],
                    old["last_spike_tick"],
                    self._refs,
                    old["silenced"],
                    new["voltage_mv"],
                    new["synaptic_mv"],
                    new["spikes"],
                    self._available,
                    np.float64(c.rest_mv),
                    np.float64(c.threshold_mv),
                    np.float64(ref._em),
                    np.float64(ref._es),
                    np.float64(ref._coupling),
                ),
            )
            slot = (self.tick - ref._delay) % (ref._delay + 1)
            arriving = new["spikes"] if ref._delay == 0 else old["history"][slot]
            self._events(
                *shape,
                (
                    np.int32(self.n),
                    np.int64(self.tick),
                    self._row,
                    self._pre,
                    self._weights,
                    arriving,
                    device_jumps,
                    old["silenced"],
                    self._available,
                    new["spikes"],
                    old["rates_hz"],
                    old["last_spike_tick"],
                    new["voltage_mv"],
                    new["synaptic_mv"],
                    new["rates_hz"],
                    new["last_spike_tick"],
                    self._error,
                    np.float64(c.reset_mv),
                    np.float64(ref._rate_decay),
                    np.float64(1000 / c.dt_ms),
                ),
            )
            if int(self._error.get()[0]):
                raise ValueError("nonfinite synaptic state; check weights and inputs")
            cp.copyto(old["history"][self.tick % (ref._delay + 1)], new["spikes"])
            for key in new:
                old[key], new[key] = new[key], old[key]
            self.tick += 1

    def observe_selected(self, indices, fields=("voltage_mv", "synaptic_mv", "spikes", "rates_hz")):
        indices = self._reference._checked_indices(indices)
        with self._device, self._stream:
            selected = self._cp.asarray(indices, dtype=self._cp.int64)
            return {name: tuple(self._state[name][selected].get().tolist()) for name in fields}

    def set_silenced(self, indices, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        indices = self._reference._checked_indices(indices)
        with self._device, self._stream:
            self._state["silenced"][self._cp.asarray(indices, dtype=self._cp.int64)] = enabled

    def snapshot(self):
        with self._device, self._stream:
            state = {key: array.get().tolist() for key, array in self._state.items()}
        return {
            "engine": self.engine,
            "fingerprint": self._reference._fingerprint,
            "tick": self.tick,
            **state,
        }

    def restore(self, state):
        # Validation mutates only the CPU validator. Published device state is
        # replaced after all transfers complete, leaving it intact on failure.
        self._reference.restore(state)
        with self._device, self._stream:
            uploaded = self._upload(self._reference.snapshot())
            self._stream.synchronize()
            self._state = uploaded
            self.tick = self._reference.tick
