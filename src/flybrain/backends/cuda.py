"""Experimental CuPy/CUDA LIF backend; import is safe without CuPy or a GPU.

Graph and state stay on the selected CUDA device. Each tick currently uploads
one host current vector and reads one error flag. This correctness-first path
is not the planned batched GPU scheduler and makes no speedup promise.
"""

from importlib import import_module
from math import ceil, exp

import numpy as np

from ..config import LIFConfig
from ..errors import BackendUnavailableError
from ..model import Connectome
from .base import Backend, SimulationState
from .cpu import CPUBackend

# One thread owns a postsynaptic row. Stable input order retains parallel edges
# and avoids nondeterministic floating-point atomic accumulation. Disable FMA to
# preserve the reference's separate multiplication and addition operations.
_KERNEL = r"""
extern "C" __global__ void lif_tick(
    int n, const long long* row, const int* pre, const double* weights,
    const double* current, const double* old_v, const unsigned char* old_spikes,
    const long long* old_refr, const double* old_rates, const unsigned char* silent,
    double* v, unsigned char* spikes, long long* refr, double* rates, int* error,
    double rest, double reset, double threshold, double leak, double rate_decay,
    double rate_hz, long long hold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double synaptic = 0.0;
    for (long long e = row[i]; e < row[i+1]; ++e)
        synaptic += weights[e] * (double)old_spikes[pre[e]];
    bool available = old_refr[i] == 0 && !silent[i];
    double value = available
        ? rest + (old_v[i] - rest) * leak + (current[i] + synaptic) * (1.0 - leak)
        : reset;
    if (!isfinite(value)) atomicExch(error, 1);
    unsigned char spike = available && value >= threshold;
    v[i] = spike ? reset : value;
    spikes[i] = spike;
    refr[i] = spike ? hold : (old_refr[i] > 0 ? old_refr[i] - 1 : 0);
    rates[i] = old_rates[i] * rate_decay + ((double)spike * rate_hz) * (1.0-rate_decay);
}
"""


def _load_cupy():
    try:
        cp = import_module("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no visible NVIDIA device")
        if cp.cuda.runtime.is_hip:
            raise RuntimeError("ROCm is not supported by this CUDA backend")
        return cp
    except Exception as error:
        raise BackendUnavailableError(
            "CUDA requires a working NVIDIA driver, visible GPU and compatible CuPy. "
            "Install the appropriate cupy-cuda12x or cupy-cuda13x wheel. "
            f"CPU remains available without them. Cause: {error}"
        ) from error


def cuda_available() -> bool:
    """Probe optional dependency and device; does not certify kernel conformance."""
    try:
        _load_cupy()
        return True
    except BackendUnavailableError:
        return False


class CUDABackend(Backend):
    name = "cuda"

    def __init__(self, model: Connectome, config: LIFConfig) -> None:
        cp = _load_cupy()
        self._cp, self._model, self.config = cp, model, config
        self.n = len(model.neuron_ids)
        if self.n > np.iinfo(np.int32).max:
            raise ValueError("CUDA reference backend supports at most int32 neuron indices")
        self._tick = 0
        self._leak = exp(-config.dt_ms / config.tau_ms)
        self._decay = exp(-config.dt_ms / config.rate_tau_ms)
        self._hold = ceil(config.refractory_ms / config.dt_ms)
        post = np.array([e.post for e in model.synapses], dtype=np.int32)
        order = np.argsort(post, kind="stable")
        rows = np.empty(self.n + 1, dtype=np.int64)
        rows[0] = 0
        rows[1:] = np.cumsum(np.bincount(post, minlength=self.n))
        try:
            self._device = cp.cuda.Device()
            self._stream = cp.cuda.Stream(non_blocking=True)
            with self._device, self._stream:
                self._row = cp.asarray(rows)
                self._pre = cp.asarray(np.array([e.pre for e in model.synapses], np.int32)[order])
                self._weights = cp.asarray(
                    np.array([e.weight for e in model.synapses], np.float64)[order]
                )
                self._voltage = cp.full(self.n, config.rest, dtype=cp.float64)
                self._spikes = cp.zeros(self.n, dtype=cp.bool_)
                self._refractory = cp.zeros(self.n, dtype=cp.int64)
                self._rates = cp.zeros(self.n, dtype=cp.float64)
                self._silenced = cp.zeros(self.n, dtype=cp.bool_)
                self._next = [cp.empty_like(x) for x in self._state_arrays()]
                self._error = cp.zeros(1, dtype=cp.int32)
                self._kernel = cp.RawKernel(_KERNEL, "lif_tick", options=("--fmad=false",))
                self._kernel.compile()
        except Exception as error:
            raise BackendUnavailableError(
                f"CUDA initialization or compilation failed: {error}"
            ) from error

    def _state_arrays(self):
        return self._voltage, self._spikes, self._refractory, self._rates

    @property
    def tick(self) -> int:
        return self._tick

    def step(self, current: np.ndarray) -> None:
        current = np.ascontiguousarray(current, dtype=np.float64)
        if current.shape != (self.n,) or not np.isfinite(current).all():
            raise ValueError("current must be a finite vector of length neuron_count")
        with self._device, self._stream:
            self._error.fill(0)
            self._kernel(
                ((self.n + 127) // 128,),
                (128,),
                (
                    np.int32(self.n),
                    self._row,
                    self._pre,
                    self._weights,
                    self._cp.asarray(current),
                    *self._state_arrays(),
                    self._silenced,
                    *self._next,
                    self._error,
                    np.float64(self.config.rest),
                    np.float64(self.config.reset),
                    np.float64(self.config.threshold),
                    np.float64(self._leak),
                    np.float64(self._decay),
                    np.float64(1000.0 / self.config.dt_ms),
                    np.int64(self._hold),
                ),
            )
            # Synchronize only a scalar error flag before committing this tick.
            if self._error.get()[0]:
                raise ValueError("nonfinite voltage; reduce synapse weights or stimulus current")
            old = self._state_arrays()
            self._voltage, self._spikes, self._refractory, self._rates = self._next
            self._next = old
            self._tick += 1

    def observe_selected(self, indices: tuple, fields: tuple) -> dict:
        arrays = {"voltage": self._voltage, "spikes": self._spikes, "rates_hz": self._rates}
        with self._device, self._stream:
            selected = self._cp.asarray(indices, dtype=self._cp.int64)
            return {k: tuple(arrays[k][selected].get().tolist()) for k in fields}

    def set_silenced(self, indices: tuple, enabled: bool) -> None:
        with self._device, self._stream:
            self._silenced[self._cp.asarray(indices, dtype=self._cp.int64)] = enabled

    def observe(self) -> SimulationState:
        values = self.observe_selected(tuple(range(self.n)), ("voltage", "spikes", "rates_hz"))
        return SimulationState(self.tick, self.tick * self.config.dt_ms, **values)

    def snapshot(self) -> dict:
        with self._device, self._stream:
            return {
                "tick": self.tick,
                "voltage": self._voltage.get().tolist(),
                "spikes": self._spikes.get().tolist(),
                "refractory": self._refractory.get().tolist(),
                "rates_hz": self._rates.get().tolist(),
                "silenced": self._silenced.get().tolist(),
            }

    def restore(self, state: dict) -> None:
        # Share the authoritative CPU validator, including schema-1 compatibility.
        # This temporarily allocates a CPU graph/state only during restoration.
        checked = CPUBackend(self._model, self.config)
        checked.restore(state)
        cp = self._cp
        with self._device, self._stream:
            values = [
                cp.asarray(x)
                for x in (
                    checked._voltage,
                    checked._spikes,
                    checked._refractory,
                    checked._rates,
                    checked._silenced,
                )
            ]
            self._stream.synchronize()
            self._voltage, self._spikes, self._refractory, self._rates, self._silenced = values
            self._tick = checked.tick
