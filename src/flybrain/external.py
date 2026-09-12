"""Explicit request/acknowledgement boundary for an external environment.

The engine owns its world. The controller owns neural time. Transport and engine
save files remain application responsibilities; this module uses only JSON data.
"""

from .brain import FlyBrain
from .config import finite_number, positive_int
from .drive import duration_ticks
from .session import LinearEncoder, RateReadout, detached


class ExternalController:
    """One outstanding action, exact retries, and acknowledged frame checkpoints.

    A failed engine must stop applying controls and restore/reset both participants.
    Never infer applied actions from requested actions. Not thread-safe: serialize
    calls in the transport. Sequence numbers are scoped to a transport session.
    """

    def __init__(self, brain, encoder, readout, *, period_ms=20):
        self.brain, self.encoder, self.readout = brain, encoder, readout
        self.period_ms = finite_number(period_ms, "period_ms")
        if not 1 <= self.period_ms <= 1000:
            raise ValueError("external period must be 1..1000 ms")
        self.ticks = duration_ticks(self.period_ms, brain.config.dt_ms)
        self.base_tick = brain.progress.tick
        self.next_seq = 0
        self.pending = None
        self.completed = []
        self.faulted = False

    def offer(self, seq, observation):
        """Integrate one observation; an identical pending retry never reintegrates."""
        seq = positive_int(seq, "seq", allow_zero=True)
        if not isinstance(observation, dict) or not all(isinstance(k, str) for k in observation):
            raise ValueError("observation must map feature names to finite numbers")
        observation = {k: finite_number(v, k) for k, v in observation.items()}
        if self.faulted:
            raise ValueError("controller faulted; restore/reset both participants")
        if self.pending is not None:
            if seq == self.pending["seq"] and observation == self.pending["observation"]:
                return detached(self.pending["response"])
            raise ValueError("acknowledge the pending action before the next observation")
        if seq != self.next_seq or len(self.completed) >= 10000:
            raise ValueError("unexpected sequence or recording limit")
        if self.brain.progress.tick != self.base_tick + seq * self.ticks:
            raise ValueError("brain clock changed outside controller")
        currents = dict(self.encoder.encode(observation))
        if currents:
            self.brain.neurons.resolve(tuple(currents))
        groups = {}
        for neuron, value in currents.items():
            value = finite_number(value, "current")
            if value:
                groups.setdefault(value, []).append(neuron)
        try:
            for amplitude, ids in groups.items():
                self.brain.drive.current(ids, amplitude=amplitude, duration_ms=self.period_ms)
            self.brain.advance(duration_ms=self.period_ms)
            action = {k: finite_number(v, k) for k, v in self.readout.decode(self.brain).items()}
        except Exception:
            self.faulted = True
            raise
        response = {
            "seq": seq,
            "brain_tick": self.brain.progress.tick,
            "duration_ms": self.period_ms,
            "requested": action,
        }
        self.pending = {
            "seq": seq,
            "observation": observation,
            "currents": currents,
            "response": response,
        }
        return detached(response)

    def acknowledge(self, seq, applied):
        """Commit actual engine controls, or return an identical completed receipt."""
        seq = positive_int(seq, "seq", allow_zero=True)
        if not isinstance(applied, dict):
            raise ValueError("applied must be a channel mapping")
        applied = {k: finite_number(v, k) for k, v in applied.items()}
        if self.completed and seq == self.completed[-1]["seq"]:
            if applied != self.completed[-1]["applied"]:
                raise ValueError("conflicting acknowledgement")
            return detached(self.completed[-1])
        if self.pending is None or seq != self.pending["seq"]:
            raise ValueError("no matching pending action")
        if applied.keys() != self.pending["response"]["requested"].keys():
            raise ValueError("applied channel names differ")
        frame = {**self.pending, "applied": applied}
        self.completed.append(detached(frame))
        self.pending = None
        self.next_seq += 1
        return detached(frame)

    def snapshot(self):
        """Save only at an acknowledged boundary; separately save the engine world."""
        if self.pending is not None or self.faulted:
            raise ValueError("checkpoint requires a healthy acknowledged boundary")
        if self.brain.progress.tick != self.base_tick + self.next_seq * self.ticks:
            raise ValueError("brain clock changed outside controller")
        return detached(
            {
                "format": "flybrain-external",
                "schema_version": 1,
                "period_ms": self.period_ms,
                "base_tick": self.base_tick,
                "next_seq": self.next_seq,
                "brain": self.brain.snapshot(),
                "encoder": self.encoder.snapshot(),
                "readout": self.readout.snapshot(),
            }
        )

    @classmethod
    def from_snapshot(cls, data, *, backend=None):
        if (
            data["format"] != "flybrain-external"
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
        ):
            raise ValueError("unsupported external checkpoint")
        result = cls(
            FlyBrain.from_snapshot(data["brain"], backend=backend),
            LinearEncoder.from_snapshot(data["encoder"]),
            RateReadout.from_snapshot(data["readout"]),
            period_ms=data["period_ms"],
        )
        result.base_tick = positive_int(data["base_tick"], "base_tick", allow_zero=True)
        result.next_seq = positive_int(data["next_seq"], "next_seq", allow_zero=True)
        result.snapshot()  # Verify the restored clock.
        return result
