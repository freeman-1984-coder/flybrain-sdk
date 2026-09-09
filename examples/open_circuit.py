"""Offline: direct currents, custom output channels and reversible intervention."""

from flybrain import FlyBrain

brain = FlyBrain.load()
brain.bind_readout({"brightness": ["motor.walk"], "sound": ["motor.jump"]})
brain.drive.current(["sense.food"], amplitude=2, duration_ms=100)
brain.advance(duration_ms=100)
print("Custom output:", brain.action().to_dict())
print("One cell:", brain.observe(["motor.walk"], fields=["rates_hz"]).to_dict())

# This is the artificial toy. Change the output mapping, not the graph.
blocked = FlyBrain.load()
blocked.bind_readout({"brightness": ["motor.walk"]})
blocked.intervene.silence(["relay.food"])
blocked.drive.current(["sense.food"], amplitude=2, duration_ms=100)
blocked.advance(duration_ms=100)
print("Relay silenced:", blocked.action().to_dict())
assert blocked.action()["brightness"] == 0
assert brain.action()["brightness"] > 0
