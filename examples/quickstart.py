"""Run after pip install -e .; no network, dataset download, or GPU."""

from pathlib import Path
from tempfile import TemporaryDirectory

from flybrain import FlyBrain


def main() -> None:
    brain = FlyBrain.load("toy", backend="cpu")
    print(f"No CUDA required | {len(brain.model.neuron_ids)} synthetic neurons")
    print("Sensory channels:", ", ".join(brain.sensory_channels))
    brain.stimulate("food", strength=0.9, duration_ms=200)
    brain.step(50)
    print("Food after 50 ms:", brain.action().to_dict())

    # Save while a stimulus is still active, then compare future trajectories.
    with TemporaryDirectory() as folder:
        checkpoint = brain.save(Path(folder) / "brain.checkpoint.json")
        restored = FlyBrain.restore(checkpoint)
        for _ in range(50):
            assert brain.step() == restored.step()
        print("Checkpoint continuation: identical for all 50 ticks")

    escape = FlyBrain.load()
    escape.stimulate("looming_left", duration_ms=100)
    escape.step(100)
    print("Threat on the left:", escape.action().to_dict())
    assert escape.action().turn_right > 0
    assert escape.action().jump > 0


if __name__ == "__main__":
    main()
