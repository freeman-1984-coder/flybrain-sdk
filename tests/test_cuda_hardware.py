"""Run on actual CUDA only. FLYBRAIN_REQUIRE_CUDA=1 makes absence a failure."""

import os
from pathlib import Path

import numpy as np
import pytest

from flybrain import BackendUnavailableError, Connectome, FlyBrain, LIFConfig, Synapse
from flybrain.backends import CPUBackend
from flybrain.backends.cuda import CUDABackend, _load_cupy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gpu():
    try:
        return _load_cupy()
    except BackendUnavailableError as error:
        if os.environ.get("FLYBRAIN_REQUIRE_CUDA") == "1":
            pytest.fail(f"Required CUDA hardware is unavailable: {error}")
        pytest.skip(f"Actual CUDA device required: {error}")


def assert_state(a, b):
    assert a.tick == b.tick
    assert a.time_ms == b.time_ms
    assert a.spikes == b.spikes
    np.testing.assert_allclose(a.voltage, b.voltage, atol=1e-10, rtol=0)
    np.testing.assert_allclose(a.rates_hz, b.rates_hz, atol=1e-10, rtol=0)


@pytest.mark.parametrize("model", ["toy", "real", "parallel-inhibitory"])
@pytest.mark.parametrize("dt", [0.5, 1.0])
def test_every_cell_tick_matches_cpu(gpu, model, dt):
    if model == "parallel-inhibitory":
        rng = np.random.default_rng(41)
        edges = tuple(
            Synapse(int(a), int(b), float(w))
            for a, b, w in zip(
                rng.integers(0, 257, 4096), rng.integers(0, 257, 4096), rng.uniform(-8, 12, 4096)
            )
        )
        graph = Connectome(
            "random signed graph",
            tuple(map(str, range(257))),
            edges + (Synapse(0, 1, 8), Synapse(0, 1, 8)),
            {},
            {},
            {},
        )
    else:
        graph = FlyBrain.load(
            "toy" if model == "toy" else ROOT / "models/male-cns-escape-v1/model.json"
        ).model
    config = LIFConfig(dt_ms=dt)
    cpu, cuda = CPUBackend(graph, config), CUDABackend(graph, config)
    rng = np.random.default_rng(17)
    currents = rng.uniform(-1, 4, (180, len(graph.neuron_ids)))
    for tick, current in enumerate(currents):
        if tick in (31, 79):
            cpu.set_silenced((0, 1), tick == 31)
            cuda.set_silenced((0, 1), tick == 31)
        cpu.step(current)
        cuda.step(current)
        assert_state(cpu.observe(), cuda.observe())
        if tick == 50:
            # GPU->CPU->GPU with active silencing and refractory state.
            cpu.restore(cuda.snapshot())
            cuda.restore(cpu.snapshot())


def test_pending_inputs_and_custom_readout_cross_device(gpu, tmp_path):
    brain = FlyBrain.load()
    brain.bind_readout({"flash": ["motor.walk"]})
    brain.stimulate("food", duration_ms=120)
    brain.drive.current(["relay.food"], amplitude=-1, duration_ms=50)
    brain.intervene.silence(["motor.jump"])
    brain.step(17)
    path = brain.save(tmp_path / "cpu.json")
    cuda = FlyBrain.restore(path, backend="cuda")
    for _ in range(150):
        assert_state(brain.step(), cuda.step())
        assert cuda.action().to_dict() == pytest.approx(brain.action().to_dict(), abs=1e-10)
    restored = FlyBrain.restore(cuda.save(tmp_path / "gpu.json"), backend="cpu")
    assert_state(restored.step(20), cuda.step(20))


def test_invalid_tick_and_restore_leave_state_unchanged(gpu):
    graph = Connectome(
        "overflow", ("a", "b"), (Synapse(0, 1, 1e308), Synapse(0, 1, 1e308)), {}, {}, {}
    )
    cuda = CUDABackend(graph, LIFConfig())
    cuda.step(np.array([20.0, 0.0]))
    before = cuda.snapshot()
    for current in (np.array([np.nan, 0]), np.zeros(3), np.zeros(2)):
        with pytest.raises(ValueError):
            cuda.step(current)
        assert cuda.snapshot() == before
    bad = dict(before, refractory=[999, 0])
    with pytest.raises(ValueError):
        cuda.restore(bad)
    assert cuda.snapshot() == before


def test_selected_observation_is_detached(gpu):
    cuda = CUDABackend(FlyBrain.load().model, LIFConfig())
    current = np.arange(cuda.n * 2, dtype=np.float64)[::-2]
    cuda.step(current)
    reference = CPUBackend(FlyBrain.load().model, LIFConfig())
    reference.step(current)
    assert_state(reference.observe(), cuda.observe())
    values = cuda.observe_selected((1, 4), ("voltage", "rates_hz"))
    before = values["voltage"]
    cuda.step(np.full(cuda.n, 20.0))
    assert values["voltage"] == before
    assert len(values["voltage"]) == 2


def test_backend_owns_stream_ordering(gpu):
    cpu = CPUBackend(FlyBrain.load().model, LIFConfig())
    cuda = CUDABackend(FlyBrain.load().model, LIFConfig())
    for _ in range(10):
        with gpu.cuda.Stream(non_blocking=True):
            cuda.step(np.full(cuda.n, 2.0))
            cuda.set_silenced((0,), True)
        cpu.step(np.full(cpu.n, 2.0))
        cpu.set_silenced((0,), True)
        assert_state(cpu.observe(), cuda.observe())


@pytest.mark.parametrize("model", ["toy", ROOT / "models/male-cns-escape-v1/model.json"])
def test_feedback_sessions_and_external_cross_device_restore(gpu, model):
    from flybrain.demos import DodgeArena, make_demo
    from flybrain.external import ExternalController
    from flybrain.session import Session, compare

    expected = make_demo("dodge", model=model)
    accelerated = make_demo("dodge", model=model, backend="cuda")
    demo = make_demo("dodge", model=model, backend="cuda")
    external = ExternalController(demo.brain, demo.encoder, demo.readout)
    world = demo.environment
    for seq in range(60):
        obs = world.observe()
        action = external.offer(seq, obs)
        tick = external.brain.progress.tick
        assert external.offer(seq, obs) == action
        assert external.brain.progress.tick == tick
        applied = world.apply(action["requested"], 20)
        external.acknowledge(seq, applied)
        reference = expected.step()
        frame = accelerated.step()
        for controls in (action["requested"], frame.requested):
            compare(controls, reference.requested)
        compare(applied, reference.applied)
        compare(frame.applied, reference.applied)
        compare(world.snapshot(), reference.environment)
        compare(frame.environment, reference.environment)
        assert_state(expected.brain.state, external.brain.state)
        assert_state(expected.brain.state, accelerated.brain.state)
        if seq in (29, 44):
            backend = "cpu" if seq == 29 else "cuda"
            external = ExternalController.from_snapshot(external.snapshot(), backend=backend)
            accelerated = Session.from_snapshot(
                accelerated.snapshot(),
                environment_factory=DodgeArena.from_snapshot,
                backend=backend,
            )
            assert external.brain.snapshot()["backend"] == backend
            assert accelerated.brain.snapshot()["backend"] == backend
