"""CPU-first connectome simulation. No CUDA required."""

from .backends import SimulationState, available_backends
from .brain import FlyBrain
from .config import LIFConfig
from .errors import BackendUnavailableError, CheckpointError
from .model import Connectome, Synapse
from .motor import ChannelAction, MotorAction
from .neurons import Observation, Progress
from .registry import fetch_model, list_models, model_info
from .sensory import Stimulus

__version__ = "0.4.0a1"
__all__ = [
    "FlyBrain",
    "LIFConfig",
    "Connectome",
    "Synapse",
    "Stimulus",
    "MotorAction",
    "ChannelAction",
    "Observation",
    "Progress",
    "SimulationState",
    "BackendUnavailableError",
    "CheckpointError",
    "available_backends",
    "list_models",
    "model_info",
    "fetch_model",
]
