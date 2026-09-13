"""Public exceptions, kept independent of optional backend dependencies."""


class BackendUnavailableError(RuntimeError):
    """A backend cannot run because its runtime, device or compiler is unavailable."""


class CheckpointError(ValueError):
    """A checkpoint is invalid or uses an unsupported schema/dynamics revision."""
