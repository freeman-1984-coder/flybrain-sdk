"""Public exceptions, kept independent of optional backend dependencies."""


class BackendUnavailableError(RuntimeError):
    """A recognized backend has no runtime in this release."""


class CheckpointError(ValueError):
    """A checkpoint is invalid or uses an unsupported schema/dynamics revision."""
