"""Optional, reproducible raw-data importers. Arrow is loaded only when needed."""

from .malecns import import_malecns

__all__ = ["import_malecns"]
