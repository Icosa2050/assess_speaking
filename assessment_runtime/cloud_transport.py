"""Process-local hook for backend-owned cloud requests; never contains secrets."""
from contextvars import ContextVar

completion = ContextVar('cloud_completion', default=None)
provenance = ContextVar('cloud_provenance', default=None)
