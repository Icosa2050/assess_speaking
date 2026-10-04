"""Serialize read-modify-write operations in the local backend process."""
from functools import wraps
import inspect
from threading import RLock

PREFERENCES_LOCK = RLock()


def synchronized_preferences(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with PREFERENCES_LOCK:
            return function(*args, **kwargs)
    wrapped.__signature__ = inspect.signature(function, eval_str=True)
    return wrapped
