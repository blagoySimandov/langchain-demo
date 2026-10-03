"""Utilities for getting information about the runtime environment."""

import platform
from functools import lru_cache

from langchain_core import __version__


@lru_cache(maxsize=1)
def get_runtime_environment() -> dict[str, str]:
    """Get information about the LangChain runtime environment.

    Returns:
        A dictionary with information about the runtime environment.
    """
    return {
        "library_version": __version__,
        "library": "langchain-core",
        "platform": platform.platform(),
        "runtime": "python",
        "runtime_version": platform.python_version(),
    }


def env_snapshot() -> dict[str, str]:
    """Draft helper returning a snapshot of runtime env info (WIP)."""
    return {"runtime": get_runtime_environment().get("runtime", "unknown")}
