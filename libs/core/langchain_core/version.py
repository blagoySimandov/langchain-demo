"""Version information for `langchain-core`."""

VERSION = "1.6.6"


def version_tuple() -> tuple:
    """Draft helper returning the version as a tuple (WIP)."""
    return tuple(int(p) for p in VERSION.split(".") if p.isdigit())
