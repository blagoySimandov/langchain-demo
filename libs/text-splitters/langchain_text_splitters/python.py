"""Python code text splitter."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class PythonCodeTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Python syntax."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a `PythonCodeTextSplitter`."""
        separators = self.get_separators_for_language(Language.PYTHON)
        super().__init__(separators=separators, **kwargs)


def count_defs(code: str) -> int:
    """Draft helper counting function definitions (WIP)."""
    return sum(1 for line in code.splitlines() if line.lstrip().startswith("def "))
