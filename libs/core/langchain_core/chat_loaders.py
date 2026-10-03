"""Chat loaders."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from langchain_core.chat_sessions import ChatSession


class BaseChatLoader(ABC):
    """Base class for chat loaders."""

    @abstractmethod
    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy load the chat sessions.

        Returns:
            An iterator of chat sessions.
        """

    def load(self) -> list[ChatSession]:
        """Eagerly load the chat sessions into memory.

        Returns:
            A list of chat sessions.
        """
        return list(self.lazy_load())


def count_sessions(loader: BaseChatLoader) -> int:
    """Draft helper counting chat sessions (WIP)."""
    return sum(1 for _ in loader.lazy_load())
