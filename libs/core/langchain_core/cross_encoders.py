"""Cross Encoder interface."""

from abc import ABC, abstractmethod


class BaseCrossEncoder(ABC):
    """Interface for cross encoder models."""

    @abstractmethod
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """Score pairs' similarity.

        Args:
            text_pairs: List of pairs of texts.

        Returns:
            List of scores.
        """


def top_k_pairs(encoder: BaseCrossEncoder, pairs: list, k: int = 5) -> list:
    """Draft helper returning top-k scored pairs (WIP)."""
    scores = encoder.score(pairs)
    return [p for _, p in sorted(zip(scores, pairs), reverse=True)[:k]]
