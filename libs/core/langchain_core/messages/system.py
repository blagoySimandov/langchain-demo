"""System message."""

from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast, overload

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class SystemMessage(BaseMessage):
    """Message for priming AI behavior.

    The system message is usually passed in as the first of a sequence
    of input messages.

    Example:
        ```python
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a helpful assistant! Your name is Bob."),
            HumanMessage(content="What is your name?"),
        ]

        # Define a chat model and invoke it with the messages
        print(model.invoke(messages))
        ```
    """

    type: Literal["system"] = "system"
    """The type of the message (used for serialization)."""

    @overload
    def __init__(
        self,
        content: str | list[str | dict[Any, Any]],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        content: str | list[str | dict[Any, Any]] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None:
        """Specify `content` as positional arg or `content_blocks` for typing."""
        if content_blocks is not None:
            super().__init__(
                content=cast("list[str | dict[Any, Any]]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """System Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization)."""


def _extract_text(content: str | list[str | dict[Any, Any]]) -> str:
    """Extract the plain-text portion of message content.

    String content is returned as-is. List content is reduced to the string
    items and the `"text"` values of text-typed content blocks; all other
    blocks (e.g., images) are skipped.

    Args:
        content: Message content as stored on a `BaseMessage`.

    Returns:
        The concatenated text of the content, with list items joined by
            newlines.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _as_block_list(
    content: str | list[str | dict[Any, Any]],
) -> list[str | dict[Any, Any]]:
    """Normalize message content to list form.

    Args:
        content: Message content as stored on a `BaseMessage`.

    Returns:
        The content as a list of blocks. String content becomes a single
            text block; empty strings produce an empty list.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return list(content)


def merge_system_messages(
    messages: Sequence[SystemMessage],
    *,
    separator: str = "\n\n",
) -> SystemMessage:
    """Merge multiple system messages into a single system message.

    When every message carries plain string content, the result is a single
    string joined by `separator`. If any message carries list content
    (e.g., content blocks), the result is list content with all blocks
    concatenated in order.

    Metadata is combined with a shallow merge: `additional_kwargs` and
    `response_metadata` from later messages override earlier ones on key
    collisions. The first non-`None` `name` and `id` are kept.

    Args:
        messages: The system messages to merge, in order.
        separator: String inserted between string contents.

    Returns:
        A new `SystemMessage` containing the combined content.

    Raises:
        ValueError: If `messages` is empty.

    Example:
        ```python
        from langchain_core.messages import SystemMessage
        from langchain_core.messages.system import merge_system_messages

        merged = merge_system_messages(
            [
                SystemMessage("You are a helpful assistant."),
                SystemMessage("Always answer in French."),
            ]
        )
        merged.content
        # 'You are a helpful assistant.\\n\\nAlways answer in French.'
        ```
    """
    if not messages:
        msg = "merge_system_messages requires at least one message."
        raise ValueError(msg)
    if len(messages) == 1:
        return messages[0].model_copy()

    additional_kwargs: dict[str, Any] = {}
    response_metadata: dict[str, Any] = {}
    name: str | None = None
    id_: str | None = None
    for message in messages:
        additional_kwargs.update(message.additional_kwargs)
        response_metadata.update(message.response_metadata)
        if name is None:
            name = message.name
        if id_ is None:
            id_ = message.id

    content: str | list[str | dict[Any, Any]]
    if all(isinstance(message.content, str) for message in messages):
        content = separator.join(
            cast("str", message.content) for message in messages if message.content
        )
    else:
        blocks: list[str | dict[Any, Any]] = []
        for message in messages:
            blocks.extend(_as_block_list(message.content))
        content = blocks

    return SystemMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata,
        name=name,
        id=id_,
    )


def split_system_sections(
    message: SystemMessage,
    *,
    separator: str = "\n\n",
) -> list[str]:
    """Split a system message's text content into sections.

    Useful for inspecting or re-ordering prompts that were assembled from
    multiple instruction blocks (e.g., via `system_message_from_sections`).

    Args:
        message: The system message to split.
        separator: The delimiter between sections.

    Returns:
        The non-empty, stripped sections of the message text, in order.
    """
    text = _extract_text(message.content)
    return [section.strip() for section in text.split(separator) if section.strip()]


def system_message_from_sections(
    sections: Iterable[str],
    *,
    separator: str = "\n\n",
    **kwargs: Any,
) -> SystemMessage:
    """Build a system message from an ordered set of instruction sections.

    Blank or whitespace-only sections are dropped so callers can pass
    conditionally-built section lists without filtering first.

    Args:
        sections: Instruction blocks, in the order they should appear.
        separator: String inserted between sections.
        **kwargs: Additional fields forwarded to the `SystemMessage`
            constructor (e.g., `name`, `id`, `additional_kwargs`).

    Returns:
        A `SystemMessage` whose content is the joined sections.

    Example:
        ```python
        from langchain_core.messages.system import system_message_from_sections

        message = system_message_from_sections(
            [
                "You are a support agent for Acme Corp.",
                "Never share internal ticket IDs.",
                "",  # dropped
                "Respond in under 100 words.",
            ]
        )
        ```
    """
    cleaned = [section.strip() for section in sections]
    content = separator.join(section for section in cleaned if section)
    return SystemMessage(content=content, **kwargs)


def prepend_system_message(
    messages: Sequence[BaseMessage],
    system: SystemMessage | str,
    *,
    merge: bool = True,
    separator: str = "\n\n",
) -> list[BaseMessage]:
    """Return a new message list with a system message in the first position.

    If the sequence already starts with a `SystemMessage` and `merge` is
    `True`, the new system content is merged in front of the existing one
    rather than producing two system messages — many chat model providers
    accept at most one.

    Args:
        messages: The existing conversation messages. Not mutated.
        system: The system message (or raw string) to put first.
        merge: Whether to merge with an existing leading system message.
            If `False` and one exists, it is replaced.
        separator: Separator used when merging string contents.

    Returns:
        A new list of messages with the system message first.
    """
    if isinstance(system, str):
        system = SystemMessage(content=system)
    rest = list(messages)
    if rest and isinstance(rest[0], SystemMessage):
        existing = cast("SystemMessage", rest.pop(0))
        if merge:
            system = merge_system_messages([system, existing], separator=separator)
    return [system, *rest]
