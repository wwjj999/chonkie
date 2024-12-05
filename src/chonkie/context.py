"""Context class for storing contextual information for chunk refinement.

This class is used to store contextual information for chunk refinement.
It can represent context that comes before a chunk at the moment.

By default, the context has no start and end indices, meaning it is not
bound to any specific text. The start and end indices are only set if the
context is part of the same text as the chunk.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Context:
    """A dataclass representing contextual information for chunk refinement.

    This class stores text and token count information that can be used to add
    context to chunks during the refinement process. It can represent context
    that comes before or after a chunk.

    Attributes:
        text (str): The context text
        token_count (int): Number of tokens in the context text
        start_index (Optional[int]): Starting position of context in original text
        end_index (Optional[int]): Ending position of context in original text

    Example:
        context = Context(
            text="This is some context.",
            token_count=5,
            start_index=0,
            end_index=20
        )

    """

    text: str
    token_count: int
    start_index: Optional[int] = None
    end_index: Optional[int] = None

    def __post_init__(self):
        """Validate the Context attributes after initialization."""
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.token_count, int):
            raise ValueError("token_count must be an integer")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if (
            self.start_index is not None
            and self.end_index is not None
            and self.start_index > self.end_index
        ):
            raise ValueError("start_index must be less than or equal to end_index")

    def __len__(self) -> int:
        """Return the length of the context text."""
        return len(self.text)

    def __str__(self) -> str:
        """Return a string representation of the Context."""
        return self.text

    def __repr__(self) -> str:
        """Return a detailed string representation of the Context."""
        return (
            f"Context(text='{self.text}', token_count={self.token_count}, "
            f"start_index={self.start_index}, end_index={self.end_index})"
        )
