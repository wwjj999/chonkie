from typing import Optional, Union, List

from chonkie.chef.base import BaseChef

class TextChef(BaseChef):
    """A chef that handles basic text processing.
    
    This chef handles basic text files and performs common text cleaning operations
    like removing extra whitespace, normalizing line endings, etc.
    """

    def __init__(self) -> None:
        """Initialize the TextChef with common text file extensions."""
        super().__init__(extensions=['.txt', '.md', '.rst', '.text'])

    def _normalize_spaces(self, text: str) -> str:
        """Normalize spaces in the text."""
        return ' '.join(text.split())
    
    def _remove_empty_lines(self, text: str) -> str:
        """Remove empty lines from the text and preserve paragraph breaks."""
        # Split the text into lines
        lines = text.split('\n')

        # Remove empty lines
        lines = [line for line in lines if line.strip()]

        # Join the lines back together
        return '\n'.join(lines)

    def _normalize_line_endings(self, text: str) -> str:
        r"""Normalize line endings to \n."""
        return text.replace('\r\n', '\n').replace('\r', '\n')

    def clean(self, text: str) -> str:
        r"""Clean the text by performing basic text processing operations.
        
        Operations performed:
        - Normalize line endings to \n
        - Remove redundant empty lines
        - Strip whitespace from start/end
        - Replace multiple spaces with single space
        
        Args:
            text: The text to clean.
        
        Returns:
            The cleaned text.

        """
        if not text:
            return text
            
        # Normalize line endings
        text = self._normalize_line_endings(text)

        # Remove redundant empty lines
        text = self._remove_empty_lines(text)

        # Replace multiple spaces with single space
        text = self._normalize_spaces(text)
        
        return text