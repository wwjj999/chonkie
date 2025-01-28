from typing import Optional, Union, List
import re

from chonkie.chef.base import BaseChef
from chonkie.chef.patterns import Abbreviations, UnicodeReplacements
class TextChef(BaseChef):
    """A chef that handles basic text processing.
    
    This chef handles basic text files and performs common text cleaning operations
    like removing extra whitespace, normalizing line endings, etc.
    """

    def __init__(self, 
                 whitespace: bool = True,
                 newlines: bool = True,
                 abbreviations: bool = True,
                 ellipsis: bool = True, 
                 sentence_endings: str = '.!?;:') -> None:
        """Initialize the TextChef with common text file extensions."""
        extensions = ['.txt', '.md', '.rst', '.text']
        super().__init__(extensions=extensions)

        # Initialize the flags
        self._enable_whitespace = whitespace
        self._enable_newlines = newlines
        self._enable_abbreviations = abbreviations
        self._enable_ellipsis = ellipsis

        # Initialize the sentence endings
        self._sentence_endings = sentence_endings
        
        # Initialize the patterns
        self._abbreviations = Abbreviations.all() if abbreviations else set()
        self._unicode_replacements = UnicodeReplacements()

        # Compiling the regex patterns
        self._ellipsis_pattern = re.compile(r'\.{3,}')
        self._newline_pattern = re.compile(r'\n+')

    def _handle_abbreviations(self, text: str) -> str:
        """Replace the fullstop in abbreviations with a dot leader."""
        for abbreviation in self._abbreviations:
            new_abbreviation = abbreviation.replace('.', self._unicode_replacements.DOT_LEADER)
            text = re.sub(abbreviation, new_abbreviation, text)
        return text

    def _replace_ellipsis(self, text: str) -> str:
        """Replace ellipsis with Unicode ellipsis character.
        
        Args:
            text: Input text
            
        Returns:
            Text with ellipsis replaced

        """
        # Replace any sequence of 3 or more dots with ellipsis character
        return self._ellipsis_pattern.sub(self._unicode_replacements.ELLIPSIS, text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace

        """
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        return text
    
    def _normalize_newlines(self, text: str) -> str:
        """Normalize newlines in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized newlines

        """
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize more than one newline as double newlines
        text = self._newline_pattern.sub('\n\n', text)

        # Remove empty lines while preserving paragraph structure
        lines = [line.strip() for line in text.split('\n')]
        result = []
        
        for i, line in enumerate(lines):
            if not line:  # Skip empty lines
                continue
                
            # If this isn't the first line and previous line doesn't end with 
            # sentence ending punctuation, join with a space instead of newline
            if result and not result[-1][-1] in self._sentence_endings:
                result[-1] = result[-1] + ' ' + line
            else:
                result.append(line)
                
        return '\n'.join(result)

    def clean(self, text: str) -> str:
        r"""Clean the text by performing basic text processing operations.
        
        A common function where one can enable/disable the operations supported by the chef.

        Args:
            text: The text to clean.
        
        Returns:
            The cleaned text.

        """
        if not text:
            return text

        # Normalize whitespace
        if self._enable_whitespace:
            text = self._normalize_whitespace(text)

        # Normalize newlines
        if self._enable_newlines:
            text = self._normalize_newlines(text)

        # Replace ellipsis
        if self._enable_ellipsis:
            text = self._replace_ellipsis(text)

        # Replace abbreviations
        if self._enable_abbreviations:
            text = self._handle_abbreviations(text)

        return text