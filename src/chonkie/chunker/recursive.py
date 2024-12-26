from dataclasses import dataclass
from typing import Any, List, Union, Optional
from itertools import accumulate
from chonkie.types import Chunk
from bisect import bisect_left
from functools import lru_cache
@dataclass
class RecursiveLevel:
    """Configuration for a single level of recursive chunking.

    Attributes:
        delimiters: The delimiters to use for the level. If None, that level will use tokens to determine chunk boundaries.
        whitespace: Whether to use whitespace as a delimiter.

    """

    delimiters: Union[List[str], str, None] = None
    whitespace: bool = False

    def __repr__(self) -> str:
        """Get a string representation of the recursive level."""
        return f"RecursiveLevel(delimiters={self.delimiters}, whitespace={self.whitespace})"
    
    def __str__(self) -> str:
        """Get a string representation of the recursive level."""
        return f"RecursiveLevel(delimiters={self.delimiters}, whitespace={self.whitespace})"

@dataclass
class RecursiveRules: 
    """Collection of rules for recursive chunking."""

    levels: Union[List[RecursiveLevel], RecursiveLevel, None] = None

    def __post_init__(self):
        """Initialize the recursive rules if not already initialized."""
        # Set default levels if not already initialized
        if self.levels is None:
            # First level should be paragraphs
            paragraph_level = RecursiveLevel(delimiters=["\n\n", "\n", "\r\n"],
                                              whitespace=False)
            # Second level should be sentences
            sentence_level = RecursiveLevel(delimiters=[".", "?", "!"],
                                            whitespace=False)
            # Third level should be words
            word_level = RecursiveLevel(delimiters=[" "],
                                        whitespace=True)
            # Fourth level should be tokens
            # NOTE: When delimiters is None, the level will use tokens to determine chunk boundaries.
            token_level = RecursiveLevel(delimiters=None,
                                        whitespace=False)
            self.levels = [paragraph_level, sentence_level, word_level, token_level]
    
    def __iter__(self):
        """Iterate over the levels."""
        return iter(self.levels)
    
    def __getitem__(self, index: int) -> RecursiveLevel:
        """Get a level by index."""
        return self.levels[index]

    def __len__(self) -> int:
        """Get the number of levels."""
        return len(self.levels)
    
    def __repr__(self) -> str:
        """Get a string representation of the recursive rules."""
        return f"RecursiveRules(levels={self.levels})"
    
    def __str__(self) -> str:
        """Get a string representation of the recursive rules."""
        return f"RecursiveRules(levels={self.levels})"


@dataclass
class RecursiveChunk(Chunk):
    """A Chunk with a level attribute."""

    level: Union[int, None] = None

    def __repr__(self) -> str:
        """Get a string representation of the recursive chunk."""
        return (f"RecursiveChunk(text={self.text}, "
                f"start_index={self.start_index}, "
                f"end_index={self.end_index}, "
                f"token_count={self.token_count}, "
                f"level={self.level})")
    
    def __str__(self) -> str:
        """Get a string representation of the recursive chunk."""
        return (f"RecursiveChunk(text={self.text}, "
                f"start_index={self.start_index}, "
                f"end_index={self.end_index}, "
                f"token_count={self.token_count}, "
                f"level={self.level})")

class RecursiveChunker:
    """Chunker that uses recursive rules to chunk text.
    
    Attributes:
        rules: The rules to use for chunking.
        chunk_size: The size of the chunks to return.
        
    """

    def __init__(self,
                 tokenizer: Union[str, Any] = "gpt2",
                 rules: RecursiveRules = RecursiveRules(),
                 chunk_size: int = 512
                 ) -> None:
        """Initialize the recursive chunker.

        Args:
            tokenizer: The tokenizer to use for encoding/decoding.
            rules: The rules to use for chunking.
            chunk_size: The size of the chunks to return.
        
        """
        super().__init__(tokenizer)
        self.rules = rules
        self.chunk_size = chunk_size

    def _split_text(self,
                    text: str,
                    rule: RecursiveLevel, 
                    sep: str = "🦛") -> List[str]:
        """Split the text into chunks using the delimiters."""
        # At every delimiter, replace it with the sep   
        if rule.delimiters:
            for delimiter in rule.delimiters:
                text = text.replace(delimiter, delimiter + sep)

            # Split the text at the sep
            splits = [s for s in text.split(sep) if s != ""]
        elif rule.whitespace:
            splits = self._split_at_whitespace(text)
        else:
            splits = self._split_at_tokens(text)

        #NOTE: Usually some splits will be very short and not meaningful, but 
        # we can at this point assume that the merge will fix this later on. 
        return splits
    
    def _split_at_whitespace(self,
                             text: str) -> List[str]:
        """Split the text at whitespace."""
        return text.split()
    
    def _split_at_tokens(self,
                         text: str) -> List[str]:
        """Split the text at tokens."""
        tokens = self._encode(text)

        # Split the tokens at the chunk size
        token_splits = [tokens[i:i+self.chunk_size] for i in range(0, len(tokens), self.chunk_size)]
        
        # Decode the tokens back to text
        splits = self._decode_batch(token_splits)
        return splits

    def _merge_splits(self,
                       splits: List[str],
                       token_counts: List[int],
                       combine_with_whitespace: bool = False) -> List[str]:
        """Merge splits that are too short."""
        # If there are no splits or token counts, return an empty list
        if not splits or not token_counts:
            return []
        
        # If the number of splits and token counts does not match, raise an error
        if len(splits) != len(token_counts):
            raise ValueError("The number of splits and token counts does not match.")

        # Usually the splits can be smaller than the chunk size; if not, 
        # we can just return the splits
        if all(tc > self.chunk_size for tc in token_counts):
            return splits
        
        # If the splits are too short, merge them
        merged = []

        if not combine_with_whitespace:
            cumulative_token_counts = list(accumulate(token_counts, lambda x, y: x + y))
        else:
            cumulative_token_counts = list(accumulate(token_counts, lambda x, y: x + y + 1)) # Add 1 for the whitespace

        current_index = 0
        merged_token_counts = []
        # Use bisect_left to find the index to merge at 
        while current_index < len(splits):
            current_token_count = cumulative_token_counts[current_index]
            required_token_count = current_token_count + self.chunk_size

            # Find the index to merge at
            index = bisect_left(cumulative_token_counts, required_token_count, lo=current_index)

            # Merge the splits at the index
            if combine_with_whitespace:
                merged.append(" ".join(splits[current_index:index]))
            else:
                merged.append("".join(splits[current_index:index]))

            # Add the token count of the merged split
            merged_token_counts.append(cumulative_token_counts[index] - current_token_count)
            
            # Update the current index
            current_index = index

        return merged, merged_token_counts

    @lru_cache(maxsize=4096)
    def _get_token_count(self,
                         text: str) -> int:
        """Get the token count of the text."""
        CHARS_PER_TOKEN = 6.0  # Avg. char per token for llama3 is b/w 6-7
        estimate = max(1, len(text) // CHARS_PER_TOKEN)
        if estimate > self.chunk_size:
            return self.chunk_size + 1
        else:
            return self._count_tokens(text)

    def _create_chunk(self,
                      text: str, 
                      token_count: int, 
                      level: int,
                      full_text: Optional[str] = None) -> RecursiveChunk:
        """Create a chunk."""
        if full_text is None:
            full_text = text

        start_index = full_text.index(text)
        end_index = start_index + len(text)
        return RecursiveChunk(text=text,
                     start_index=start_index,
                     end_index=end_index,
                     token_count=token_count,
                     level=level)

    def _recursive_chunk(self, 
                        text: str,
                        level: int = 0, 
                        full_text: Optional[str] = None) -> List[RecursiveChunk]:
        """Recursive chunking logic."""
        # First make sure the text is not empty
        if not text:
            return []
        
        # If level is out of bounds, return the text as a chunk
        if level >= len(self.rules):
            return [self._create_chunk(text, self._get_token_count(text), level, full_text)]

        # If full_text is not provided, use the text
        if full_text is None:
            full_text = text
        
        # Get the current level
        rule = self.rules[level]

        # Split the text at the current level
        splits = self._split_text(text, rule)

        # Get the token counts for the splits
        token_counts = [self._get_token_count(split) for split in splits]

        # Merge the splits
        if rule.delimiters is None and not rule.whitespace:
            # If the level is at tokens, we don't need to merge the splits
            merged, merged_token_counts = splits, token_counts

        elif rule.delimiters is None and rule.whitespace:
            # If the level is at tokens and whitespace, we need to merge the splits
            merged, merged_token_counts = self._merge_splits(splits, token_counts, combine_with_whitespace=True)
        else:
            # If the level is at sentences, we need to merge the splits
            merged, merged_token_counts = self._merge_splits(splits, token_counts, combine_with_whitespace=False)

        # Recursively chunk the merged splits if they are too long
        chunks = []
        for split, token_count in zip(merged, merged_token_counts):
            if token_count > self.chunk_size:
                chunks.extend(self._recursive_chunk(split, level + 1, full_text))
            else:
                if rule.delimiters is None and not rule.whitespace:
                    decoded_text = self._decode(self._encode(full_text))
                    chunks.append(self._create_chunk(split, token_count, level, decoded_text))
                else:
                    chunks.append(self._create_chunk(split, token_count, level, full_text))

        return chunks


    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the text."""
        return self._recursive_chunk(text)
    