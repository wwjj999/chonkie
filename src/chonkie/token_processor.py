"""Factory class for creating and managing tokenizers.

This factory class is used to create and manage tokenizers for the Chonkie
package. It provides a simple interface for initializing, encoding, decoding,
and counting tokens using different tokenizer backends.

This is used in the Chunker and Refinery classes to ensure consistent tokenization
across different parts of the pipeline.
"""

from typing import TYPE_CHECKING
import importlib
import inspect
from typing import Any, Callable, List, Union

if TYPE_CHECKING:
    import tiktoken
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer


class TokenProcessor:
    """Handles tokenization operations using various backends.
    
    This class is used to handle tokenization operations using various backends.
    It provides a simple interface for initializing, encoding, decoding,
    and counting tokens using different tokenizer backends.

    Args:
        tokenizer_or_token_counter (Union[str, Callable, "tiktoken.Encoding", "Tokenizer"]):
            The tokenizer or token counter to use.

    """
    
    def __init__(self,
                tokenizer_or_token_counter: Union[str, Callable, "tiktoken.Encoding", "Tokenizer"]
                ) -> None:
        """Initialize the TokenProcessor."""
        # If the tokenizer_or_token_counter is a callable, then it's a token counter
        if callable(tokenizer_or_token_counter):
            self.token_counter = tokenizer_or_token_counter
            self._tokenizer_backend = "callable"
            self.tokenizer = None
        # If the tokenizer_or_token_counter is a string, then it's a tokenizer name
        elif isinstance(tokenizer_or_token_counter, str):
            self.tokenizer = self._load_tokenizer(tokenizer_or_token_counter)
            self._tokenizer_backend = self._get_tokenizer_backend()
            self.token_counter = self._get_tokenizer_counter()
        # If the tokenizer_or_token_counter is a tiktoken.Encoding or Tokenizer object, then it's a tokenizer
        else:
            self.tokenizer = tokenizer_or_token_counter
            self._tokenizer_backend = self._get_tokenizer_backend()
            self.token_counter = self._get_tokenizer_counter()
        
    def _get_tokenizer_backend(self) -> str:
        """Get the tokenizer backend."""
        # If the tokenizer is a tiktoken.Encoding object, then the backend is "tiktoken"
        if isinstance(self.tokenizer, tiktoken.Encoding):
            return "tiktoken"
        # If the tokenizer is a Tokenizer object, then the backend is "tokenizers"
        elif isinstance(self.tokenizer, Tokenizer):
            return "tokenizers"
        # If the tokenizer is a transformers.AutoTokenizer object, then the backend is "transformers"
        elif isinstance(self.tokenizer, AutoTokenizer):
            return "transformers"
        else:
            # Raise a Chonkie like vibey error
            raise ValueError("OOOOOOPS! We don't support this tokenizer backend yet! 🦛😅")
    
    def _load_tokenizer(self) -> "tiktoken.Encoding":
        """Load the tokenizer."""
        # If it can be loaded with tiktoken, then do so
        if importlib.util.find_spec("tiktoken") is not None:
            try:    
                from tiktoken import get_encoding
                return get_encoding(self.tokenizer)
            except Exception:
                # If it can be loaded with autotiktokenizer, then do so
                if importlib.util.find_spec("autotiktokenizer") is not None:
                    from autotiktokenizer import AutoTikTokenizer
                    return AutoTikTokenizer.from_pretrained(self.tokenizer)
                else:
                    raise ValueError("No suitable tokenizer backend found")
        else:
            raise ValueError("TikToken library not found. Trying autotiktokenizer.")
                    
    def _get_tokenizer_counter(self) -> Callable[[str], int]:
        """Get the tokenizer counter."""
        pass

    def encode(self, text: str) -> List[int]:
        """Encode a text string into a list of tokens."""
        pass
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a list of text strings into a list of lists of tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of tokens into a text string."""
        pass
    
    def decode_batch(self, tokens: List[List[int]]) -> List[str]:
        """Decode a list of lists of tokens into a list of text strings."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        pass

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count the number of tokens in a list of text strings."""
        pass
    
    def __repr__(self) -> str:
        """Return the string representation of the TokenProcessor."""
        pass