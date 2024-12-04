"""Factory class for creating and managing tokenizers.

This factory class is used to create and manage tokenizers for the Chonkie
package. It provides a simple interface for initializing, encoding, decoding,
and counting tokens using different tokenizer backends.

This is used in the Chunker and Refinery classes to ensure consistent tokenization
across different parts of the pipeline.
"""

from typing import Callable, List, TYPE_CHECKING


if TYPE_CHECKING:
    import tiktoken
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer

class TokenFactory:
    """Factory class for creating and managing tokenizers."""
    pass