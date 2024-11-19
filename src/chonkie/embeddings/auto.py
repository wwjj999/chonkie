from typing import Any, Union
from .base import BaseEmbeddings
from .registry import EmbeddingsRegistry
import warnings

class AutoEmbeddings:
    """Factory class for automatically loading embeddings.
    
    This class provides a factory interface for loading embeddings based on an
    identifier string. It will try to find a matching embeddings implementation
    based on the identifier and load it with the provided arguments.


    Examples:
        # Get sentence transformers embeddings
        embeddings = AutoEmbeddings.get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
        
        # Get OpenAI embeddings
        embeddings = AutoEmbeddings.get_embeddings("openai://text-embedding-ada-002", api_key="...")
        
        # Get Anthropic embeddings
        embeddings = AutoEmbeddings.get_embeddings("anthropic://claude-v1", api_key="...")
    """
    
    @classmethod
    def get_embeddings(cls, model: Union[str, BaseEmbeddings, Any], **kwargs) -> BaseEmbeddings:
        """Get embeddings instance based on identifier.
        
        Args:
            identifier: Identifier for the embeddings (name, path, URL, etc.)
            **kwargs: Additional arguments passed to the embeddings constructor
            
        Returns:
            Initialized embeddings instance
            
        Raises:
            ValueError: If no suitable embeddings implementation is found
            
        Examples:
            # Get sentence transformers embeddings
            embeddings = AutoEmbeddings.get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
            
            # Get OpenAI embeddings
            embeddings = AutoEmbeddings.get_embeddings("openai://text-embedding-ada-002", api_key="...")
            
            # Get Anthropic embeddings
            embeddings = AutoEmbeddings.get_embeddings("anthropic://claude-v1", api_key="...")
        """
        # Try to find matching implementation
        try:
            embeddings_cls = EmbeddingsRegistry.match(model)
            if embeddings_cls and embeddings_cls.is_available():
                try:
                    return embeddings_cls(model, **kwargs)
                except Exception as e:
                    warnings.warn(f"Failed to load {embeddings_cls.__name__}: {e}")
        except Exception:
            # Fallback to sentence-transformers if no match found
            from .sentence_transformer import SentenceTransformerEmbeddings
            try:
                return SentenceTransformerEmbeddings(model, **kwargs)
            except Exception as e:
                raise ValueError(f"Failed to load embeddings: {e}")
