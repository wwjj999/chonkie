from typing import Dict, List, Optional, Type, Union, Pattern
from dataclasses import dataclass
import re

from .base import BaseEmbeddings
from .sentence_transformer import SentenceTransformerEmbeddings
from .openai import OpenAIEmbeddings
from .model2vec import Model2VecEmbeddings

@dataclass
class RegistryEntry:
    """Registry entry containing the embeddings class and optional pattern."""
    embeddings_cls: Type[BaseEmbeddings]
    pattern: Optional[Pattern] = None

class EmbeddingsRegistry:
    """Registry for embedding implementations with pattern matching support."""
    
    _registry: Dict[str, RegistryEntry] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        embedding_cls: Type[BaseEmbeddings],
        pattern: Optional[Union[str, Pattern]] = None,
    ):
        """Register a new embeddings implementation.
        
        Args:
            name: Unique identifier for this implementation
            embedding_cls: The embeddings class to register
            pattern: Optional regex pattern string or compiled pattern
            
        Examples:
            # Register with exact name
            EmbeddingsRegistry.register("sentence-transformers", SentenceTransformerEmbeddings)
            
            # Register with pattern for OpenAI
            EmbeddingsRegistry.register(
                "openai", 
                OpenAIEmbeddings,
                pattern=r"^openai://.+|^text-embedding-ada-\d+$"
            )
        """
        if not issubclass(embedding_cls, BaseEmbeddings):
            raise ValueError(f"{embedding_cls} must be a subclass of BaseEmbeddings")
            
        # Compile pattern if provided as string
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
            
        cls._registry[name] = RegistryEntry(
            embeddings_cls=embedding_cls,
            pattern=pattern
        )
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseEmbeddings]]:
        """Get embeddings class by exact name match."""
        entry = cls._registry.get(name)
        return entry.embeddings_cls if entry else None
    
    @classmethod
    def match(cls, identifier: str) -> Optional[Type[BaseEmbeddings]]:
        """Find matching embeddings class using both exact matches and patterns.
        
        Args:
            identifier: String to match against registry entries
            
        Returns:
            Matching embeddings class or None if no match found
            
        Examples:
            # Match exact name
            cls.match("sentence-transformers") -> SentenceTransformerEmbeddings
            
            # Match OpenAI pattern
            cls.match("openai://my-embedding") -> OpenAIEmbeddings
            
            # Match model name pattern
            cls.match("text-embedding-ada-002") -> OpenAIEmbeddings
        """
        # First try exact match
        if identifier in cls._registry:
            return cls._registry[identifier].embeddings_cls
        
        # Then try patterns
        for entry in cls._registry.values():
            if entry.pattern and entry.pattern.match(identifier):
                return entry.embeddings_cls
        
        return None
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List names of available embeddings implementations."""
        return [
            name for name, entry in cls._registry.items() 
            if entry.embeddings_cls.is_available()
        ]

# Register all the available embeddings in the EmbeddingsRegistry! 
# This is essential for the `AutoEmbeddings` to work properly.

# Register SentenceTransformer embeddings with pattern
EmbeddingsRegistry.register(
    "sentence-transformer",
    SentenceTransformerEmbeddings, 
    pattern=r"^sentence-transformers/|^all-MiniLM-|^paraphrase-|^multi-qa-|^msmarco-"
)

# Register OpenAI embeddings with pattern
EmbeddingsRegistry.register(
    "openai",
    OpenAIEmbeddings,
    pattern=r"^openai|^text-embedding-"
)
EmbeddingsRegistry.register(
    "text-embedding-ada-002",
    OpenAIEmbeddings
)
EmbeddingsRegistry.register(
    "text-embedding-3-small",
    OpenAIEmbeddings
)
EmbeddingsRegistry.register(
    "text-embedding-3-large",
    OpenAIEmbeddings
)

# Register model2vec embeddings
EmbeddingsRegistry.register(
    "model2vec",
    Model2VecEmbeddings, 
    pattern=r"^minishlab/|^minishlab/potion-base-|^minishlab/potion-|^potion-|"
)
