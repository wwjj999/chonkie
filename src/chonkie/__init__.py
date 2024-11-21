from .chunker import (BaseChunker, Chunk, SDPMChunker, SemanticChunk,
                      SemanticChunker, SemanticSentence, Sentence,
                      SentenceChunk, SentenceChunker, TokenChunker,
                      WordChunker)
from .embeddings import (BaseEmbeddings, SentenceTransformerEmbeddings, 
                         Model2VecEmbeddings, OpenAIEmbeddings, AutoEmbeddings)

__version__ = "0.2.1"
__name__ = "chonkie"
__author__ = "Bhavnick Minhas"

# Add basic package metadata to __all__
__all__ = [
    "__name__",
    "__version__",
    "__author__",
]

# Add all data classes to __all__
__all__ += [
    "Chunk",
    "SentenceChunk",
    "SemanticChunk",
    "Sentence",
    "SemanticSentence",
]

# Add all chunker classes to __all__
__all__ += [    
    "BaseChunker",
    "TokenChunker",
    "WordChunker",
    "SentenceChunker",
    "SemanticChunker",
    "SDPMChunker",
]

# Add all embeddings classes to __all__
__all__ += [
    "BaseEmbeddings",
    "Model2VecEmbeddings",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    "AutoEmbeddings",
]
