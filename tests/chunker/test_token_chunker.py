import pytest

import tiktoken
from transformers import AutoTokenizer
from tokenizers import Tokenizer

from chonkie import Chunk
from chonkie import TokenChunker

@pytest.fixture
def transformers_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture
def tiktokenizer():
    return tiktoken.get_encoding("gpt2")

@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")

@pytest.fixture
def sample_text():
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text

def test_token_chunker_initialization_tok(tokenizer):
    """
    Test that the TokenChunker can be initialized with a tokenizer. 
    """
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    
    assert chunker is not None
    assert chunker.tokenizer == tokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128

def test_token_chunker_initialization_hftok(transformers_tokenizer):
    """
    Test that the TokenChunker can be initialized with a tokenizer. 
    """
    chunker = TokenChunker(tokenizer=transformers_tokenizer, chunk_size=512, chunk_overlap=128)
    
    assert chunker is not None
    assert chunker.tokenizer == transformers_tokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128


def test_token_chunker_initialization_tik(tiktokenizer):
    """
    Test that the TokenChunker can be initialized with a tokenizer. 
    """
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    
    assert chunker is not None
    assert chunker.tokenizer == tiktokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128

def test_token_chunker_chunking(tokenizer, sample_text):
    """
    Test that the TokenChunker can chunk a sample text into tokens.
    """
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)
    
    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])

def test_token_chunker_chunking(transformers_tokenizer, sample_text):
    """
    Test that the TokenChunker can chunk a sample text into tokens.
    """
    chunker = TokenChunker(tokenizer=transformers_tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)
    
    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])

def test_token_chunker_chunking(tiktokenizer, sample_text):
    """
    Test that the TokenChunker can chunk a sample text into tokens.
    """
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk(sample_text)
    
    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])

def test_token_chunker_empty_text(tokenizer):
    """
    Test that the TokenChunker can handle empty text input.
    """
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("")
    
    assert len(chunks) == 0

def test_token_chunker_single_token_text(tokenizer):
    """
    Test that the TokenChunker can handle text with a single token.
    """
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello")
    
    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].text == "Hello"

def test_token_chunker_single_chunk_text(tokenizer):
    """
    Test that the TokenChunker can handle text that fits within a single chunk.
    """
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk("Hello, how are you?")
    
    assert len(chunks) == 1
    assert chunks[0].token_count == 6
    assert chunks[0].text == "Hello, how are you?"

def test_token_chunker_repr(tokenizer):
    """
    Test that the TokenChunker has a string representation.
    """
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    
    assert repr(chunker) == "TokenChunker(chunk_size=512, chunk_overlap=128)"

def test_token_chunker_call(tokenizer, sample_text):
    """
    Test that the TokenChunker can be called directly.
    """
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    chunks = chunker(sample_text)
    
    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])


if __name__ == "__main__":
    pytest.main()