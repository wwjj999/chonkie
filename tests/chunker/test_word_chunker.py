import pytest
from tokenizers import Tokenizer

from chonkie.chunker.base import Chunk
from chonkie.chunker.word import WordChunker


@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")

@pytest.fixture
def sample_text():
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text

def test_word_chunker_initialization(tokenizer):
    """
    Test that the WordChunker can be initialized with a tokenizer. 
    """
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128, mode="simple")
    
    assert chunker is not None
    assert chunker.tokenizer == tokenizer
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128
    assert chunker.mode == "simple"

def test_word_chunker_chunking(tokenizer, sample_text):
    """
    Test that the WordChunker can chunk a sample text into words.
    """
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128, mode="simple")
    chunks = chunker.chunk(sample_text)
    
    assert len(chunks) > 0, print(f"Chunks: {chunks}")
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])

def test_word_chunker_empty_text(tokenizer):
    """
    Test that the WordChunker can handle empty text input.
    """
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128, mode="simple")
    chunks = chunker.chunk("")
    
    assert len(chunks) == 0

def test_word_chunker_single_word_text(tokenizer):
    """
    Test that the WordChunker can handle text with a single word.
    """
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128, mode="simple")
    chunks = chunker.chunk("Hello")
    
    assert len(chunks) == 1
    assert chunks[0].token_count == 1
    assert chunks[0].text == "Hello"

def test_word_chunker_single_chunk_text(tokenizer):
    """
    Test that the WordChunker can handle text that fits within a single chunk.
    """
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128, mode="simple")
    chunks = chunker.chunk("Hello, how are you?")

    assert len(chunks) == 1, print(f"Chunks: {chunks}")
    assert chunks[0].token_count == 6
    assert chunks[0].text == "Hello, how are you?"

def test_word_chunker_repr(tokenizer):
    """
    Test that the WordChunker has a string representation.
    """
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128, mode="simple")
    
    assert repr(chunker) == "WordChunker(chunk_size=512, chunk_overlap=128, mode='simple')"

def test_word_chunker_call(tokenizer, sample_text):
    """
    Test that the WordChunker can be called directly.
    """
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128, mode="simple")
    chunks = chunker(sample_text)
    
    assert len(chunks) > 0
    assert type(chunks[0]) is Chunk
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])

def test_word_chunker_overlap(tokenizer, sample_text):
    """
    Test that the WordChunker creates overlapping chunks correctly.
    """
    chunker = WordChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128, mode="simple")
    chunks = chunker.chunk(sample_text)
    
    for i in range(1, len(chunks)):
        assert chunks[i].start_index < chunks[i-1].end_index

if __name__ == "__main__":
    pytest.main()