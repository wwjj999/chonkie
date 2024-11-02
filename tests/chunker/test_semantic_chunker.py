import pytest
from tokenizers import Tokenizer
from chonkie.chunker.semantic import SemanticChunker, SemanticChunk

@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("gpt2")

@pytest.fixture
def sample_text():
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text

def test_semantic_chunker_initialization(tokenizer):
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_threshold=0.5
    )
    
    assert chunker is not None
    assert chunker.tokenizer == tokenizer
    assert chunker.max_chunk_size == 512
    assert chunker.similarity_threshold == 0.5
    assert chunker.sentence_mode == "heuristic"
    assert chunker.initial_sentences == 1

def test_semantic_chunker_chunking(tokenizer, sample_text):
    """Test that the SemanticChunker can chunk a sample text."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    
    assert len(chunks) > 0
    assert isinstance(chunks[0], SemanticChunk)
    assert all([chunk.token_count <= 512 for chunk in chunks])
    assert all([chunk.token_count > 0 for chunk in chunks])
    assert all([chunk.text is not None for chunk in chunks])
    assert all([chunk.start_index is not None for chunk in chunks])
    assert all([chunk.end_index is not None for chunk in chunks])
    assert all([chunk.sentences is not None for chunk in chunks])

def test_semantic_chunker_empty_text(tokenizer):
    """Test that the SemanticChunker can handle empty text input."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_threshold=0.5
    )
    chunks = chunker.chunk("")
    
    assert len(chunks) == 0

def test_semantic_chunker_single_sentence(tokenizer):
    """Test that the SemanticChunker can handle text with a single sentence."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_threshold=0.5
    )
    chunks = chunker.chunk("This is a single sentence.")
    
    assert len(chunks) == 1
    assert chunks[0].text == "This is a single sentence."
    assert len(chunks[0].sentences) == 1

def test_semantic_chunker_single_chunk_text(tokenizer):
    """Test that the SemanticChunker can handle text that fits in a single chunk."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_threshold=0.5
    )
    text = "Hello, how are you? I am doing well."
    chunks = chunker.chunk(text)
    
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert len(chunks[0].sentences) == 2

def test_semantic_chunker_repr(tokenizer):
    """Test that the SemanticChunker has a string representation."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_threshold=0.5
    )
    
    expected = ("SemanticChunker(max_chunk_size=512, similarity_threshold=0.5, "
                "initial_sentences=1, sentence_mode='heuristic')")
    assert repr(chunker) == expected

def test_semantic_chunker_similarity_threshold(tokenizer):
    """Test that the SemanticChunker respects similarity threshold."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_threshold=0.9  # High threshold should create more chunks
    )
    text = ("This is about cars. This is about planes. "
            "This is about trains. This is about boats.")
    chunks = chunker.chunk(text)
    
    # With high similarity threshold, we expect more chunks due to low similarity
    assert len(chunks) > 1

def test_semantic_chunker_percentile_mode(tokenizer, sample_text):
    """Test that the SemanticChunker works with percentile-based similarity."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_percentile=50  # Use median similarity as threshold
    )
    chunks = chunker.chunk(sample_text)
    
    assert len(chunks) > 0
    assert all([isinstance(chunk, SemanticChunk) for chunk in chunks])

def test_semantic_chunker_spacy_mode(tokenizer, sample_text):
    """Test that the SemanticChunker works with spaCy mode."""
    chunker = SemanticChunker(
        tokenizer=tokenizer,
        sentence_transformer_model="all-MiniLM-L6-v2",
        max_chunk_size=512,
        similarity_threshold=0.5,
        sentence_mode="spacy"
    )
    chunks = chunker.chunk(sample_text)
    
    assert len(chunks) > 0
    assert chunker.sentence_mode == "spacy"

if __name__ == "__main__":
    pytest.main()