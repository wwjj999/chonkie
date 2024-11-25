import os
from typing import List

import pytest

from chonkie import Chunk, SemanticChunk, SemanticChunker
from chonkie.embeddings import Model2VecEmbeddings, OpenAIEmbeddings


@pytest.fixture
def sample_text():
    text = """The process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. In this post, we explore the challenges of text chunking in RAG applications and propose a novel approach that leverages recent advances in transformer-based language models to achieve a more effective balance between these competing requirements."""
    return text


@pytest.fixture
def embedding_model():
    return Model2VecEmbeddings("minishlab/potion-base-8M")


@pytest.fixture
def openai_embedding_model():
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@pytest.fixture
def sample_complex_markdown_text():
    text = """# Heading 1
    This is a paragraph with some **bold text** and _italic text_. 
    ## Heading 2
    - Bullet point 1
    - Bullet point 2 with `inline code`
    ```python
    # Code block
    def hello_world():
        print("Hello, world!")
    ```
    Another paragraph with [a link](https://example.com) and an image:
    ![Alt text](https://example.com/image.jpg)
    > A blockquote with multiple lines
    > that spans more than one line.
    Finally, a paragraph at the end.
    """
    return text


def test_semantic_chunker_initialization(embedding_model):
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        similarity_threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.similarity_threshold == 0.5
    assert chunker.initial_sentences == 1


def test_semantic_chunker_initialization_openai(openai_embedding_model):
    """Test that the SemanticChunker can be initialized with required parameters."""
    chunker = SemanticChunker(
        embedding_model=openai_embedding_model,
        chunk_size=512,
        similarity_threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.similarity_threshold == 0.5
    assert chunker.initial_sentences == 1


def test_semantic_chunker_initialization_sentence_transformer():
    """Test that the SemanticChunker can be initialized with SentenceTransformer model."""
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        similarity_threshold=0.5,
    )

    assert chunker is not None
    assert chunker.chunk_size == 512
    assert chunker.similarity_threshold == 0.5
    assert chunker.initial_sentences == 1


def test_semantic_chunker_chunking(embedding_model, sample_text):
    """Test that the SemanticChunker can chunk a sample text."""
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        similarity_threshold=0.5,
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


def test_semantic_chunker_empty_text(embedding_model):
    """Test that the SemanticChunker can handle empty text input."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        similarity_threshold=0.5,
    )
    chunks = chunker.chunk("")

    assert len(chunks) == 0


def test_semantic_chunker_single_sentence(embedding_model):
    """Test that the SemanticChunker can handle text with a single sentence."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        similarity_threshold=0.5,
    )
    chunks = chunker.chunk("This is a single sentence.")

    assert len(chunks) == 1
    assert chunks[0].text == "This is a single sentence."
    assert len(chunks[0].sentences) == 1


def test_semantic_chunker_repr(embedding_model):
    """Test that the SemanticChunker has a string representation."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        similarity_threshold=0.5,
    )

    expected = (
        "SemanticChunker(chunk_size=512, similarity_threshold=0.5, "
        "initial_sentences=1)"
    )
    assert repr(chunker) == expected


def test_semantic_chunker_similarity_threshold(embedding_model):
    """Test that the SemanticChunker respects similarity threshold."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        similarity_threshold=0.9,  # High threshold should create more chunks
    )
    text = (
        "This is about cars. This is about planes. "
        "This is about trains. This is about boats."
    )
    chunks = chunker.chunk(text)

    # With high similarity threshold, we expect more chunks due to low similarity
    assert len(chunks) > 1


def test_semantic_chunker_percentile_mode(embedding_model, sample_text):
    """Test that the SemanticChunker works with percentile-based similarity."""
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        chunk_size=512,
        similarity_percentile=50,  # Use median similarity as threshold
    )
    chunks = chunker.chunk(sample_text)

    assert len(chunks) > 0
    assert all([isinstance(chunk, SemanticChunk) for chunk in chunks])


def verify_chunk_indices(chunks: List[Chunk], original_text: str):
    """Verify that chunk indices correctly map to the original text."""
    for i, chunk in enumerate(chunks):
        # Extract text using the indices
        extracted_text = original_text[chunk.start_index : chunk.end_index]
        # Remove any leading/trailing whitespace from both texts for comparison
        chunk_text = chunk.text.strip()
        extracted_text = extracted_text.strip()

        assert chunk_text == extracted_text, (
            f"Chunk {i} text mismatch:\n"
            f"Chunk text: '{chunk_text}'\n"
            f"Extracted text: '{extracted_text}'\n"
            f"Indices: [{chunk.start_index}:{chunk.end_index}]"
        )


def test_sentence_chunker_indices(embedding_model, sample_text):
    """Test that the SentenceChunker correctly maps chunk indices to the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, similarity_threshold=0.5
    )
    chunks = chunker.chunk(sample_text)
    verify_chunk_indices(chunks, sample_text)


def test_sentence_chunker_indices_complex_md(
    embedding_model, sample_complex_markdown_text
):
    """Test that the SentenceChunker correctly maps chunk indices to the original text."""
    chunker = SemanticChunker(
        embedding_model=embedding_model, chunk_size=512, similarity_threshold=0.5
    )
    chunks = chunker.chunk(sample_complex_markdown_text)
    verify_chunk_indices(chunks, sample_complex_markdown_text)


if __name__ == "__main__":
    pytest.main()
