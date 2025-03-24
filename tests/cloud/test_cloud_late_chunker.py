"""Test the Chonkie Cloud Late Chunker."""

import os

import pytest

from chonkie.cloud.chunker import LateChunker


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_initialization() -> None:
    """Test that the late chunker can be initialized."""
    # Check if not passing the API key raises an error
    with pytest.raises(ValueError):
        LateChunker(api_key=None)

    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        LateChunker(embedding_model="all-minilm-l6-v2", chunk_size=-1)
    
    # Check if the mode is not "token" or "sentence"
    with pytest.raises(ValueError):
        LateChunker(embedding_model="all-minilm-l6-v2", chunk_size=512, mode="not_a_string")
    
    # Check if the min_sentences_per_chunk is not a positive integer
    with pytest.raises(ValueError):
        LateChunker(embedding_model="all-minilm-l6-v2", chunk_size=512, min_sentences_per_chunk=-1)
    
    # Check if the min_characters_per_sentence is not a positive integer
    with pytest.raises(ValueError):
        LateChunker(embedding_model="all-minilm-l6-v2", chunk_size=512, min_characters_per_sentence=-1)
    
    # Check if the approximate is not a boolean
    with pytest.raises(ValueError):
        LateChunker(embedding_model="all-minilm-l6-v2", chunk_size=512, approximate="not_a_boolean")
    
    # Check if the delim is not a list of strings or a string
    with pytest.raises(ValueError):
        LateChunker(embedding_model="all-minilm-l6-v2", chunk_size=512, delim=1)
        
    # Check if the include_delim is not "prev" or "next"
    with pytest.raises(ValueError):
        LateChunker(embedding_model="all-minilm-l6-v2", chunk_size=512, include_delim="not_a_string")
    
    # Finally, check if the attributes are set correctly
    chunker = LateChunker(embedding_model="all-minilm-l6-v2", chunk_size=512)
    assert chunker.embedding_model == "all-minilm-l6-v2"
    assert chunker.chunk_size == 512
    assert chunker.mode == "sentence"
    assert chunker.min_sentences_per_chunk == 1
    assert chunker.min_characters_per_sentence == 12
    assert chunker.approximate == True
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.include_delim == "prev"
    assert chunker.return_type == "chunks"

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_single_sentence() -> None:
    """Test that the late chunker works with a single sentence."""
    late_chunker = LateChunker(
        embedding_model="all-minilm-l6-v2",
        chunk_size=512,
     )

    result = late_chunker("Hello, world!")

    assert len(result) == 1
    assert result[0]["text"] == "Hello, world!"
    assert result[0]["token_count"] == 4
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 13

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_batch() -> None:
    """Test that the late chunker works with a batch of texts."""
    late_chunker = LateChunker(
        embedding_model="all-minilm-l6-v2",
        chunk_size=512,
    )

    result = late_chunker(["Hello, world!", "This is another sentence.", "This is a third sentence."])

    assert len(result) == 3
    assert result[0][0]["text"] == "Hello, world!"
    assert result[0][0]["token_count"] == 4
    assert result[0][0]["start_index"] == 0
    assert result[0][0]["end_index"] == 13

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_late_chunker_empty_text() -> None:
    """Test that the late chunker works with an empty text."""
    late_chunker = LateChunker(
        embedding_model="all-minilm-l6-v2",
        chunk_size=512,
    )

    result = late_chunker("")

    assert len(result) == 0