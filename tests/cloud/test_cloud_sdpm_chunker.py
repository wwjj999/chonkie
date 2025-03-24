"""Test the Chonkie Cloud SDPM Chunker."""

import os

import pytest

from chonkie.cloud import SDPMChunker


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_initialization() -> None:
    """Test that the SDPM Chunker can be initialized."""
    # Check if not passing the API key raises an error
    with pytest.raises(ValueError):
        SDPMChunker(api_key=None)

    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        SDPMChunker(chunk_size=-1)

    # Check if similarity_window is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(similarity_window=-1)

    # Check if min_sentences is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_sentences=-1)
    
    # Check if min_chunk_size is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_chunk_size=-1)

    # Check if min_characters_per_sentence is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(min_characters_per_sentence=-1)
    
    # Check if threshold is a str but not "auto"
    with pytest.raises(ValueError):
        SDPMChunker(threshold="not_auto")
    
    # Check if threshold is a float but not between 0 and 1
    with pytest.raises(ValueError):
        SDPMChunker(threshold=1.1)
    
    # Check if threshold is an int but not between 1 and 100
    with pytest.raises(ValueError):
        SDPMChunker(threshold=101)
    
    # Check if threshold_step is not a number between 0 and 1
    with pytest.raises(ValueError):
        SDPMChunker(threshold_step=-0.1)
    
    # Check if delim is not a list of strings or a string
    with pytest.raises(ValueError):
        SDPMChunker(delim=1)
    
    # Check if skip_window is not a positive integer
    with pytest.raises(ValueError):
        SDPMChunker(skip_window=-1)
    
    # Check if return_type is not "chunks" or "texts"
    with pytest.raises(ValueError):
        SDPMChunker(return_type="not_a_string")

    # Finally, check if the attributes are set correctly
    chunker = SDPMChunker(embedding_model="minishlab/potion-base-32M",
                         chunk_size=512,
                         mode="window",
                         threshold="auto",
                         similarity_window=1,
                         min_sentences=1,
                         min_chunk_size=2,
                         min_characters_per_sentence=12,
                         threshold_step=0.01,
                         delim=[".", "!", "?", "\n"],
                         skip_window=1,
                         return_type="chunks")
    
    assert chunker.embedding_model == "minishlab/potion-base-32M"
    assert chunker.chunk_size == 512
    assert chunker.mode == "window"
    assert chunker.threshold == "auto"
    assert chunker.similarity_window == 1
    assert chunker.min_sentences == 1
    assert chunker.min_chunk_size == 2
    assert chunker.min_characters_per_sentence == 12
    assert chunker.threshold_step == 0.01
    assert chunker.delim == [".", "!", "?", "\n"]
    assert chunker.skip_window == 1
    assert chunker.return_type == "chunks"

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_single_sentence() -> None:
    """Test that the SDPM Chunker works with a single sentence."""
    sdpm_chunker = SDPMChunker(
        embedding_model="all-minilm-l6-v2",
        chunk_size=512,
    )

    result = sdpm_chunker("Hello, world!")
    assert len(result) == 1
    assert result[0]["text"] == "Hello, world!"
    assert result[0]["token_count"] == 4
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 13


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_batch() -> None:
    """Test that the SDPM Chunker works with a batch of texts."""
    sdpm_chunker = SDPMChunker(
        embedding_model="all-minilm-l6-v2",
        chunk_size=512,
    )

    result = sdpm_chunker(["Hello, world!", "This is another sentence.", "This is a third sentence."])
    
    assert len(result) == 3
    assert result[0][0]["text"] == "Hello, world!"
    assert result[0][0]["token_count"] == 4
    assert result[0][0]["start_index"] == 0
    assert result[0][0]["end_index"] == 13

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_sdpm_chunker_empty_text() -> None:
    """Test that the SDPM Chunker works with an empty text."""
    sdpm_chunker = SDPMChunker(
        embedding_model="all-minilm-l6-v2",
        chunk_size=512,
    )

    result = sdpm_chunker("")

    assert len(result) == 0
