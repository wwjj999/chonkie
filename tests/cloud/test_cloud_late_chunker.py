"""Test the Chonkie Cloud Late Chunker."""

import os

import pytest

from chonkie.cloud.chunker import LateChunker

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