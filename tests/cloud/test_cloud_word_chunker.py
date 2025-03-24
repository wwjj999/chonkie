"""Test for the Chonkie Cloud Word Chunker class."""

import os

import pytest

from chonkie.cloud import WordChunker


@pytest.mark.skipif(    
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_word_chunker_initialization() -> None:
    """Test that the word chunker can be initialized."""
    # Check if not passing the API key raises an error
    with pytest.raises(ValueError):
        WordChunker(api_key=None)

    # Check if the chunk_size < 0 raises an error
    with pytest.raises(ValueError):
        WordChunker(tokenizer_or_token_counter="gpt2", chunk_size=-1, chunk_overlap=0)

    # Check if the chunk_overlap < 0 raises an error
    with pytest.raises(ValueError):
        WordChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=-1)

    # Check if the tokenizer_or_token_counter is not a string raises an error
    with pytest.raises(ValueError):
        WordChunker(tokenizer_or_token_counter=1, chunk_size=512, chunk_overlap=0)

    # Finally, check if the attributes are set correctly
    chunker = WordChunker(tokenizer_or_token_counter="gpt2", chunk_size=512, chunk_overlap=0)
    assert chunker.tokenizer_or_token_counter == "gpt2"
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 0

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_word_chunker_simple() -> None:
    """Test that the word chunker works."""
    word_chunker = WordChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        chunk_overlap=0,
    )
    result = word_chunker("Hello, world!")

    # Check the result
    assert isinstance(result, list) and isinstance(result[0], dict) and len(result) == 1
    assert result[0]["text"] == "Hello, world!"
    assert result[0]["token_count"] == 4
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 13


@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_word_chunker_multiple_sentences() -> None:
    """Test that the word chunker works with a complex text."""
    word_chunker = WordChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=5,
        chunk_overlap=0,
    )
    text = "This is one sentence. This is another sentence. This is a third sentence."
    result = word_chunker(text)

    # Check the result
    assert len(result) > 1
    assert isinstance(result, list)
    assert all(isinstance(item, dict) for item in result)
    assert all(isinstance(item["text"], str) for item in result)
    assert all(isinstance(item["token_count"], int) for item in result)
    assert all(isinstance(item["start_index"], int) for item in result)
    assert all(isinstance(item["end_index"], int) for item in result)

@pytest.mark.skipif(
    "CHONKIE_API_KEY" not in os.environ,
    reason="CHONKIE_API_KEY is not set",
)
def test_cloud_word_chunker_batch() -> None:
    """Test that the word chunker works with a batch of texts."""
    word_chunker = WordChunker(
        tokenizer_or_token_counter="gpt2",
        chunk_size=512,
        chunk_overlap=0,
    )
    texts = ["Hello, world!", "This is another sentence.", "This is a third sentence."]
    result = word_chunker(texts)

    # Check the result
    assert len(result) == len(texts)
    assert isinstance(result, list) and isinstance(result[0], list) and len(result[0]) == 1
    assert all(isinstance(item, dict) for item in result[0])
    assert all(isinstance(item["text"], str) for item in result[0])
    assert all(isinstance(item["token_count"], int) for item in result[0])
    assert all(isinstance(item["start_index"], int) for item in result[0])
    assert all(isinstance(item["end_index"], int) for item in result[0])