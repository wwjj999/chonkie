from typing import List

import pytest
from transformers import AutoTokenizer

from chonkie.chunker import Chunk, Sentence, SentenceChunk
from chonkie.context import Context
from chonkie.refinery import OverlapRefinery


@pytest.fixture
def tokenizer():
    """Fixture providing a GPT-2 tokenizer for testing."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def basic_chunks() -> List[Chunk]:
    """Fixture providing a list of basic Chunks for testing."""
    return [
        Chunk(
            text="This is the first chunk of text.",
            start_index=0,
            end_index=30,
            token_count=8,
        ),
        Chunk(
            text="This is the second chunk of text.",
            start_index=31,
            end_index=62,
            token_count=8,
        ),
        Chunk(
            text="This is the third chunk of text.",
            start_index=63,
            end_index=93,
            token_count=8,
        ),
    ]


@pytest.fixture
def sentence_chunks() -> List[SentenceChunk]:
    """Fixture providing a list of SentenceChunks for testing."""
    sentences1 = [
        Sentence(text="First sentence.", start_index=0, end_index=14, token_count=3),
        Sentence(text="Second sentence.", start_index=15, end_index=30, token_count=3),
    ]
    sentences2 = [
        Sentence(text="Third sentence.", start_index=31, end_index=45, token_count=3),
        Sentence(text="Fourth sentence.", start_index=46, end_index=62, token_count=3),
    ]
    return [
        SentenceChunk(
            text="First sentence. Second sentence.",
            start_index=0,
            end_index=30,
            token_count=6,
            sentences=sentences1,
        ),
        SentenceChunk(
            text="Third sentence. Fourth sentence.",
            start_index=31,
            end_index=62,
            token_count=6,
            sentences=sentences2,
        ),
    ]


def test_overlap_refinery_initialization():
    """Test that OverlapRefinery initializes correctly with different parameters."""
    # Test default initialization
    refinery = OverlapRefinery()
    assert refinery.context_size == 128
    assert refinery.merge_context is True
    assert refinery.approximate is True
    assert not hasattr(refinery, "tokenizer")

    # Test initialization with tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    refinery = OverlapRefinery(
        context_size=64, tokenizer=tokenizer, merge_context=False, approximate=False
    )
    assert refinery.context_size == 64
    assert refinery.merge_context is False
    assert refinery.approximate is False
    assert hasattr(refinery, "tokenizer")
    assert refinery.tokenizer == tokenizer


def test_overlap_refinery_empty_input():
    """Test that OverlapRefinery handles empty input correctly."""
    refinery = OverlapRefinery()
    assert refinery.refine([]) == []


def test_overlap_refinery_single_chunk():
    """Test that OverlapRefinery handles single chunk input correctly."""
    refinery = OverlapRefinery()
    chunk = Chunk(text="Single chunk.", start_index=0, end_index=12, token_count=3)
    refined = refinery.refine([chunk])
    assert len(refined) == 1
    assert refined[0].context is None


def test_overlap_refinery_basic_chunks_approximate(basic_chunks):
    """Test approximate overlap calculation with basic Chunks."""
    refinery = OverlapRefinery(context_size=4)  # Small context for testing
    refined = refinery.refine(basic_chunks)

    # First chunk should have no context
    assert refined[0].context is None

    # Subsequent chunks should have context from previous chunks
    for i in range(1, len(refined)):
        assert refined[i].context is not None
        assert isinstance(refined[i].context, Context)
        assert refined[i].context.token_count <= 4


def test_overlap_refinery_basic_chunks_exact(basic_chunks, tokenizer):
    """Test exact overlap calculation with basic Chunks using tokenizer."""
    refinery = OverlapRefinery(context_size=4, tokenizer=tokenizer, approximate=False)
    refined = refinery.refine(basic_chunks)

    # Check context for subsequent chunks
    for i in range(1, len(refined)):
        assert refined[i].context is not None
        assert isinstance(refined[i].context, Context)
        # Verify exact token count using tokenizer
        actual_tokens = len(tokenizer.encode(refined[i].context.text))
        assert actual_tokens <= 4


def test_overlap_refinery_sentence_chunks(sentence_chunks):
    """Test overlap calculation with SentenceChunks."""
    refinery = OverlapRefinery(context_size=4)
    refined = refinery.refine(sentence_chunks)

    # Check context for second chunk
    assert refined[1].context is not None
    assert isinstance(refined[1].context, Context)
    assert refined[1].context.token_count <= 4


def test_overlap_refinery_no_merge_context(basic_chunks):
    """Test behavior when merge_context is False."""
    refinery = OverlapRefinery(context_size=4, merge_context=False)
    refined = refinery.refine(basic_chunks)

    # Chunks should maintain original text
    for i in range(len(refined)):
        assert refined[i].text == basic_chunks[i].text
        assert refined[i].token_count == basic_chunks[i].token_count


def test_overlap_refinery_context_size_limits(basic_chunks):
    """Test that context size limits are respected."""
    refinery = OverlapRefinery(context_size=2)  # Very small context
    refined = refinery.refine(basic_chunks)

    # Check that no context exceeds size limit
    for chunk in refined[1:]:  # Skip first chunk
        assert chunk.context.token_count <= 2


def test_overlap_refinery_merge_context(basic_chunks, tokenizer):
    """Test merging context into chunk text."""
    refinery = OverlapRefinery(
        context_size=4, tokenizer=tokenizer, merge_context=True, approximate=False
    )

    # Create a deep copy to preserve originals
    chunks_copy = [
        Chunk(
            text=chunk.text,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            token_count=chunk.token_count,
        )
        for chunk in basic_chunks
    ]

    refined = refinery.refine(chunks_copy)

    # First chunk should be unchanged
    assert refined[0].text == basic_chunks[0].text
    assert refined[0].token_count == basic_chunks[0].token_count

    # Subsequent chunks should have context prepended
    for i in range(1, len(refined)):
        assert refined[i].context is not None
        assert refined[i].text.startswith(refined[i].context.text)
        # Verify token count increase
        original_tokens = len(tokenizer.encode(basic_chunks[i].text))
        new_tokens = len(tokenizer.encode(refined[i].text))
        assert new_tokens > original_tokens


def test_overlap_refinery_mixed_chunk_types():
    """Test that refinery raises error for mixed chunk types."""
    # Create chunks of different types
    chunks = [
        Chunk(text="Basic chunk.", start_index=0, end_index=12, token_count=3),
        SentenceChunk(
            text="Sentence chunk.",
            start_index=13,
            end_index=27,
            token_count=3,
            sentences=[],
        ),
    ]

    refinery = OverlapRefinery()
    with pytest.raises(ValueError, match="All chunks must be of the same type"):
        refinery.refine(chunks)


if __name__ == "__main__":
    pytest.main()
