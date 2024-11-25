import pytest
import numpy as np
from chonkie import Model2VecEmbeddings
from model2vec import StaticModel


@pytest.fixture
def embedding_model():
    return Model2VecEmbeddings("minishlab/potion-base-8M")


@pytest.fixture
def sample_text():
    return "This is a sample text for testing."


@pytest.fixture
def sample_texts():
    return [
        "This is the first sample text.",
        "Here is another example sentence.",
        "Testing embeddings with multiple sentences.",
    ]


def test_initialization_with_model_name():
    embeddings = Model2VecEmbeddings("minishlab/potion-base-8M")
    assert (
        embeddings.model_name_or_path == "minishlab/potion-base-8M"
    )  # for now its None, see comments in model_2_vec.py
    assert embeddings.model is not None


def test_initialization_with_model_instance():
    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    embeddings = Model2VecEmbeddings(model)
    assert embeddings.model_name_or_path == model.base_model_name
    assert embeddings.model is model


def test_embed_single_text(embedding_model, sample_text):
    embedding = embedding_model.embed(sample_text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.dimension,)


def test_count_tokens_batch_texts(embedding_model, sample_texts):
    token_counts = embedding_model.count_tokens_batch(sample_texts)
    assert isinstance(token_counts, list)
    assert len(token_counts) == len(sample_texts)
    assert all(isinstance(count, int) for count in token_counts)
    assert all(count > 0 for count in token_counts)


def test_similarity(embedding_model, sample_texts):
    embeddings = embedding_model.embed_batch(sample_texts)
    similarity_score = embedding_model.similarity(embeddings[0], embeddings[1])
    assert isinstance(similarity_score, float)
    assert 0.0 <= similarity_score <= 1.0


def test_dimension_property(embedding_model):
    assert isinstance(embedding_model.dimension, int)
    assert embedding_model.dimension > 0


def test_is_available():
    assert Model2VecEmbeddings.is_available() is True


def test_repr(embedding_model):
    repr_str = repr(embedding_model)
    assert isinstance(repr_str, str)
    assert repr_str.startswith("Model2VecEmbeddings")


if __name__ == "__main__":
    pytest.main()
