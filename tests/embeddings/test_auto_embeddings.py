import pytest
from chonkie.embeddings.auto import AutoEmbeddings
from chonkie.embeddings.model2vec import Model2VecEmbeddings
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chonkie.embeddings.openai import OpenAIEmbeddings

@pytest.fixture
def model2vec_identifier():
    return "minishlab/potion-base-8M"

@pytest.fixture
def sentence_transformer_identifier():
    return "all-MiniLM-L6-v2"

@pytest.fixture
def sentence_transformer_identifier_small():
    return "all-minilm-l6-v2"

@pytest.fixture
def openai_identifier():
    return "text-embedding-3-small"

@pytest.fixture
def invalid_identifier():
    return "invalid-identifier-123"

def test_auto_embeddings_model2vec(model2vec_identifier):
    embeddings = AutoEmbeddings.get_embeddings(model2vec_identifier)
    assert isinstance(embeddings, Model2VecEmbeddings)
    assert embeddings.model_name_or_path == model2vec_identifier

def test_auto_embeddings_sentence_transformer(sentence_transformer_identifier):
    embeddings = AutoEmbeddings.get_embeddings(sentence_transformer_identifier)
    assert isinstance(embeddings, SentenceTransformerEmbeddings)
    assert embeddings.model_name_or_path == sentence_transformer_identifier

def test_auto_embeddings_sentence_transformer_alt(sentence_transformer_identifier_small):
    embeddings = AutoEmbeddings.get_embeddings(sentence_transformer_identifier_small)
    assert isinstance(embeddings, SentenceTransformerEmbeddings)
    assert embeddings.model_name_or_path == sentence_transformer_identifier_small

def test_auto_embeddings_openai(openai_identifier):
    embeddings = AutoEmbeddings.get_embeddings(openai_identifier, api_key="your_openai_api_key")
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == openai_identifier

def test_auto_embeddings_invalid_identifier(invalid_identifier):
    with pytest.raises(ValueError):
        AutoEmbeddings.get_embeddings(invalid_identifier)

if __name__ == "__main__":
    pytest.main()