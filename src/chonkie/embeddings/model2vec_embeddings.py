from typing import List, TYPE_CHECKING

from numpy import ndarray

# for type checking
if TYPE_CHECKING:
    import numpy as np

from chonkie.embeddings import BaseEmbeddings


class Model2VecEmbeddings(BaseEmbeddings):

    def __init__(self, model_name: str = "minishlab/potion-base-8M"):
        """
        Initialize the PotionEmbeddings class.

        Args:
            model_name (str): Name of the pre-trained model to use for StaticEmbedding.
        """
        super().__init__()
        if not self.is_available():
            raise ImportError(
                "Required packages 'sentence_transformers' or 'StaticEmbedding' are not available. "
                "Please install them with 'pip install sentence-transformers'."
            )
        static_embedding = StaticEmbedding.from_model2vec(model_name)
        self.model = SentenceTransformer(modules=[static_embedding])

    def embed(self, text: str) -> "np.ndarray":
        """Embed a single text string into a vector representation."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Embed a list of text strings into vector representations."""
        return self.model.encode(texts, convert_to_numpy=True)

    def similarity(self, u: ndarray, v: ndarray) -> float:
        """Compute cosine similarity of two embeddings."""
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def count_tokens(self, text: str) -> int:
        return super().count_tokens(text)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        return super().count_tokens_batch(texts)

    @classmethod
    def is_available(cls) -> bool:
        """Check if the SentenceTransformer and StaticEmbedding dependencies are available."""
        try:
            global SentenceTransformer, StaticEmbedding
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import StaticEmbedding

            return True
        except ImportError:
            return False
