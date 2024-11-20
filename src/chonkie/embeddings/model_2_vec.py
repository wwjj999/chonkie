from typing import List, Union
import importlib.util

from chonkie.embeddings.base import BaseEmbeddings

import numpy as np


class Model2VecEmbeddings(BaseEmbeddings):

    def __init__(self, model: Union[str, "StaticModel"]) -> None:

        if not self.is_available():
            raise ImportError("model2vec is not available. Please install it via pip.")
        else:
            global StaticModel
            from model2vec import StaticModel

        if isinstance(model, str):
            self.model_name_or_path = model
            self.model = StaticModel.from_pretrained(self.model_name_or_path)
        elif isinstance(model, StaticModel):
            self.model = model
            # TODO: `base_model_name` is mentioned in here -
            # https://github.com/MinishLab/model2vec/blob/b1358a9c2e777800e8f89c7a5f830fa2176c15b5/model2vec/model.py#L165`
            # but its `None` for potion models
            self.model_name_or_path = self.model.base_model_name
        else:
            raise ValueError("model must be a string or model2vec.StaticModel instance")
        self._dimension = self.model.dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> "np.ndarray":
        """Embed a single text using the model2vec model."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Embed multiple texts using the model2vec model."""
        return self.model.encode(texts, convert_to_numpy=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.model.tokenizer.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts using the model's tokenizer."""
        encodings = self.model.tokenizer.encode_batch(texts)
        return [len(enc) for enc in encodings]

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> float:
        """Compute cosine similarity of two embeddings."""
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("model2vec") is not None
