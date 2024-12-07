"""Class for Late Chunking."""

from typing import Any, Callable, List, Union
from .base import BaseChunker
from chonkie.types import Chunk
from chonkie.embeddings import BaseEmbeddings


class LateChunker(BaseChunker):
    """Class for Late Chunking.

    In late chunking, we first take the embeddings of the entire text,
    after which we split the text into chunks. When we split the text, 
    we can use the embeddings generated earlier, mean pool them and 
    use them as the sentence embeddings for the chunks. 

    This class particularly implements the Sentence-style LateChunking 
    approach by changing the way we prepare the sentences before the 
    chunking. 

    Args: 
        mode: Mode of the LateChunker to split the text in
    """  

    def __init__(self,
                 embedding_model: Union[str, BaseEmbeddings] = "all-minilm-l6-v2", 
                 mode: str = "sentence",
                 chunk_size: int = 512, 
                 min_sentences_per_chunk: int = 1, 
                 min_chunk_size: int = 2, 
                 approximate: bool = True, 
                 delim: Union[str, List[str]] = ['.', '!', '?', '\n']
                 ) -> None:
        """Initialise the LateChunker."""
        # Assign the values if they make sense
        if mode not in ['token', 'sentence']:
            raise ValueError(
                "Mode must be one of the following: ['token', 'sentence']"
            )
        if chunk_size <= 0: 
            raise ValueError("chunk_size must be a positive non-zero value!")
        if min_sentences_per_chunk < 0 and mode == 'sentence':
            raise ValueError(f"min_sentences_per_chunk was assigned {min_sentences_per_chunk}; but it must be non-negative value!")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be a positive non-zero value!")
        if type(delim) not in [str, list]:
            raise TypeError("delim must be of type str or list of str")
        
        self.mode = mode
        self.chunk_size = chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_chunk_size = min_chunk_size
        self.approximate = approximate
        self.delim = delim
        self.sep = '🦛'

        # Initialise the embeddings via AutoEmbeddings
        if isinstance(embedding_model, BaseEmbeddings): 
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            from chonkie.embeddings.auto import AutoEmbeddings
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            raise ValueError(
                "ooops! seems like your embedding model is of the wrong type " +
                "currently, this class only supports BaseEmbeddings objects or str objects "+
                f"and it received {type(embedding_model)} object!"
            )

        # Check if the embedding model has been assigned properly
        if self.embedding_model is None:
            raise ImportError(
                "embedding_model is not a valid embedding model",
                "Please install the `semantic` extra to use this feature"
            )
        

        # Keeping the tokenizer the same as the sentence model is important 
        # for the semantic meaning to be calculated properly
        super().__init__(self.embedding_model.get_tokenizer_or_token_counter())

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        """Fast sentence splitting while maintaining accuracy.

        This method is faster than using regex for sentence splitting and is more accurate than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences
            delim: Delimiters to split sentences on
            sep: Separator to use when splitting sentences

        Returns:
            List of sentences

        """
        t = text
        for c in self.delim:
            t = t.replace(c, c + self.sep)

        # Initial split
        splits = [s for s in t.split(self.sep) if s != ""]
        # print(splits)

        # Combine short splits with previous sentence
        sentences = []
        current = ""

        for s in splits:
            if len(s.strip()) < self.min_characters_per_sentence:
                current += s
            else:
                if current:
                    sentences.append(current)
                current = s

        if current:
            sentences.append(current)

        return sentences

    def _get_token_counts(self, sentences: List[str]) -> List[int]:
        """Get token counts for a list of sentences in batch.

        Args:
            sentences: List of sentences

        Returns:
            List of token counts for each sentence

        """
        # Batch encode all sentences at once
        encoded_sentences = self._encode_batch(sentences)
        return [len(encoded) for encoded in encoded_sentences]

    def _estimate_token_counts(self, text: str) -> int:
        """Estimate token count using character length."""
        CHARS_PER_TOKEN = 6.0  # Avg. char per token for llama3 is b/w 6-7
        if type(text) is str:
            return max(1, int(len(text) / CHARS_PER_TOKEN))
        elif type(text) is list and type(text[0]) is str:
            return [max(1, int(len(t) / CHARS_PER_TOKEN)) for t in text]
        else:
            raise ValueError(
                f"Unknown type passed to _estimate_token_count: {type(text)}"
            )
    def _get_feedback(self, estimate: int, actual: int) -> float:
        """Validate against the actual token counts and correct the estimates."""
        feedback = 1 - ((estimate - actual) / estimate)
        return feedback

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the text via Late Chunking."""
        pass
