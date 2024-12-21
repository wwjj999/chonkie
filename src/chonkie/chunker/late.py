"""Class definition for Late Chunking."""

from typing import TYPE_CHECKING, List, Union

from chonkie.embeddings import BaseEmbeddings
from chonkie.types import LateChunk

from .base import BaseChunker

if TYPE_CHECKING:
    import numpy as np

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
        embedding_model: The embedding model to use for the LateChunker
        mode: Mode of the LateChunker to split the text in
        chunk_size: Maximum number of tokens per chunk
        min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
        min_characters_per_sentence: Minimum number of characters per sentence (defaults to 12)
        delim: Delimiters to split sentences on
        **kwargs: Additional keyword arguments to pass to the EmbeddingModel
        
    """  

    def __init__(self,
                 embedding_model: Union[str, BaseEmbeddings] = "all-minilm-l6-v2", 
                 mode: str = "sentence",
                 chunk_size: int = 512, 
                 min_sentences_per_chunk: int = 1, 
                 min_characters_per_sentence: int = 12,
                 delim: Union[str, List[str]] = ['.', '!', '?', '\n'],
                 **kwargs
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
        if min_characters_per_sentence <= 0:
            raise ValueError("min_characters_per_sentence must be a positive non-zero value!")
        if type(delim) not in [str, list]:
            raise TypeError("delim must be of type str or list of str")
        
        self.mode = mode
        self.chunk_size = chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.delim = delim
        self.sep = '🦛'

        # Initialise the embeddings via AutoEmbeddings
        if isinstance(embedding_model, BaseEmbeddings): 
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            from chonkie.embeddings.auto import AutoEmbeddings
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model, **kwargs)
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

    def _create_token_chunks(self,
                            chunk_texts: List[str],
                            token_counts: List[int],
                            decoded_text: str,
                            ) -> List[LateChunk]:
        """Create chunks from a list of texts."""
        # package everything as Chunk objects and send out the result
        chunks = []
        current_index = 0
        for chunk_text, token_count in zip(chunk_texts, token_counts):
            start_index = decoded_text.find(
                chunk_text, current_index
            )  # Find needs to be run every single time because of unknown overlap length
            end_index = start_index + len(chunk_text)
            chunks.append(
                LateChunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count,
                )
            )
            current_index = end_index
        return chunks

    def _token_chunk(self, text: str) -> List[LateChunk]:
        """Split text into overlapping chunks of specified token size.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        # Encode full text
        text_tokens = self._encode(text)

        # We decode the text because the tokenizer might result in a different output than text
        decoded_text = self._decode(text_tokens)

        # Calculate chunk positions
        token_groups = [
            text_tokens[
                start_index : min(start_index + self.chunk_size, len(text_tokens))
            ]
            for start_index in range(
                0, len(text_tokens), self.chunk_size
            )
        ]
        token_counts = [
            len(toks) for toks in token_groups
        ]  # get the token counts; it's prolly chunk_size, but len doesn't take too long

        chunk_texts = self._decode_batch(
            token_groups
        )  # decrease the time by decoding in one go (?)

        chunks = self._create_token_chunks(chunk_texts, token_counts, decoded_text)

        return chunks

    def _split_sentences(self,
                         text: str, 
                         ) -> List[str]:
        """Fast sentence splitting while maintaining accuracy.

        This method is faster than using regex for sentence splitting and is more accurate than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences
            
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
            if len(s.strip()) < self.min_characters_in_sentence:
                current += s
            else:
                if current:
                    sentences.append(current)
                current = s

        if current:
            sentences.append(current)

        return sentences


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
    
    def _sentence_chunk(self, text: str) -> List[LateChunk]:
        """Chunk the text into sentences."""
        pass

    def _get_chunks(self, text: str) -> List[LateChunk]:
        """Get chunks from the text."""
        if self.mode == 'token':
            return self._token_chunk(text)
        elif self.mode == 'sentence':
            return self._sentence_chunk(text)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    # TODO: Add the embedding part here, such that we can get token embeddings
    # for the entire text, and then use the mean of the embeddings to get the chunk embeddings
    def _embedding_split(self, token_embeddings: "np.ndarray", token_counts: List[int]) -> List["np.ndarray"]:
        """Split the embedding into chunks."""
        pass

    def _mean_pool(self, embeddings: "np.ndarray") -> "np.ndarray":
        """Mean pool the embeddings."""
        return np.mean(embeddings, axis=0)

    def chunk(self, text: str) -> List[LateChunk]:
        """Chunk the text via Late Chunking."""
        # Get the chunks first
        chunks = self._get_chunks(text)
        token_counts = [chunk.token_count for chunk in chunks]

        # Get the token embeddings for the entire text
        token_embeddings = ...   # Shape: (n_tokens, embedding_dim)
        chunk_token_embeddings = self._embedding_split(token_embeddings, token_counts)  # Shape: (n_chunks, n_tokens, embedding_dim)

        # Get the chunk embeddings by averaging the token embeddings
        chunk_embeddings = [self._mean_pool(token_emb) for token_emb in chunk_token_embeddings] # Shape: (n_chunks, embedding_dim)

        # Add the chunk embeddings to the chunks
        for chunk, embedding in zip(chunks, chunk_embeddings):
            chunk.embedding = embedding

        return chunks
