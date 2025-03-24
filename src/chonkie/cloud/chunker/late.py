"""Late Chunking for Chonkie API."""

import os
from typing import Dict, List, Literal, Optional, Union, cast

import requests

from chonkie.cloud.chunker.base import CloudChunker


class LateChunker(CloudChunker):
    """Late Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    API_KEY = os.getenv("CHONKIE_API_KEY")
    
    def __init__(self,
                 embedding_model: str = "minishlab/potion-base-32M",
                 mode: str = "sentence",
                 chunk_size: int = 512,
                 min_sentences_per_chunk: int = 1,
                 min_characters_per_sentence: int = 12,
                 approximate: bool = True,
                 delim: Union[str, List[str]] = [".", "!", "?", "\n"],
                 include_delim: Union[Literal["prev", "next"], None] = "prev",
                 api_key: Optional[str] = None,
                 ) -> None:
        """Initialize the LateChunker."""
        # Check for the API key
        self.api_key = os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Please set the CHONKIE_API_KEY environment variable" +
                             "or pass an API key to the LateChunker constructor.")
        
        # Validate the values
        if mode not in ["token", "sentence"]:
            raise ValueError("Mode must be one of the following: ['token', 'sentence']")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive non-zero value!")
        if min_sentences_per_chunk < 0 and mode == "sentence":
            raise ValueError(
                f"min_sentences_per_chunk was assigned {min_sentences_per_chunk}; but it must be non-negative value!"
            )
        if min_characters_per_sentence <= 0:
            raise ValueError(
                "min_characters_per_sentence must be a positive non-zero value!"
            )
        if include_delim not in ["prev", "next", None]:
            raise ValueError("include_delim must be one of the following: ['prev', 'next', None]")
        if not (isinstance(delim, list) and isinstance(delim[0], str)) and not isinstance(delim, str):
            raise ValueError("delim must be a list of strings or a string!")
        if approximate not in [True, False]:
            raise ValueError("approximate must be a boolean!")

        # Add all the attributes
        self.embedding_model = embedding_model
        self.mode = mode
        self.chunk_size = chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.approximate = approximate
        self.delim = delim
        self.include_delim = include_delim
        
        # Check if the API is up right now
        response = requests.get(f"{self.BASE_URL}/")
        if response.status_code != 200:
            raise ValueError("Oh no! You caught Chonkie at a bad time. It seems to be down right now." +
                             "Please try again in a short while." +
                             "If the issue persists, please contact support at support@chonkie.ai.")
        
    def chunk(self, text: Union[str, List[str]]) -> List[Dict]:
        """Chunk the text into a list of chunks."""
        # Make the payload
        payload = {
            "text": text,
            "embedding_model": self.embedding_model,
            "mode": self.mode,
            "chunk_size": self.chunk_size,
            "min_sentences_per_chunk": self.min_sentences_per_chunk,
            "min_characters_per_sentence": self.min_characters_per_sentence,
            "approximate": self.approximate,
            "delim": self.delim,
            "include_delim": self.include_delim,
        }
        
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/late",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        
        # Try to parse the response
        try:
            result: List[Dict]  = cast(List[Dict], response.json())
        except Exception as error:
            raise ValueError("Oh no! The Chonkie API returned an invalid response." +
                             "Please try again in a short while." +
                             "If the issue persists, please contact support at support@chonkie.ai.") from error

        return result
    
    def __call__(self, text: Union[str, List[str]]) -> List[Dict]:
        """Call the chunker."""
        return self.chunk(text)
        
