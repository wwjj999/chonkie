"""SDPM Chunking for Chonkie API."""

import os
from typing import Dict, List, Literal, Optional, Union, cast

import requests

from chonkie.cloud.chunker.base import CloudChunker


class SDPMChunker(CloudChunker):
    """SDPM Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    API_KEY = os.getenv("CHONKIE_API_KEY")

    def __init__(self,
                 embedding_model: str = "minishlab/potion-base-32M",
                 mode: str = "window",
                 threshold: Union[str, float, int] = "auto",
                 chunk_size: int = 512,
                 similarity_window: int = 1,
                 min_sentences: int = 1,
                 min_chunk_size: int = 2,
                 min_characters_per_sentence: int = 12,
                 threshold_step: float = 0.01,
                 delim: Union[str, List[str]] = [".", "!", "?", "\n"],
                 skip_window: int = 1,
                 return_type: Literal["chunks", "texts"] = "chunks",
                 api_key: Optional[str] = None,
                 ) -> None:
        """Initialize the SDPMChunker."""
        # Get the API key
        self.api_key = os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Please set the CHONKIE_API_KEY environment variable" +
                             "or pass an API key to the SDPMChunker constructor.")

        # Check if the chunk size is valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        
        # Check if the similarity window is valid
        if similarity_window <= 0:
            raise ValueError("Similarity window must be greater than 0.")
        
        # Check if the minimum sentences is valid
        if min_sentences <= 0:
            raise ValueError("Minimum sentences must be greater than 0.")

        # Check if the minimum chunk size is valid
        if min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be greater than 0.")
        
        # Check if the minimum characters per sentence is valid
        if min_characters_per_sentence <= 0:
            raise ValueError("Minimum characters per sentence must be greater than 0.")
        
        # Check if the threshold is valid
        if isinstance(threshold, str) and threshold != "auto":
            raise ValueError("Threshold must be either 'auto' or a number between 0 and 1.")
        elif isinstance(threshold, float) and (threshold <= 0 or threshold > 1):
            raise ValueError("Threshold must be between 0 and 1 when a float.")
        elif isinstance(threshold, int) and (threshold <= 1 or threshold > 100):
            raise ValueError("Threshold must be between 1 and 100 when an int.")
        
        # Check if the threshold step is valid
        if threshold_step <= 0:
            raise ValueError("Threshold step must be greater than 0.")

        # Check if the delim is valid
        if not (isinstance(delim, list) and isinstance(delim[0], str)) and not isinstance(delim, str):
            raise ValueError("Delim must be a list of strings or a string.")
        
        # Check if the skip window is valid
        if skip_window <= 0:
            raise ValueError("Skip window must be greater than 0.")
        
        # Check if the return type is valid
        if return_type not in ["chunks", "texts"]:
            raise ValueError("Return type must be either 'chunks' or 'texts'.")

        # Add all the attributes
        self.embedding_model = embedding_model
        self.mode = mode
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.similarity_window = similarity_window
        self.min_sentences = min_sentences
        self.min_chunk_size = min_chunk_size
        self.min_characters_per_sentence = min_characters_per_sentence
        self.threshold_step = threshold_step
        self.delim = delim
        self.skip_window = skip_window
        self.return_type = return_type
        
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
            "threshold": self.threshold,
            "chunk_size": self.chunk_size,
            "similarity_window": self.similarity_window,
            "min_sentences": self.min_sentences,
            "min_chunk_size": self.min_chunk_size,
            "min_characters_per_sentence": self.min_characters_per_sentence,
            "threshold_step": self.threshold_step,
            "delim": self.delim,
            "skip_window": self.skip_window,
            "return_type": self.return_type,
        }
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/sdpm",
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