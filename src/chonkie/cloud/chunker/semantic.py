"""Semantic Chunking for Chonkie API."""

import os
from typing import Dict, List, Union, cast

import requests

from chonkie.cloud.chunker.base import CloudChunker


class SemanticChunker(CloudChunker):
    """Semantic Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    API_KEY = os.getenv("CHONKIE_API_KEY")

    def __init__(self,
                 embedding_model: str = "minishlab/potion-base-32M",
                 chunk_size: int = 512,
                 ) -> None:
        """Initialize the SemanticChunker."""
        # Get the API key
        self.api_key = os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Please set the CHONKIE_API_KEY environment variable" +
                             "or pass an API key to the SemanticChunker constructor.")
        
        # Check if the chunk size is valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        
        # Add attributes
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size

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
            "chunk_size": self.chunk_size,
        }
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/semantic",
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