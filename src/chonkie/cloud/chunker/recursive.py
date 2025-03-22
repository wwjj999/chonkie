"""Recursive Chunking for Chonkie API."""

import os
from typing import Callable, Dict, List, Union, cast

import requests

from chonkie.cloud.chunker.base import CloudChunker
from chonkie.types import RecursiveRules


class RecursiveChunker(CloudChunker):
    """Recursive Chunking for Chonkie API."""

    BASE_URL = "https://api.chonkie.ai"
    VERSION = "v1"
    API_KEY = os.getenv("CHONKIE_API_KEY")

    def __init__(self,
                 tokenizer_or_token_counter: Union[str, Callable] = "gpt2",
                 chunk_size: int = 512,
                 rules: RecursiveRules = RecursiveRules(),
                 api_key: Union[str, None] = None,
                 ) -> None:
        """Initialize the RecursiveChunker."""
        # If no API key is provided, use the environment variable
        self.api_key = api_key or os.getenv("CHONKIE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Please set the CHONKIE_API_KEY environment variable" +
                             "or pass an API key to the RecursiveChunker constructor.")

        # Check if the chunk size is valid
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        
        # Add attributes
        self.tokenizer_or_token_counter = tokenizer_or_token_counter
        self.chunk_size = chunk_size
        self.rules = rules

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
            "tokenizer_or_token_counter": self.tokenizer_or_token_counter,
            "chunk_size": self.chunk_size,
            "rules": self.rules.to_dict(),
        }
        # Make the request to the Chonkie API
        response = requests.post(
            f"{self.BASE_URL}/{self.VERSION}/chunk/recursive",
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
        """Call the RecursiveChunker."""
        return self.chunk(text)