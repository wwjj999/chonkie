"""Base class for all chefs."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import os

class BaseChef(ABC):
    """Base class for all chefs.
    
    This class provides a base for all chefs. Chefs are used to process text
    and handle issues that might cause indigestion to Chonkie. 
    
    Args:
        extensions: The supported extensions.
    
    """

    def __init__(self, 
                 extensions: Optional[List[str]] = None) -> None:
        """Initialize the BaseChef class.
        
        Args:
            extensions: The supported extensions.
        
        """
        self._extensions = extensions

    def _read_file(self, file_path: str) -> str:
        """Read a single file and return its content."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if the file is a supported extension
        if not any(file_path.endswith(ext) for ext in self._extensions):
            raise ValueError(f"File {file_path} is not a supported extension")
        
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            return str(f.read())

    def _read_files(self, file_paths: List[str]) -> List[str]:
        """Read multiple files and return their contents."""
        return [self._read_file(f) for f in file_paths]
    
    def fetch(self,
             files: Optional[Union[str, List[str]]] = None,
             dir: Optional[str] = None) -> Union[str, List[str]]:
        """Fetch the files from the given directory or file paths.

        Reads the file or files from the given directory or file paths and
        returns the content of the file or files.
        
        Args:
            files: The files to fetch.
            dir: The directory to fetch the files from.

        Returns:
            The content of the file or files.

        """
        if files is None and dir is None:
            raise ValueError("Either files or dir must be provided")
        if files is not None and dir is not None:
            raise ValueError("Only one of files or dir can be provided")
        
        # If dir is provided, fetch all the files in the supported extensions
        # If supported extensions are not provided, fetch all the files
        if dir is not None:
            if not os.path.isdir(dir):
                raise FileNotFoundError(f"Directory not found: {dir}")
            
            # Get all files from directory which are supported extensions
            all_files = [os.path.join(dir, f) for f in os.listdir(dir)
                        if os.path.isfile(os.path.join(dir, f)) and 
                        any(f.endswith(ext) for ext in self._extensions)]
            return self._read_files(all_files)
        
        # If single file, then convert to list
        if isinstance(files, str):
            files = [files]
        
        return self._read_files(files)
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean the text.
        
        This method is used to clean the text.
        
        Args:
            text: The text to clean.
        
        Returns:
            The cleaned text.
        
        """
        return text

    def process(self,
                texts: Union[str, List[str]],
                files: Optional[Union[str, List[str]]] = None,
                dir: Optional[str] = None) -> str:
        """Process the text or the files/dir.

        Args:
            texts: The text to process.
            files: The files to process.
            dir: The directory to process the files from.

        Returns:
            The processed text.

        """
        # Text and files are mutually exclusive
        if texts is not None and (files is not None or dir is not None):
            raise ValueError("Either text or files/dir must be provided, not both")
        
        # If files is provided, fetch the files
        if files is not None:
            texts = self.fetch(files, dir)
        
        # Clean the texts
        texts = [self.clean(text) for text in texts]

        # return the cleaned texts
        return texts

    def __call__(self,
                 texts: Optional[Union[str, List[str]]] = None,
                 files: Optional[Union[str, List[str]]] = None,
                 dir: Optional[str] = None) -> str:
        """Call the chef."""
        return self.process(texts, files, dir)
