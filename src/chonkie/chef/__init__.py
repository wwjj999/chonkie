"""A module for handling text files."""

from .base import BaseChef
from .patterns import Abbreviations, UnicodeReplacements
from .text import TextChef

__all__ = ['BaseChef', 
           'TextChef',
           'Abbreviations',
           'UnicodeReplacements']
