import re
from dataclasses import dataclass
from typing import Any, List, Union

from .base import BaseChunker, Chunk


@dataclass
class Sentence:
    text: str
    start_index: int
    end_index: int
    token_count: int


@dataclass
class SentenceChunk(Chunk):
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentences: List[Sentence] = None


class SentenceChunker(BaseChunker):
    """
    SentenceChunker splits the sentences in a text based on token limits and sentence boundaries.

    Args:
        tokenizer: The tokenizer instance to use for encoding/decoding
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
    
    Raises:
        ValueError: If parameters are invalid
    """

    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_sentences_per_chunk: int = 1
    ):
        """Initialize the SentenceChunker with configuration parameters.

        SentenceChunker splits the sentences in a text based on token limits and sentence boundaries.

        Args:
            tokenizer: The tokenizer instance to use for encoding/decoding
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if min_sentences_per_chunk < 1:
            raise ValueError("min_sentences_per_chunk must be at least 1")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using enhanced regex patterns.
        
        Handles various cases including:
        - Standard sentence endings across multiple writing systems
        - Quotations and parentheses
        - Common abbreviations
        - Decimal numbers
        - Ellipsis
        - Lists and enumerations
        - Special punctuation
        - Common honorifics and titles

        Args:
            text: Input text to be split into sentences
        
        Returns:
            List of sentences
        """
        # Define sentence ending punctuation marks from various writing systems
        sent_endings = (
            r'[!.?։؟۔܀܁܂߹।॥၊။።፧፨᙮᜵᜶᠃᠉᥄᥅᪨᪩᪪᪫᭚᭛᭞᭟᰻᰼᱾᱿'
            r'‼‽⁇⁈⁉⸮⸼꓿꘎꘏꛳꛷꡶꡷꣎꣏꤯꧈꧉꩝꩞꩟꫰꫱꯫﹒﹖﹗！．？𐩖𐩗'
            r'𑁇𑁈𑂾𑂿𑃀𑃁𑅁𑅂𑅃𑇅𑇆𑇍𑇞𑇟𑈸𑈹𑈻𑈼𑊩𑑋𑑌𑗂𑗃𑗉𑗊𑗋𑗌𑗍𑗎𑗏𑗐𑗑𑗒'
            r'𑗓𑗔𑗕𑗖𑗗𑙁𑙂𑜼𑜽𑜾𑩂𑩃𑪛𑪜𑱁𑱂𖩮𖩯𖫵𖬷𖬸𖭄𛲟𝪈｡。]'
        )
        
        # Common abbreviations and titles that don't end sentences
        abbrevs = (
            r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|viz|al|Gen|Col|Fig|Lt|Mt|St"
            r"|etc|approx|appt|apt|dept|est|min|max|misc|no|ps|seq|temp|etal"
            r"|e\.g|i\.e|vol|vs|cm|mm|km|kg|lb|ft|pd|hr|sec|min|sq|fx|Feb|Mar"
            r"|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
        )
        
        # First, protect periods in known abbreviations
        text_protected = re.sub(rf"({abbrevs})\.", r"\1@POINT@", text, flags=re.IGNORECASE)

        # Protect decimal numbers
        text_protected = re.sub(r"(\d+)\.(\d+)", r"\1@POINT@\2", text_protected)

        # Protect ellipsis
        text_protected = re.sub(r"\.\.\.", "@ELLIPSIS@", text_protected)

        # Protect email addresses and websites
        text_protected = re.sub(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", r"@EMAIL@\1@EMAIL@", text_protected)
        text_protected = re.sub(r"(https?://[^\s]+)", r"@URL@\1@URL@", text_protected)

        # Handle parentheses and brackets
        text_protected = re.sub(r'\([^)]*\.[^)]*\)', lambda m: m.group().replace('.', '@POINT@'), text_protected)
        text_protected = re.sub(r'\[[^\]]*\.[^\]]*\]', lambda m: m.group().replace('.', '@POINT@'), text_protected)

        # Handle quotations with sentence endings
        text_protected = re.sub(rf'({sent_endings})"(\s+[A-Z])', r'\1"\n\2', text_protected)

        # Handle standard sentence endings
        text_protected = re.sub(rf'({sent_endings})(\s+[A-Z"]|\s*$)', r'\1\n\2', text_protected)

        # Handle lists and enumerations
        text_protected = re.sub(r'(\d+\.)(\s+[A-Z])', r'\1\n\2', text_protected)
        text_protected = re.sub(r'([a-zA-Z]\.)(\s+[A-Z])', r'\1\n\2', text_protected)

        # Restore protected periods and symbols
        text_protected = text_protected.replace("@POINT@", ".")
        text_protected = text_protected.replace("@ELLIPSIS@", "...")
        text_protected = re.sub(r'@EMAIL@([^@]+)@EMAIL@', r'\1', text_protected)
        text_protected = re.sub(r'@URL@([^@]+)@URL@', r'\1', text_protected)

        # Split into sentences
        split_sentences = [s.strip() for s in text_protected.split('\n') if s.strip()]

        # Track exact positions
        result_sentences = []
        current_pos = 0

        for sentence in split_sentences:
            start_idx = current_pos
            end_idx = start_idx + len(sentence)

            # Update positions
            result_sentences.append(Sentence(
                text=sentence,
                start_index=start_idx,
                end_index=end_idx,
                token_count=len(self._encode_batch([sentence])[0])  # Get token count for the sentence
            ))
            current_pos = end_idx + 1  # Account for the newline or space separator

        return result_sentences

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

    def _create_chunk(
        self, sentences: List[Sentence], start_idx: int, token_count: int
    ) -> Chunk:
        """Create a chunk from a list of sentences.

        Args:
            sentences: List of sentences to create chunk from
            start_idx: Starting index in original text
            token_count: Total token count for the chunk

        Returns:
            Chunk object
        """
        chunk_text = " ".join([sentence.text for sentence in sentences])
        return SentenceChunk(
            text=chunk_text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
            sentences=sentences,
        )

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into overlapping chunks based on sentences while respecting token limits.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata
        """
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        token_counts = [sentence.token_count for sentence in sentences]

        chunks = []
        current_sentences = []
        current_tokens = 0
        last_chunk_end = 0

        for i, (sentence, token_count) in enumerate(zip(sentences, token_counts)):
            # Calculate total tokens if we add this sentence
            test_tokens = (
                current_tokens + token_count + (1 if current_sentences else 0)
            )  # Add 1 for space between sentences

            can_add_sentence = test_tokens <= self.chunk_size or (
                len(current_sentences) < self.min_sentences_per_chunk
                and len(current_sentences) + 1 <= self.min_sentences_per_chunk
            )

            if can_add_sentence:
                # Sentence fits within limits, add it
                current_sentences.append(sentence)
                current_tokens = test_tokens
            else:
                # Sentence would exceed limits, create chunk if we have enough sentences
                if len(current_sentences) >= self.min_sentences_per_chunk:
                    chunk = self._create_chunk(
                        current_sentences, last_chunk_end, current_tokens
                    )
                    chunks.append(chunk)

                    # Calculate overlap for next chunk
                    if self.chunk_overlap > 0:
                        # Keep sentences from the end of current chunk until we hit overlap limit
                        overlap_sentences = []
                        overlap_tokens = 0
                        for sent, tokens in zip(
                            reversed(current_sentences),
                            reversed(token_counts[i - len(current_sentences) : i]),
                        ):
                            test_overlap_tokens = (
                                overlap_tokens
                                + tokens
                                + (1 if overlap_sentences else 0)
                            )
                            if test_overlap_tokens <= self.chunk_overlap:
                                overlap_sentences.insert(0, sent)
                                overlap_tokens = test_overlap_tokens
                            else:
                                break

                        current_sentences = overlap_sentences
                        current_tokens = overlap_tokens
                    else:
                        current_sentences = []
                        current_tokens = 0

                    last_chunk_end += current_tokens

                # Add current sentence (either after creating chunk or when forced to meet minimum)
                current_sentences.append(sentence)
                current_tokens = (
                    current_tokens
                    + token_count
                    + (1 if len(current_sentences) > 1 else 0)
                )

        # Handle remaining sentences
        if current_sentences:
            chunk = self._create_chunk(
                current_sentences, last_chunk_end, current_tokens
            )
            chunks.append(chunk)

        return chunks

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk})"
        )
