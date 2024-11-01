from typing import List
from .semantic import SemanticChunker, SemanticChunk, Sentence

try:
    import numpy as np
except ImportError:
    raise ImportError("SPDMChunker requires numpy to be installed")

class SPDMChunker(SemanticChunker):
    def __init__(
        self,
        tokenizer,
        sentence_transformer_model: str,
        max_chunk_size: int,
        similarity_threshold: float = None,
        similarity_percentile: float = None,
        initial_sentences: int = 1,
        sentence_mode: str = "heuristic",
        spacy_model: str = "en_core_web_sm",
        skip_window: int = 1  # How many chunks to skip when looking for similarities
    ):
        """Initialize the SPDMChunker.
        
        Args:
            Same as SemanticChunker, plus:
            skip_window: Number of chunks to skip when looking for similarities
        """
        super().__init__(
            tokenizer=tokenizer,
            sentence_transformer_model=sentence_transformer_model,
            max_chunk_size=max_chunk_size,
            similarity_threshold=similarity_threshold,
            similarity_percentile=similarity_percentile,
            initial_sentences=initial_sentences,
            sentence_mode=sentence_mode,
            spacy_model=spacy_model
        )
        self.skip_window = skip_window

    def _merge_similar_groups(
        self, 
        groups: List[List[Sentence]], 
        similarity_threshold: float
    ) -> List[List[Sentence]]:
        """Merge similar groups considering skip window.
        
        Args:
            groups: List of sentence groups
            similarity_threshold: Threshold for merging groups
            
        Returns:
            List of merged sentence groups
        """
        if len(groups) <= 1:
            return groups

        # Track merged groups and their embeddings
        merged = [False] * len(groups)
        embeddings = [self._compute_group_embedding(group) for group in groups]
        result_groups = []

        i = 0
        while i < len(groups):
            if merged[i]:
                i += 1
                continue

            current_group = groups[i].copy()
            current_embedding = embeddings[i]
            
            # Look ahead with skip window
            look_ahead_start = i + 1
            checked_indices = set()

            while look_ahead_start < len(groups):
                # Find next unmerged group to check
                j = look_ahead_start
                while j < min(look_ahead_start + self.skip_window + 1, len(groups)):
                    if not merged[j] and j not in checked_indices:
                        break
                    j += 1

                if j >= len(groups):
                    break

                # Compare embeddings
                similarity = self._get_semantic_similarity(
                    current_embedding,
                    embeddings[j]
                )

                if similarity >= similarity_threshold:
                    # Merge groups
                    # Add all sentences from intermediate groups
                    for k in range(i + 1, j + 1):
                        current_group.extend(groups[k])
                        merged[k] = True

                    # Update current embedding
                    current_embedding = self._compute_group_embedding(current_group)
                    
                    # Move look ahead window
                    look_ahead_start = j + 1
                else:
                    checked_indices.add(j)
                    look_ahead_start = j + 1

            result_groups.append(current_group)
            i += 1

        return result_groups

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into chunks using the SPDM approach.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of SemanticChunk objects
        """
        if not text.strip():
            return []

        # Prepare sentences with precomputed information
        sentences = self._prepare_sentences(text)
        if len(sentences) < self.initial_sentences:
            return [self._create_chunk(sentences)]

        # First pass: Group sentences by semantic similarity
        initial_groups = self._group_sentences(sentences)

        # Get or compute similarity threshold
        if self.similarity_percentile is not None:
            # Compute all pairwise similarities between groups
            all_similarities = []
            for i in range(len(initial_groups) - 1):
                for j in range(i + 1, min(i + self.skip_window + 2, len(initial_groups))):
                    emb1 = self._compute_group_embedding(initial_groups[i])
                    emb2 = self._compute_group_embedding(initial_groups[j])
                    similarity = self._get_semantic_similarity(emb1, emb2)
                    all_similarities.append(similarity)
            
            similarity_threshold = float(np.percentile(all_similarities, self.similarity_percentile))
        else:
            similarity_threshold = self.similarity_threshold

        # Second pass: Merge similar groups with skip window
        merged_groups = self._merge_similar_groups(initial_groups, similarity_threshold)
        
        # Final pass: Split into size-appropriate chunks
        chunks = self._split_chunks(merged_groups)
        
        return chunks

    def __repr__(self) -> str:
        threshold_info = (
            f"similarity_threshold={self.similarity_threshold}"
            if self.similarity_threshold is not None
            else f"similarity_percentile={self.similarity_percentile}"
        )
        return (f"SPDMChunker(max_chunk_size={self.max_chunk_size}, "
                f"{threshold_info}, "
                f"initial_sentences={self.initial_sentences}, "
                f"sentence_mode='{self.sentence_mode}', "
                f"skip_window={self.skip_window})")