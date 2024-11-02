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

    def _merge_groups(self, groups: List[List[Sentence]]) -> List[Sentence]:
        """Merge the groups together"""
        merged_group = []
        for group in groups:
            merged_group.extend(group)
        return merged_group

    def _skip_and_merge(self,
                        groups: List[List[Sentence]],
                        similarity_threshold: float) -> List[List[Sentence]]:
        """Merge similar groups considering skip window."""
        if len(groups) <= 1:
            return groups

        result_groups = []
        grps = groups.copy()
        embeddings = [self._compute_group_embedding(group) for group in grps]
        
        while grps:
            if len(grps) == 1:
                result_groups.append(grps[0])
                break
                
            # Calculate skip index ensuring it's valid
            skip_index = min(self.skip_window + 1, len(grps) - 1)
            
            # Compare current group with skipped group
            similarity = self._get_semantic_similarity(embeddings[0], embeddings[skip_index])
            
            if similarity >= similarity_threshold:
                # Merge groups from 0 to skip_index (inclusive)
                merged = self._merge_groups(grps[:skip_index + 1])
                
                # Remove the merged groups
                for _ in range(skip_index + 1):
                    grps.pop(0)
                    embeddings.pop(0)
                    
                # Add merged group back at the start
                grps.insert(0, merged)
                embeddings.insert(0, self._compute_group_embedding(merged))
            else:
                # No merge possible, move first group to results
                result_groups.append(grps.pop(0))
                embeddings.pop(0)
        
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