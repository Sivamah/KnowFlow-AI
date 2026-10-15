"""
KnowFlow AI - Maximal Marginal Relevance (MMR) Retriever
Fetches diverse, highly relevant document passages with cosine normalization,
document filtering, and page metadata preservation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from src.knowflow.core.config import (
    DEFAULT_FETCH_K,
    DEFAULT_LAMBDA_MMR,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD
)
from src.knowflow.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document passage with relevance metrics and citations."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    relevance_score: float
    metadata: dict = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """Standard human-readable citation label."""
        return f"{self.filename} (Page {self.page_number})"


class KnowFlowRetriever:
    """
    Retrieves document passages using Maximal Marginal Relevance (MMR)
    over unit-normalized sentence embeddings.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mmr: float = DEFAULT_LAMBDA_MMR,
        doc_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Execute MMR search for the query against ChromaDB.

        Args:
            query: The user research question.
            top_k: Number of final diverse chunks to return.
            fetch_k: Number of candidate chunks to fetch initially.
            lambda_mmr: Tradeoff between relevance (1.0) and diversity (0.0).
            doc_filter: Optional document ID or filename to restrict scope.

        Returns:
            List of RetrievedChunk sorted by MMR rank.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        collection = self.vector_store.collection
        total_chunks = collection.count()
        if total_chunks == 0:
            return []

        # Encode query with unit normalization
        query_vec = self.vector_store.embed_model.encode(
            [clean_query],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

        # Prepare query filter if requested
        where_filter = None
        if doc_filter:
            # Check if filter matches doc_id or filename
            if any(doc["doc_id"] == doc_filter for doc in self.vector_store.get_indexed_documents()):
                where_filter = {"doc_id": doc_filter}
            else:
                where_filter = {"filename": doc_filter}

        candidate_k = min(fetch_k, total_chunks)

        try:
            results = collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=candidate_k,
                where=where_filter,
                include=["documents", "embeddings", "metadatas"]
            )
        except Exception as e:
            logger.error(f"Vector search query failed: {e}")
            return []

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        docs_list = results["documents"][0]
        embeds = np.array(results["embeddings"][0], dtype=np.float32)
        metas = results["metadatas"][0]
        chunk_ids = results["ids"][0]

        n_candidates = len(docs_list)
        if n_candidates == 0:
            return []

        # Query relevance scores (cosine similarity since vectors are unit normalized)
        query_relevance = np.dot(embeds, query_vec)

        # Maximal Marginal Relevance selection
        selected_indices: List[int] = []
        remaining_indices = list(range(n_candidates))
        final_k = min(top_k, n_candidates)

        for _ in range(final_k):
            if not remaining_indices:
                break

            rem_relevance = query_relevance[remaining_indices]

            if selected_indices:
                # Max similarity between remaining chunks and already selected chunks
                selected_embeds = embeds[selected_indices]
                cross_sim = np.dot(embeds[remaining_indices], selected_embeds.T)
                max_sim_to_selected = np.max(cross_sim, axis=1)
            else:
                max_sim_to_selected = np.zeros(len(remaining_indices))

            # MMR formula
            mmr_scores = lambda_mmr * rem_relevance - (1.0 - lambda_mmr) * max_sim_to_selected

            best_idx_in_remaining = int(np.argmax(mmr_scores))
            chosen_candidate_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(chosen_candidate_idx)
            remaining_indices.pop(best_idx_in_remaining)

        # Build output objects
        retrieved_chunks: List[RetrievedChunk] = []
        for idx in selected_indices:
            meta = metas[idx] or {}
            score = float(query_relevance[idx])
            page_num = int(meta.get("page_number", meta.get("page", 1)))
            doc_id = str(meta.get("doc_id", "unknown"))
            filename = str(meta.get("filename", meta.get("source", "document")))
            chunk_idx = int(meta.get("chunk_index", 0))

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_ids[idx],
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_num,
                    chunk_index=chunk_idx,
                    text=docs_list[idx],
                    relevance_score=round(score, 4),
                    metadata=meta
                )
            )

        return retrieved_chunks
