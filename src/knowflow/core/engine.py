"""
KnowFlow AI - Main Engine Facade
Unifies loading, chunking, persistent vector storage, MMR retrieval,
and grounded citation generation into a clean, testable interface.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.knowflow.core.chunker import DocumentChunk, PageAwareChunker
from src.knowflow.core.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FETCH_K,
    DEFAULT_GROQ_MODEL,
    DEFAULT_LAMBDA_MMR,
    DEFAULT_TOP_K
)
from src.knowflow.core.document_loader import DocumentLoader, LoadedDocument
from src.knowflow.core.generator import AnswerGenerator, GeneratedAnswer
from src.knowflow.core.retriever import KnowFlowRetriever, RetrievedChunk
from src.knowflow.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class KnowFlowEngine:
    """
    High-level facade orchestrating all KnowFlow RAG operations.
    Can be used by Streamlit, CLI tools, evaluation harnesses, or background jobs.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        persist_dir: Optional[Union[str, Path]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        self.vector_store = VectorStoreManager(persist_dir=persist_dir)
        self.chunker = PageAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = KnowFlowRetriever(vector_store=self.vector_store)
        self.groq_api_key = groq_api_key
        self._generator: Optional[AnswerGenerator] = None

        if groq_api_key:
            self._generator = AnswerGenerator(groq_api_key=groq_api_key)

    @property
    def generator(self) -> AnswerGenerator:
        """Access or lazily configure AnswerGenerator."""
        if self._generator is None:
            if not self.groq_api_key:
                raise ValueError("Groq API Key is required for answer generation.")
            self._generator = AnswerGenerator(groq_api_key=self.groq_api_key)
        return self._generator

    def set_groq_api_key(self, api_key: str) -> None:
        """Update Groq API key and reinitialize generator."""
        self.groq_api_key = api_key
        self._generator = AnswerGenerator(groq_api_key=api_key)

    def ingest_document_from_bytes(
        self,
        file_bytes: bytes,
        filename: str
    ) -> Tuple[LoadedDocument, int]:
        """
        Load, page-aware chunk, and persist an uploaded file.
        Returns the LoadedDocument and count of newly indexed chunks.
        """
        doc = DocumentLoader.load_from_bytes(file_bytes=file_bytes, filename=filename)
        chunks = self.chunker.chunk_document(doc)
        stored_count = self.vector_store.index_document(doc=doc, chunks=chunks)
        print(f"[DIAGNOSTIC] Ingested '{filename}': {len(doc.pages)} pages -> {len(chunks)} chunks created -> {stored_count} vectors stored.")
        return doc, stored_count

    def ingest_raw_text(
        self,
        text: str,
        title: str = "Pasted Notes"
    ) -> Tuple[LoadedDocument, int]:
        """
        Ingest raw text directly into the persistent knowledge base.
        """
        doc = DocumentLoader.load_raw_text(text=text, title=title)
        chunks = self.chunker.chunk_document(doc)
        stored_count = self.vector_store.index_document(doc=doc, chunks=chunks)
        print(f"[DIAGNOSTIC] Ingested raw text '{title}': {len(doc.pages)} pages -> {len(chunks)} chunks created -> {stored_count} vectors stored.")
        return doc, stored_count

    def delete_document(self, doc_id: str) -> bool:
        """Delete document vectors from ChromaDB and the catalog."""
        return self.vector_store.delete_document(doc_id=doc_id)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mmr: float = DEFAULT_LAMBDA_MMR,
        doc_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """Retrieve most relevant, diverse passages using MMR."""
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            fetch_k=fetch_k,
            lambda_mmr=lambda_mmr,
            doc_filter=doc_filter
        )

    def ask(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mmr: float = DEFAULT_LAMBDA_MMR,
        doc_filter: Optional[str] = None,
        model: Optional[str] = None
    ) -> GeneratedAnswer:
        """
        End-to-end RAG question answering:
        Retrieves context passages, checks sufficiency, and synthesizes grounded answer with citations.
        """
        docs = self.list_documents()
        stats = self.get_stats()

        chunks = self.search(
            query=query,
            top_k=top_k,
            fetch_k=fetch_k,
            lambda_mmr=lambda_mmr,
            doc_filter=doc_filter
        )

        # Temporary diagnostic output as requested
        print("\n" + "=" * 60)
        print(f"[DIAGNOSTIC TRACE] Query: {query}")
        print(f"1. Number of uploaded documents in manifest: {len(docs)}")
        print(f"2. Total vectors stored in ChromaDB: {stats.get('total_chunks', 0)}")
        print(f"3. Retrieved {len(chunks)} context passages:")
        raw_citations = []
        for idx, c in enumerate(chunks):
            raw_c = f"{c.filename} · Page {c.page_number}"
            raw_citations.append(raw_c)
            print(f"   - Chunk [{idx}] ID: {c.chunk_id}")
            print(f"     Source Metadata: filename='{c.filename}', page={c.page_number}, score={c.relevance_score:.4f}")
            print(f"     Snippet: {c.text[:100]}...")

        print(f"4. Citations BEFORE deduplication ({len(raw_citations)} items): {raw_citations}")

        ans = self.generator.generate(
            query=query,
            retrieved_chunks=chunks,
            model=model
        )

        deduped_labels = [c.label for c in ans.citations]
        print(f"5. Citations AFTER deduplication ({len(deduped_labels)} items): {deduped_labels}")
        print("=" * 60 + "\n")

        return ans

    def list_documents(self) -> List[dict]:
        """List all indexed documents in the persistent catalog."""
        return self.vector_store.get_indexed_documents()

    def get_stats(self) -> dict:
        """Return system and vector store statistics."""
        return self.vector_store.get_collection_stats()

    def clear(self) -> None:
        """Clear all knowledge base data."""
        self.vector_store.clear_all()
