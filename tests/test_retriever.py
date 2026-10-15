"""
Tests for KnowFlowRetriever: MMR search, page metadata retention, and document filtering.
"""

import pytest
from src.knowflow.core.chunker import DocumentChunk
from src.knowflow.core.document_loader import DocumentPage, LoadedDocument
from src.knowflow.core.retriever import KnowFlowRetriever
from src.knowflow.core.vector_store import VectorStoreManager


@pytest.fixture
def populated_vector_store(tmp_path):
    persist_dir = tmp_path / "chroma_retriever_test"
    manager = VectorStoreManager(persist_dir=persist_dir, collection_name="test_retriever")

    # Document 1: Physics notes
    doc1 = LoadedDocument(
        doc_id="physics_101",
        filename="physics_notes.pdf",
        file_type="PDF",
        file_size_bytes=500,
        sha256="sha_physics",
        pages=[
            DocumentPage(page_number=1, text="Newton's laws of motion describe the relationship between a body and forces."),
            DocumentPage(page_number=2, text="Quantum mechanics investigates the behavior of matter and light on atomic scales.")
        ]
    )
    chunks1 = [
        DocumentChunk(
            chunk_id="phys_c1", doc_id="physics_101", filename="physics_notes.pdf",
            page_number=1, chunk_index=0, text="Newton's laws of motion describe the relationship between a body and forces."
        ),
        DocumentChunk(
            chunk_id="phys_c2", doc_id="physics_101", filename="physics_notes.pdf",
            page_number=2, chunk_index=1, text="Quantum mechanics investigates the behavior of matter and light on atomic scales."
        )
    ]
    manager.index_document(doc1, chunks1)

    # Document 2: Economics report
    doc2 = LoadedDocument(
        doc_id="econ_202",
        filename="economics_quarterly.txt",
        file_type="TXT",
        file_size_bytes=400,
        sha256="sha_econ",
        pages=[
            DocumentPage(page_number=1, text="Inflation rates stabilized at 3.2% while central banks adjusted repo interest rates.")
        ]
    )
    chunks2 = [
        DocumentChunk(
            chunk_id="econ_c1", doc_id="econ_202", filename="economics_quarterly.txt",
            page_number=1, chunk_index=0, text="Inflation rates stabilized at 3.2% while central banks adjusted repo interest rates."
        )
    ]
    manager.index_document(doc2, chunks2)

    return manager


def test_mmr_retrieval_returns_page_and_score(populated_vector_store):
    retriever = KnowFlowRetriever(vector_store=populated_vector_store)
    results = retriever.retrieve("What did Newton discover about motion?", top_k=2)

    assert len(results) >= 1
    top_hit = results[0]
    assert "Newton" in top_hit.text
    assert top_hit.page_number == 1
    assert top_hit.filename == "physics_notes.pdf"
    assert top_hit.relevance_score > 0.4
    assert top_hit.citation_label == "physics_notes.pdf (Page 1)"


def test_document_scoped_filtering(populated_vector_store):
    retriever = KnowFlowRetriever(vector_store=populated_vector_store)

    # Search economics query with physics filter -> should return physics docs only
    results = retriever.retrieve(
        query="inflation and interest rates",
        top_k=2,
        doc_filter="physics_101"
    )
    for r in results:
        assert r.doc_id == "physics_101"
        assert r.filename == "physics_notes.pdf"


def test_empty_vector_store_returns_empty_list(tmp_path):
    empty_manager = VectorStoreManager(persist_dir=tmp_path / "empty_store", collection_name="empty")
    retriever = KnowFlowRetriever(vector_store=empty_manager)
    results = retriever.retrieve("Hello world?")
    assert results == []
