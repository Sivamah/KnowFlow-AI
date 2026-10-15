"""
Tests for VectorStoreManager: persistence across restarts, document deletion, and catalog tracking.
"""

import pytest
from src.knowflow.core.chunker import DocumentChunk
from src.knowflow.core.document_loader import DocumentPage, LoadedDocument
from src.knowflow.core.vector_store import VectorStoreManager


def make_dummy_document(doc_id: str, filename: str, texts: list) -> tuple:
    pages = [DocumentPage(page_number=i + 1, text=t) for i, t in enumerate(texts)]
    doc = LoadedDocument(
        doc_id=doc_id,
        filename=filename,
        file_type="TXT",
        file_size_bytes=sum(len(t.encode("utf-8")) for t in texts),
        sha256=f"hash_{doc_id}",
        pages=pages
    )
    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_c{i}",
            doc_id=doc_id,
            filename=filename,
            page_number=i + 1,
            chunk_index=i,
            text=t
        )
        for i, t in enumerate(texts)
    ]
    return doc, chunks


def test_persistence_survives_restarts(tmp_path):
    """
    CRITICAL TEST: Verify that indexing persists to disk and is fully recovered
    when a new VectorStoreManager instance boots up pointing to the same storage folder.
    """
    persist_dir = tmp_path / "chroma_test"

    # Step 1: Initialize first session and index data
    manager1 = VectorStoreManager(persist_dir=persist_dir, collection_name="test_persist")
    doc, chunks = make_dummy_document(
        doc_id="persist_doc_1",
        filename="persistent_notes.txt",
        texts=[
            "ChromaDB persistence is verified across process boundaries.",
            "KnowFlow AI reliably stores document embeddings on disk."
        ]
    )
    indexed_count = manager1.index_document(doc=doc, chunks=chunks)
    assert indexed_count == 2
    assert manager1.collection.count() == 2

    # Step 2: Simulate application shutdown by discarding manager1
    del manager1

    # Step 3: Boot up second session pointing to the same persist_dir
    manager2 = VectorStoreManager(persist_dir=persist_dir, collection_name="test_persist")

    # Verify all records and catalog survive!
    assert manager2.collection.count() == 2
    docs = manager2.get_indexed_documents()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "persist_doc_1"
    assert docs[0]["filename"] == "persistent_notes.txt"
    assert docs[0]["chunk_count"] == 2


def test_document_deletion_removes_vectors_from_chromadb(tmp_path):
    """
    CRITICAL TEST: In the reference codebase, removing a doc only popped it from session_state,
    leaving chunks in ChromaDB. This test verifies that deletion truly purges vectors.
    """
    persist_dir = tmp_path / "chroma_delete_test"
    manager = VectorStoreManager(persist_dir=persist_dir, collection_name="test_delete")

    docA, chunksA = make_dummy_document(
        doc_id="doc_A",
        filename="doc_a.txt",
        texts=["Information regarding Project Alpha.", "Project Alpha budget is ₹500,000."]
    )
    docB, chunksB = make_dummy_document(
        doc_id="doc_B",
        filename="doc_b.txt",
        texts=["Information regarding Project Beta.", "Project Beta deadline is December 2026."]
    )

    manager.index_document(docA, chunksA)
    manager.index_document(docB, chunksB)

    assert manager.collection.count() == 4
    assert len(manager.get_indexed_documents()) == 2

    # Delete Doc A
    deleted = manager.delete_document("doc_A")
    assert deleted is True

    # Total vectors in ChromaDB must now be exactly 2 (only Doc B chunks remain)
    assert manager.collection.count() == 2
    remaining_docs = manager.get_indexed_documents()
    assert len(remaining_docs) == 1
    assert remaining_docs[0]["doc_id"] == "doc_B"

    # Verify no chunks with doc_A remain in ChromaDB
    res = manager.collection.get(where={"doc_id": "doc_A"})
    assert len(res["ids"]) == 0


def test_deduplication_on_reindex(tmp_path):
    """Re-indexing a document should overwrite previous entries rather than duplicate them."""
    persist_dir = tmp_path / "chroma_dedup_test"
    manager = VectorStoreManager(persist_dir=persist_dir, collection_name="test_dedup")

    doc, chunks_v1 = make_dummy_document(
        doc_id="doc_versioned",
        filename="specs.txt",
        texts=["Version 1 of the specification.", "Specification details A."]
    )
    manager.index_document(doc, chunks_v1)
    assert manager.collection.count() == 2

    # Re-index with updated content (3 chunks)
    _, chunks_v2 = make_dummy_document(
        doc_id="doc_versioned",
        filename="specs.txt",
        texts=["Version 2 updated spec.", "New details B.", "New details C."]
    )
    manager.index_document(doc, chunks_v2)

    # Must be 3 chunks, not 2 + 3 = 5!
    assert manager.collection.count() == 3
