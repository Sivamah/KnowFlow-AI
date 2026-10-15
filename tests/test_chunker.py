"""
Tests for PageAwareChunker: page preservation, full-coverage (no 6,000-char truncation), and overlap.
"""

import pytest
from src.knowflow.core.chunker import PageAwareChunker, DocumentChunk
from src.knowflow.core.document_loader import LoadedDocument, DocumentPage


def test_page_number_preservation():
    """Verify that chunks created from multi-page documents strictly preserve their originating page numbers."""
    pages = [
        DocumentPage(page_number=1, text="Page 1 discusses initial hypothesis and problem formulation."),
        DocumentPage(page_number=2, text="Page 2 presents the experimental setup, hardware specifications, and data splits."),
        DocumentPage(page_number=3, text="Page 3 analyzes results, benchmark metrics, and ablation studies.")
    ]
    doc = LoadedDocument(
        doc_id="doc_123_test",
        filename="research_paper.pdf",
        file_type="PDF",
        file_size_bytes=1000,
        sha256="dummy_sha256",
        pages=pages
    )

    chunker = PageAwareChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3

    p1_chunks = [c for c in chunks if c.page_number == 1]
    p2_chunks = [c for c in chunks if c.page_number == 2]
    p3_chunks = [c for c in chunks if c.page_number == 3]

    assert len(p1_chunks) >= 1
    assert len(p2_chunks) >= 1
    assert len(p3_chunks) >= 1

    assert "initial hypothesis" in p1_chunks[0].text
    assert "experimental setup" in p2_chunks[0].text
    assert "ablation studies" in p3_chunks[0].text


def test_full_document_coverage_eliminates_6000_char_bug():
    """
    CRITICAL TEST: The reference codebase truncated documents at 6,000 characters:
    text[:6000], silently dropping everything beyond.
    This test verifies that KnowFlow indexes documents of 15,000+ characters with 100% coverage.
    """
    # Create a 15,000 character document with a distinct marker at the very end
    repeated_sentence = "The rapid evolution of retrieval-augmented generation architectures requires verifiable attribution. "
    repetitions = (15000 // len(repeated_sentence)) + 5
    body_text = repeated_sentence * repetitions
    end_marker = "CRITICAL_FINAL_INSIGHT: The final conclusion at character offset 15,200 is verified."
    full_text = body_text + "\n\n" + end_marker

    doc = LoadedDocument(
        doc_id="large_doc_456",
        filename="lengthy_treatise.txt",
        file_type="TXT",
        file_size_bytes=len(full_text.encode("utf-8")),
        sha256="sha_large",
        pages=[DocumentPage(page_number=1, text=full_text)]
    )

    assert len(full_text) > 15000

    chunker = PageAwareChunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_document(doc)

    # Chunks must cover the entire document length
    assert len(chunks) > 20

    # Ensure the final marker at character 15,000+ is fully present in the last chunk
    last_chunk = chunks[-1]
    assert "CRITICAL_FINAL_INSIGHT" in last_chunk.text
    assert "character offset 15,200 is verified" in last_chunk.text


def test_chunk_overlap_preserves_continuity():
    """Ensure consecutive chunks on a page share overlap context."""
    text = (
        "Alpha section describes the initial vector indexing pipeline. "
        "Beta section introduces the Maximal Marginal Relevance reranker for diversity. "
        "Gamma section covers prompt synthesis and hallucination guardrails. "
        "Delta section provides end-to-end latency benchmarks."
    )
    doc = LoadedDocument(
        doc_id="continuity_doc",
        filename="continuity.txt",
        file_type="TXT",
        file_size_bytes=len(text.encode("utf-8")),
        sha256="sha_cont",
        pages=[DocumentPage(page_number=1, text=text)]
    )

    chunker = PageAwareChunker(chunk_size=150, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    # Ensure chunk 0 and chunk 1 share some text
    c0_words = set(chunks[0].text.split())
    c1_words = set(chunks[1].text.split())
    shared = c0_words.intersection(c1_words)
    assert len(shared) > 0, "Adjacent chunks must share overlap words"
