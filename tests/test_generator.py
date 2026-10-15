"""
Tests for AnswerGenerator: structured citations, deduplication, missing metadata resilience,
insufficient evidence detection, and guards.
"""

from unittest.mock import MagicMock
import pytest
from src.knowflow.core.generator import AnswerGenerator, Citation
from src.knowflow.core.retriever import RetrievedChunk


def create_mock_generator(response_text: str = "Test response."):
    """Helper to instantiate AnswerGenerator with a mocked Groq client."""
    generator = AnswerGenerator(groq_api_key="mock_key")
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=response_text))
    ]
    generator.client = MagicMock()
    generator.client.chat.completions.create.return_value = mock_response
    generator.get_available_models = MagicMock(return_value=["mock-model"])
    return generator


def test_zero_retrieved_chunks_returns_warning():
    generator = AnswerGenerator(groq_api_key="mock_key")
    result = generator.generate(query="What is quantum gravity?", retrieved_chunks=[])

    assert result.has_sufficient_evidence is False
    assert "No relevant context found" in result.answer
    assert len(result.citations) == 0


def test_insufficient_evidence_detection():
    generator = create_mock_generator(
        "Based on the provided documents, there is insufficient evidence to answer this question regarding dark matter."
    )

    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        filename="report.pdf",
        page_number=3,
        chunk_index=0,
        text="Solar radiation measurements in Q3.",
        relevance_score=0.15
    )

    result = generator.generate(query="What is dark matter?", retrieved_chunks=[chunk])
    assert result.has_sufficient_evidence is False
    assert "insufficient evidence" in result.answer.lower()
    assert len(result.citations) == 1
    assert result.citations[0].filename == "report.pdf"
    assert result.citations[0].page_number == 3


def test_citation_formatting():
    citation = Citation(
        filename="financial_results.pdf",
        page_number=12,
        excerpt="Operating revenue reached ₹45.2 Crore.",
        relevance_score=0.88
    )
    assert citation.label == "financial_results.pdf · Page 12"
    assert citation.source_key == "financial_results.pdf::p12"


def test_citation_deduplication_same_document_and_page():
    """
    CRITICAL TEST: When multiple chunks originate from the same document and same page,
    AnswerGenerator must deduplicate them into exactly one citation badge while preserving
    the highest relevance score and keeping all raw passages in supporting_passages.
    """
    generator = create_mock_generator("The training programme costs ₹15,000 per employee.")

    # 4 chunks all from notes_20261015_110343.txt Page 1 with varying scores
    chunks = [
        RetrievedChunk(
            chunk_id="c1", doc_id="d1", filename="notes_20261015_110343.txt",
            page_number=1, chunk_index=0, text="Training programme costs ₹15,000.",
            relevance_score=0.82
        ),
        RetrievedChunk(
            chunk_id="c2", doc_id="d1", filename="notes_20261015_110343.txt",
            page_number=1, chunk_index=1, text="Each employee receives 20 hours.",
            relevance_score=0.74
        ),
        RetrievedChunk(
            chunk_id="c3", doc_id="d1", filename="notes_20261015_110343.txt",
            page_number=1, chunk_index=2, text="Eligibility is full-time staff.",
            relevance_score=0.88  # Highest score
        ),
        RetrievedChunk(
            chunk_id="c4", doc_id="d1", filename="notes_20261015_110343.txt",
            page_number=1, chunk_index=3, text="Approved by human resources.",
            relevance_score=0.65
        )
    ]

    result = generator.generate(query="How much does the training programme cost?", retrieved_chunks=chunks)

    # Must be deduplicated to exactly ONE citation badge
    assert len(result.citations) == 1
    top_citation = result.citations[0]
    assert top_citation.filename == "notes_20261015_110343.txt"
    assert top_citation.page_number == 1
    assert top_citation.label == "notes_20261015_110343.txt · Page 1"
    # Highest relevance score is preserved
    assert top_citation.relevance_score == 0.88

    # All 4 original evidence passages are preserved in supporting_passages
    assert len(result.supporting_passages) == 4


def test_citation_preserves_multiple_distinct_sources_and_pages():
    """
    Ensure genuinely different documents or pages are preserved and not accidentally merged.
    """
    generator = create_mock_generator("Cross-document summary answer.")

    chunks = [
        RetrievedChunk(
            chunk_id="c1", doc_id="doc_a", filename="policy_handbook.pdf",
            page_number=1, chunk_index=0, text="Section 1: General Conduct",
            relevance_score=0.85
        ),
        RetrievedChunk(
            chunk_id="c2", doc_id="doc_a", filename="policy_handbook.pdf",
            page_number=1, chunk_index=1, text="Section 1 duplicate chunk",
            relevance_score=0.70
        ),
        RetrievedChunk(
            chunk_id="c3", doc_id="doc_a", filename="policy_handbook.pdf",
            page_number=4, chunk_index=5, text="Section 4: Remote Work",
            relevance_score=0.78
        ),
        RetrievedChunk(
            chunk_id="c4", doc_id="doc_b", filename="benefits_guide.docx",
            page_number=2, chunk_index=2, text="Dental and Health insurance",
            relevance_score=0.81
        )
    ]

    result = generator.generate(query="What are the policies?", retrieved_chunks=chunks)

    # 4 chunks -> 3 unique source combinations:
    # 1. policy_handbook.pdf Page 1
    # 2. policy_handbook.pdf Page 4
    # 3. benefits_guide.docx Page 2
    assert len(result.citations) == 3

    labels = [c.label for c in result.citations]
    assert "policy_handbook.pdf · Page 1" in labels
    assert "policy_handbook.pdf · Page 4" in labels
    assert "benefits_guide.docx · Page 2" in labels


def test_citation_handles_missing_page_and_missing_filename_safely():
    """
    Verify safe fallback behavior when page metadata is None/0 or filename is missing.
    """
    # Test Citation properties directly
    c_no_page = Citation(filename="plain_notes.txt", page_number=None, excerpt="Some notes", relevance_score=0.5)
    assert c_no_page.label == "plain_notes.txt"
    assert "Page None" not in c_no_page.label
    assert c_no_page.source_key == "plain_notes.txt::nopage"

    c_zero_page = Citation(filename="data.csv", page_number=0, excerpt="Values", relevance_score=0.5)
    assert c_zero_page.label == "data.csv"
    assert "Page 0" not in c_zero_page.label

    c_no_filename = Citation(filename="", page_number=2, excerpt="Excerpt", relevance_score=0.5)
    assert "Unknown Document" in c_no_filename.label
    assert "Page 2" in c_no_filename.label

    # Test through generator with missing metadata chunks
    generator = create_mock_generator("Answer with missing metadata.")
    chunks = [
        RetrievedChunk(
            chunk_id="c1", doc_id="d1", filename="readme.txt",
            page_number=None, chunk_index=0, text="Readme contents part 1",
            relevance_score=0.70
        ),
        RetrievedChunk(
            chunk_id="c2", doc_id="d1", filename="readme.txt",
            page_number=None, chunk_index=1, text="Readme contents part 2",
            relevance_score=0.75
        )
    ]

    result = generator.generate(query="What is readme?", retrieved_chunks=chunks)
    assert len(result.citations) == 1
    assert result.citations[0].label == "readme.txt"
    assert result.citations[0].relevance_score == 0.75


def test_deduplicate_inline_citations():
    """Verify consecutive duplicate inline citations are collapsed to a single citation."""
    from src.knowflow.core.generator import deduplicate_inline_citations

    # 4 consecutive duplicate inline citations
    raw = (
        "The training programme costs ₹15,000 per employee. "
        "[Doc: notes_20261015_110343.txt, Page: 1] "
        "[Doc: notes_20261015_110343.txt, Page: 1] "
        "[Doc: notes_20261015_110343.txt, Page: 1] "
        "[Doc: notes_20261015_110343.txt, Page: 1]"
    )
    cleaned = deduplicate_inline_citations(raw)
    assert cleaned == "The training programme costs ₹15,000 per employee. [Doc: notes_20261015_110343.txt, Page: 1]"

    # Comma-separated consecutive duplicate inline citations
    raw_comma = (
        "The policy handbook outlines requirements "
        "[Doc: handbook.pdf, Page: 3], [Doc: handbook.pdf, Page: 3]"
    )
    cleaned_comma = deduplicate_inline_citations(raw_comma)
    assert cleaned_comma == "The policy handbook outlines requirements [Doc: handbook.pdf, Page: 3]"

    # Distinct citations are preserved
    distinct = "[Doc: doc_a.txt, Page: 1] and [Doc: doc_b.txt, Page: 2]"
    assert deduplicate_inline_citations(distinct) == distinct

