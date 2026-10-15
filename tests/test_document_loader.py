"""
Tests for DocumentLoader: formats, page extraction, Unicode, and error handling.
"""

import io
import pytest
from PyPDF2 import PdfWriter
from src.knowflow.core.document_loader import (
    DocumentLoader,
    EmptyDocumentError,
    FileSizeLimitError,
    UnsupportedFileTypeError,
    sanitize_filename
)


def create_sample_pdf_bytes(pages_text: list) -> bytes:
    """Helper to generate a multi-page PDF in memory."""
    # Since reportlab might not be installed, we can build a minimal valid PDF using PyPDF2 if possible,
    # or create a minimal raw PDF byte stream with text objects.
    # Let's write minimal PDF stream or use PyPDF2 if it supports writing, or build a clean text PDF:
    buffer = io.BytesIO()
    writer = PdfWriter()
    for text in pages_text:
        writer.add_blank_page(width=612, height=792)
    writer.write(buffer)
    # PyPDF2 blank pages have no text, so let's verify if writer can add text or test with mock/real text
    return buffer.getvalue()


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("my report (2026) #1.pdf") == "my report _2026_ _1.pdf"
    assert sanitize_filename("") == "document"


def test_load_txt_with_unicode_and_currency():
    """Ensure Indian Rupee ₹, accents, and non-English scripts are perfectly preserved."""
    sample_text = (
        "Quarterly Financial Statement 2026\n"
        "Total Revenue: ₹14,500,000.50 (Indian Rupees)\n"
        "Net Profit: ₹3,200,000.00\n"
        "International Operations: €450,000 and $520,000\n"
        "Foreign notes: naïve façade café & हिंदी (Hindi) / தமிழ் (Tamil) scripts."
    )
    raw_bytes = sample_text.encode("utf-8")
    doc = DocumentLoader.load_from_bytes(raw_bytes, "finance_report.txt")

    assert doc.filename == "finance_report.txt"
    assert doc.file_type == "TXT"
    assert doc.total_pages == 1
    assert "₹14,500,000.50" in doc.pages[0].text
    assert "हिंदी" in doc.pages[0].text
    assert "தமிழ்" in doc.pages[0].text
    assert len(doc.sha256) == 64


def test_load_markdown():
    md_content = "# System Architecture\n\nKnowFlow AI utilizes persistent ChromaDB.\n\n## Subsections\nMMR retrieval."
    raw_bytes = md_content.encode("utf-8")
    doc = DocumentLoader.load_from_bytes(raw_bytes, "architecture.md")

    assert doc.file_type == "MD"
    assert doc.total_pages == 1
    assert "persistent ChromaDB" in doc.pages[0].text


def test_empty_file_raises_error():
    with pytest.raises(EmptyDocumentError):
        DocumentLoader.load_from_bytes(b"", "empty.txt")

    with pytest.raises(EmptyDocumentError):
        DocumentLoader.load_from_bytes(b"   \n\t  ", "spaces.txt")


def test_unsupported_file_type():
    with pytest.raises(UnsupportedFileTypeError):
        DocumentLoader.load_from_bytes(b"some content", "spreadsheet.xlsx")

    with pytest.raises(UnsupportedFileTypeError):
        DocumentLoader.load_from_bytes(b"print('hello')", "script.py")


def test_load_raw_text():
    text = "Key findings: The experimental accuracy improved by 14.8% after fine-tuning."
    doc = DocumentLoader.load_raw_text(text, title="Experiment Notes")

    assert doc.filename == "Experiment Notes.txt"
    assert doc.total_pages == 1
    assert "14.8%" in doc.pages[0].text
    assert doc.metadata["source_type"] == "raw_input"


def test_file_size_limit():
    huge_bytes = b"X" * (26 * 1024 * 1024)  # 26 MB exceeds 25 MB limit
    with pytest.raises(FileSizeLimitError):
        DocumentLoader.load_from_bytes(huge_bytes, "huge_file.txt")
