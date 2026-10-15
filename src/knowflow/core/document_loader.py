"""
KnowFlow AI - Document Loader Module
Extracts text from PDF, DOCX, TXT, MD, and raw inputs with page-level preservation,
SHA-256 deduplication, Unicode integrity, and robust error handling.
"""

import os
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
from PyPDF2 import PdfReader

from src.knowflow.core.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_BYTES


class DocumentLoaderError(Exception):
    """Base exception for document loading and extraction failures."""
    pass


class UnsupportedFileTypeError(DocumentLoaderError):
    """Raised when an unsupported file extension is provided."""
    pass


class EmptyDocumentError(DocumentLoaderError):
    """Raised when a document has zero readable text."""
    pass


class FileSizeLimitError(DocumentLoaderError):
    """Raised when a document exceeds the maximum permitted file size."""
    pass


@dataclass
class DocumentPage:
    """Represents an individual page within a document."""
    page_number: int  # 1-indexed
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class LoadedDocument:
    """Normalized document representation containing extracted pages and metadata."""
    doc_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    sha256: str
    pages: List[DocumentPage]
    metadata: dict = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def total_characters(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in self.pages)


def sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent directory traversal and remove problematic characters."""
    base = os.path.basename(name)
    cleaned = re.sub(r'[^\w\s\.\-]', '_', base).strip()
    return cleaned or "document"


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of binary content."""
    return hashlib.sha256(data).hexdigest()


class DocumentLoader:
    """Production-grade document loader with page tracking and encoding resilience."""

    @classmethod
    def load_from_bytes(cls, file_bytes: bytes, filename: str) -> LoadedDocument:
        """
        Load a document from raw bytes and filename.
        Preserves page boundaries for PDFs and Unicode across all formats.
        """
        sanitized_name = sanitize_filename(filename)
        ext = os.path.splitext(sanitized_name)[-1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if len(file_bytes) == 0:
            raise EmptyDocumentError(f"File '{sanitized_name}' is empty (0 bytes).")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise FileSizeLimitError(
                f"File '{sanitized_name}' exceeds the maximum allowed size of {max_mb} MB."
            )

        sha256_hash = compute_sha256(file_bytes)
        doc_id = f"{sha256_hash[:12]}_{sanitized_name}"

        pages: List[DocumentPage] = []

        if ext == ".pdf":
            pages = cls._extract_pdf(file_bytes, sanitized_name)
        elif ext in [".txt", ".md"]:
            pages = cls._extract_text(file_bytes)
        elif ext == ".docx":
            pages = cls._extract_docx(file_bytes)

        if not pages or all(len(p.text.strip()) == 0 for p in pages):
            raise EmptyDocumentError(
                f"No readable text could be extracted from '{sanitized_name}'. "
                "The file may be corrupted, image-only/scanned, or password-protected."
            )

        return LoadedDocument(
            doc_id=doc_id,
            filename=sanitized_name,
            file_type=ext[1:].upper(),
            file_size_bytes=len(file_bytes),
            sha256=sha256_hash,
            pages=pages,
            metadata={"original_name": filename}
        )

    @classmethod
    def load_from_path(cls, file_path: Union[str, Path]) -> LoadedDocument:
        """Load a document from a local file path."""
        path = Path(file_path)
        if not path.is_file():
            raise DocumentLoaderError(f"File not found: {file_path}")
        with open(path, "rb") as f:
            content = f.read()
        return cls.load_from_bytes(content, path.name)

    @classmethod
    def load_raw_text(cls, text: str, title: str = "Pasted Text") -> LoadedDocument:
        """Load directly supplied text into a LoadedDocument."""
        stripped = text.strip()
        if not stripped:
            raise EmptyDocumentError("The provided text content is empty.")

        encoded = text.encode("utf-8")
        sha256_hash = compute_sha256(encoded)
        clean_title = sanitize_filename(title)
        if not clean_title.endswith(".txt"):
            clean_title = f"{clean_title}.txt"

        doc_id = f"{sha256_hash[:12]}_{clean_title}"
        pages = [DocumentPage(page_number=1, text=stripped)]

        return LoadedDocument(
            doc_id=doc_id,
            filename=clean_title,
            file_type="TXT",
            file_size_bytes=len(encoded),
            sha256=sha256_hash,
            pages=pages,
            metadata={"source_type": "raw_input"}
        )

    @staticmethod
    def _extract_pdf(file_bytes: bytes, filename: str) -> List[DocumentPage]:
        """Extract pages from PDF while maintaining 1-based page numbers."""
        import io
        pages = []
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as e:
                    raise DocumentLoaderError(f"PDF '{filename}' is password-protected and cannot be read.") from e

            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                # Normalize excessive consecutive newlines but preserve paragraph structure
                clean_page = re.sub(r'\n{3,}', '\n\n', page_text).strip()
                if clean_page:
                    pages.append(DocumentPage(page_number=idx + 1, text=clean_page))
        except Exception as e:
            if isinstance(e, DocumentLoaderError):
                raise
            raise DocumentLoaderError(f"Failed to parse PDF '{filename}': {str(e)}") from e
        return pages

    @staticmethod
    def _extract_text(file_bytes: bytes) -> List[DocumentPage]:
        """Extract text from bytes with multi-encoding fallback preserving Unicode."""
        for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
            try:
                decoded = file_bytes.decode(encoding)
                cleaned = re.sub(r'\r\n', '\n', decoded).strip()
                return [DocumentPage(page_number=1, text=cleaned)]
            except UnicodeDecodeError:
                continue
        # Fallback with replacement characters if all failed
        decoded = file_bytes.decode("utf-8", errors="replace").strip()
        return [DocumentPage(page_number=1, text=decoded)]

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> List[DocumentPage]:
        """Extract paragraphs from DOCX into structured logical pages."""
        import io
        import docx
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            return [DocumentPage(page_number=1, text=full_text)]
        except Exception as e:
            raise DocumentLoaderError(f"Failed to read DOCX file: {str(e)}") from e
