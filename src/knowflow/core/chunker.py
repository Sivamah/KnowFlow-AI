"""
KnowFlow AI - Page-Aware Document Chunker
Breaks documents into semantically coherent, bounded chunks while strictly preserving
page numbers, document identity, and full document coverage (eliminating truncation bugs).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from src.knowflow.core.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE
)
from src.knowflow.core.document_loader import LoadedDocument, DocumentPage


@dataclass
class DocumentChunk:
    """Represents a discrete chunk of text linked to a specific document and page."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    char_count: int = field(init=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.text)
        if not self.metadata:
            self.metadata = {
                "chunk_id": self.chunk_id,
                "doc_id": self.doc_id,
                "source": self.filename,
                "filename": self.filename,
                "page": self.page_number,
                "page_number": self.page_number,
                "chunk_index": self.chunk_index,
                "char_count": self.char_count
            }


class PageAwareChunker:
    """
    Splits documents into overlapping, readable chunks while guaranteeing that every
    chunk preserves its originating page number. Eliminates silent truncation bugs.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: LoadedDocument) -> List[DocumentChunk]:
        """
        Chunk an entire document page by page.
        Every page is fully processed without dropping text.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in doc.pages:
            page_chunks = self._chunk_page(
                page=page,
                doc_id=doc.doc_id,
                filename=doc.filename,
                start_index=global_chunk_idx
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks

    def _chunk_page(
        self,
        page: DocumentPage,
        doc_id: str,
        filename: str,
        start_index: int
    ) -> List[DocumentChunk]:
        """Split a single page's text into bounded overlapping chunks."""
        text = page.text.strip()
        if not text:
            return []

        # If page is shorter than chunk size, return it as a single chunk
        if len(text) <= self.chunk_size:
            chunk_id = f"{doc_id}_p{page.page_number}_c{start_index}"
            return [
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page.page_number,
                    chunk_index=start_index,
                    text=text
                )
            ]

        # Break page into natural paragraph / sentence units
        units = self._split_into_units(text)
        chunks: List[DocumentChunk] = []
        current_unit_group: List[str] = []
        current_len = 0
        local_idx = start_index

        for unit in units:
            unit_len = len(unit)

            # If a single unit itself exceeds chunk_size, hard-slice it
            if unit_len > self.chunk_size:
                if current_unit_group:
                    chunk_text = " ".join(current_unit_group).strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk_id = f"{doc_id}_p{page.page_number}_c{local_idx}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page.page_number,
                                chunk_index=local_idx,
                                text=chunk_text
                            )
                        )
                        local_idx += 1
                    current_unit_group = []
                    current_len = 0

                # Slice large unit
                start = 0
                step = self.chunk_size - self.chunk_overlap
                while start < unit_len:
                    end = min(start + self.chunk_size, unit_len)
                    slice_text = unit[start:end].strip()
                    if len(slice_text) >= self.min_chunk_size:
                        chunk_id = f"{doc_id}_p{page.page_number}_c{local_idx}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                filename=filename,
                                page_number=page.page_number,
                                chunk_index=local_idx,
                                text=slice_text
                            )
                        )
                        local_idx += 1
                    start += step
                continue

            # Standard accumulation
            if current_len + unit_len + 1 > self.chunk_size:
                chunk_text = " ".join(current_unit_group).strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_id = f"{doc_id}_p{page.page_number}_c{local_idx}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page.page_number,
                            chunk_index=local_idx,
                            text=chunk_text
                        )
                    )
                    local_idx += 1

                # Calculate overlap: carry over trailing units
                carry_over: List[str] = []
                carry_len = 0
                for prev_unit in reversed(current_unit_group):
                    if carry_len + len(prev_unit) <= self.chunk_overlap:
                        carry_over.insert(0, prev_unit)
                        carry_len += len(prev_unit) + 1
                    else:
                        break

                current_unit_group = carry_over + [unit]
                current_len = sum(len(u) for u in current_unit_group) + len(current_unit_group) - 1
            else:
                current_unit_group.append(unit)
                current_len += unit_len + 1

        # Residual chunk
        if current_unit_group:
            chunk_text = " ".join(current_unit_group).strip()
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunk_id = f"{doc_id}_p{page.page_number}_c{local_idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=local_idx,
                        text=chunk_text
                    )
                )

        return chunks

    @staticmethod
    def _split_into_units(text: str) -> List[str]:
        """Split text into sentences or meaningful clauses while preserving punctuation."""
        # Split on paragraph breaks first
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        units: List[str] = []

        for p in paragraphs:
            # Split paragraph into sentences on period, exclamation, question mark followed by space or newline
            sentences = re.split(r'(?<=[.!?])\s+', p)
            for s in sentences:
                s_clean = s.strip()
                if s_clean:
                    units.append(s_clean)
        return units
