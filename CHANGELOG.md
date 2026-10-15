# 📝 KnowFlow AI — Comprehensive Changelog & Technical Justification

This document details all modifications, new architectural components, and design decisions introduced during the transformation of the learning reference repository into **KnowFlow AI**.

---

## Summary of Changes

| Category | File | Action | Technical Justification |
| :--- | :--- | :---: | :--- |
| **Core Config** | `src/knowflow/core/config.py` | **New** | Centralizes paths, model names, chunking bounds, and retrieval parameters, eliminating hardcoded constants. |
| **Document Ingestion** | `src/knowflow/core/document_loader.py` | **New** | Implements page-aware PDF extraction (1-indexed), multi-encoding fallback preserving Unicode (₹, accented chars, non-English scripts), SHA-256 deduplication, and input sanitization. |
| **Chunking Engine** | `src/knowflow/core/chunker.py` | **New** | Resolves the critical 6,000-character truncation bug. Processes 100% of document content while retaining exact page numbers and sliding-window overlap. |
| **Persistent Storage** | `src/knowflow/core/vector_store.py` | **New** | Replaces ephemeral in-memory ChromaDB with `chromadb.PersistentClient`. Adds document-level vector deletion and a metadata manifest. |
| **Retrieval Engine** | `src/knowflow/core/retriever.py` | **New** | Refactors MMR with unit-normalized sentence embeddings, page metadata preservation, and document-level search filtering. |
| **Generation Engine** | `src/knowflow/core/generator.py` | **New** | Implements grounded prompt synthesis with explicit `[Doc, Page]` citations, model discovery fallbacks, and insufficient evidence safeguards. |
| **Unified Facade** | `src/knowflow/core/engine.py` | **New** | Provides a clean, testable orchestrator decoupling backend RAG logic from Streamlit UI. |
| **Evaluation Suite** | `src/knowflow/eval/evaluator.py` | **New** | Implements an independent RAG benchmark harness measuring Retrieval Recall@K, Citation Accuracy, Token F1, Hallucination Resistance, and Latency. |
| **Benchmark Data** | `eval_data/benchmark_dataset.json` | **New** | Multi-document evaluation dataset containing factual lookups, financial queries (₹), and unanswerable negative controls. |
| **Evaluation Report** | `EVALUATION_REPORT.md` | **New** | Documents empirical, measured evaluation results (100% Retrieval Recall, 100% Citation Accuracy, 1.30s mean latency). |
| **Automated Tests** | `tests/test_document_loader.py` | **New** | Unit tests for PDF page preservation, Unicode/INR handling, empty files, and error handling. |
| **Automated Tests** | `tests/test_chunker.py` | **New** | Unit tests verifying that documents > 15,000 characters are never truncated and that page numbers are preserved. |
| **Automated Tests** | `tests/test_vector_store.py` | **New** | Integration tests confirming ChromaDB persistence across process restarts and true vector deletion. |
| **Automated Tests** | `tests/test_retriever.py` | **New** | Unit tests for MMR diversity, relevance scores, and document-scoped filtering. |
| **Automated Tests** | `tests/test_generator.py` | **New** | Tests for prompt structure, citation parsing, and insufficient evidence detection. |
| **Presentation Layer** | `app.py` | **Modified** | Re-engineered Streamlit UI to connect to `KnowFlowEngine`, adding live knowledge base stats, document scope filtering, source inspector drawers, and chat export. |
| **Configuration** | `.env.example` | **Modified** | Replaced minimal template with a secure, well-documented configuration guide. |
| **Dependencies** | `requirements.txt` | **Modified** | Added `pytest` for automated test execution. |
| **Git Invariants** | `.gitignore` | **Modified** | Added `.pytest_cache/` and `eval_data/chroma_eval/` to prevent test artifacts from entering source control. |
| **Architecture** | `ARCHITECTURE.md` | **New** | Comprehensive architectural specification and Mermaid pipeline diagrams. |
| **Portfolio** | `PORTFOLIO_SUMMARY.md` | **New** | Honest disclosure of learning reference provenance and catalog of original engineering contributions. |
| **Documentation** | `README.md` | **Modified** | Complete overhaul documenting KnowFlow AI's features, architecture, benchmarks, setup guide, and attribution. |

---

## Detailed Technical Justifications

### 1. Eliminating Silent Data Loss in Chunking
- **Previous Code**:
  ```python
  def agentic_chunk(doc):
      text = doc["text"]
      # ...
      prompt = f"Text:\n{text[:6000]}"
  ```
- **Consequence**: Uploading a 20-page document resulted in the silent loss of all content past the first ~1,000 words.
- **KnowFlow Fix**: Built `PageAwareChunker`, which iterates across all pages of the document, chunking sequentially with sliding-window overlap. Verified by `test_full_document_coverage_eliminates_6000_char_bug` on 15,000+ character documents.

### 2. Preserving Page-Level Boundaries
- **Previous Code**:
  ```python
  text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
  ```
- **Consequence**: Page numbers were stripped at ingestion. Citations could only mention the document filename.
- **KnowFlow Fix**: `DocumentLoader` preserves each PDF page in a discrete `DocumentPage(page_number=idx+1, text=...)` object. These page numbers propagate through `DocumentChunk` into the vector store metadata and final citations.

### 3. Transitioning from Ephemeral to Persistent Vector Store
- **Previous Code**:
  ```python
  chroma = chromadb.Client()  # In-memory ephemeral client
  ```
- **Consequence**: All indexed documents and embeddings were lost upon application reload.
- **KnowFlow Fix**: Integrated `chromadb.PersistentClient(path="./data/chroma_db")` alongside a catalog manifest (`documents_manifest.json`). Verified by `test_persistence_survives_restarts`.

### 4. True Vector Deletion
- **Previous Code**:
  ```python
  st.session_state.docs.pop(i)
  ```
- **Consequence**: Documents were removed from the UI list, but their vectors remained in ChromaDB and continued to answer queries.
- **KnowFlow Fix**: `VectorStoreManager.delete_document(doc_id)` explicitly queries and deletes all corresponding chunk IDs from ChromaDB. Verified by `test_document_deletion_removes_vectors_from_chromadb`.

### 5. Grounded Citations & Hallucination Suppression
- **Previous Code**: Simple prompt with unverified source string tags.
- **KnowFlow Fix**: Formats numbered context passages with page numbers, requires the LLM to provide inline citations `[Doc: <name>, Page: <page>]`, checks for insufficient evidence, and flags unanswerable queries.

### 6. Citation Deduplication & Evidence Preservation
- **Issue**: When MMR retrieval fetched multiple chunks from the same document and page, the citation badge row rendered the identical `[Doc: filename, Page: X]` badge repeatedly.
- **KnowFlow Fix**:
  1. Updated `AnswerGenerator.generate()` in `src/knowflow/core/generator.py` to aggregate citations using a stable key (`fname.lower()::p{page_number}` or `fname.lower()::nopage`), retaining the highest relevance score and excerpt among duplicates.
  2. Preserved all retrieved raw chunks in `GeneratedAnswer.supporting_passages` so no valid evidence is discarded.
  3. Added defensive deduplication to `cite_chips` in `app.py` so the citation row displays each unique document/page combination exactly once.
  4. Handled missing filenames or page numbers safely without rendering `Page None` or crashing.
  5. Added automated unit tests (`tests/test_generator.py`) verifying deduplication, multi-source preservation, and missing metadata fallbacks.
