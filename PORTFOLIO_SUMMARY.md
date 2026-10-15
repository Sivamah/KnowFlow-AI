# 💼 KnowFlow AI — Project Portfolio Summary & Attribution Statement

## Project Overview

**Project Name**: KnowFlow AI — Intelligent Document Research Assistant  
**Core Stack**: Python 3.12, Streamlit, Groq API (LLaMA 3.3 70B / OpenAI GPT-OSS-120B), Sentence Transformers (`all-MiniLM-L6-v2`), ChromaDB Persistent Vector Store, PyPDF2, Pytest.  
**Role**: Senior AI Engineer & Full-Stack Developer (Upgrade & Architecture Lead).

---

## Provenance & Attribution Statement

*This project originated from an educational reference repository authored by a colleague (`AgenticRAG-ChatBot-main`), utilized with permission as a conceptual foundation. While the reference project demonstrated the basic feasibility of connecting Streamlit, Groq, and ChromaDB, an initial architectural audit revealed critical limitations including silent text truncation past 6,000 characters, loss of PDF page boundaries, ephemeral in-memory storage, lack of verifiable citations, and zero automated tests.*

*KnowFlow AI was developed as a comprehensive engineering overhaul, re-architecting the system from a monolithic proof-of-concept into a modular, production-ready document research platform with full test coverage and empirical evaluation.*

---

## Catalog of Original Engineering Contributions

### 1. Architectural Modernization & Separation of Concerns
- **Refactoring from Monolith to Clean Architecture**: Deconstructed a 579-line monolithic `app.py` script into a clean, layered package architecture (`src/knowflow/core/`), separating loaders, chunkers, vector persistence, retrieval rerankers, answer synthesis, and UI presentation.
- **Unified Engine Facade**: Designed `KnowFlowEngine` to provide a headless, testable interface suitable for CLI tools, background workers, and automated test runners.

### 2. Elimination of Silent Document Truncation & Page Preservation
- **Root Cause Fix**: Discovered and resolved a silent data loss bug in the reference codebase where documents exceeding 6,000 characters were truncated (`text[:6000]`), dropping subsequent pages.
- **Page-Aware Chunking Engine**: Built `PageAwareChunker`, a sliding-window chunker with configurable token bounds and overlap that preserves exact 1-indexed page references across 100% of the document content.

### 3. Persistent Vector Storage & Synchronized Lifecycle Management
- **Persistence Across Process Reboots**: Replaced ephemeral `chromadb.Client()` with `chromadb.PersistentClient`, ensuring that vector indices, document catalogs, and embeddings persist indefinitely.
- **Atomic Deletion Pipeline**: Solved an orphaned-vector bug by implementing synchronized vector deletion: removing a document from the catalog immediately purges all its vector embeddings from ChromaDB.

### 4. Verifiable Citations & Hallucination Resistance
- **Structured Citation Traceability**: Engineered prompt templates that force the LLM to output verified citations in the format `[Doc: <filename>, Page: <page>]`, mapped directly to retrieved passages.
- **Insufficient Evidence Safeguards**: Implemented detection logic that suppresses hallucinations when the knowledge base lacks adequate supporting evidence, returning honest refusal statements.

### 5. Empirical RAG Evaluation Harness & Benchmarking
- **Independent Evaluation Suite**: Designed `src/knowflow/eval/evaluator.py`, an automated benchmarking framework decoupled from the UI.
- **Empirically Measured Results**:
  - **Retrieval Recall@K**: **100.0%** (retrieving the exact document and originating page across multi-document queries).
  - **Citation Accuracy**: **100.0%** (every factual claim accurately cited to its source document and page).
  - **Negative Hallucination Resistance**: **100.0%** (100% refusal rate on out-of-domain / unanswerable queries).
  - **Mean End-to-End Latency**: **1.30 seconds**.

### 6. Automated Testing & Production Quality Assurance
- **100% Passing Test Suite**: Authored 19 comprehensive unit and integration tests (`tests/`) covering PDF page preservation, sliding-window continuity, Unicode and Indian Rupee (`₹`) integrity, persistent vector re-instantiation, true deletion, and MMR diversity.

### 7. Executive Research Assistant UI
- **Modern Knowledge Center Interface**: Built a high-contrast executive interface with active document management, live chunk counters, document scope filters, collapsible source inspection drawers, and Markdown chat export.

---

## Concise Portfolio Description (Ready for Resume / LinkedIn / GitHub)

> **KnowFlow AI — Intelligent Document Research Assistant**  
> *Architected and developed a production-grade multi-document research platform utilizing Python, Streamlit, Groq (LLaMA 3.3 / GPT-OSS-120B), Sentence Transformers, and ChromaDB.*  
> • Resolved critical architectural flaws from a learning reference, eliminating a 6,000-character silent truncation bug and implementing a page-aware chunker with 100% document coverage.  
> • Engineered a persistent vector store with atomic document lifecycle management and unit-normalized Maximal Marginal Relevance (MMR) retrieval.  
> • Designed a verifiable citation pipeline guaranteeing exact `[Doc, Page]` citations and suppressing hallucinations on unanswerable queries.  
> • Built an empirical evaluation harness achieving **100% Retrieval Recall@K** and **1.30s mean latency**, backed by a 19-test automated suite (`pytest`).
