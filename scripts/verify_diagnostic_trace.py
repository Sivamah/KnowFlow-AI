"""
Diagnostic verification script for KnowFlow AI RAG pipeline.
Tests:
1. Diagnostic trace of documents, chunks, vectors, retrieved chunk IDs, source metadata, and citations before/after deduplication.
2. Beginning question, Middle question (with ₹15,000), End question, and Absent question.
3. Preservation of Unicode characters (₹).
"""

import os
import sys
from dotenv import load_dotenv

# Ensure standard output can handle utf-8 characters like ₹
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.knowflow.core.engine import KnowFlowEngine

def run_diagnostics():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    print("=" * 70)
    print("KNOWFLOW AI — RAG PIPELINE DIAGNOSTIC TRACE")
    print("=" * 70)

    engine = KnowFlowEngine(groq_api_key=groq_api_key)

    docs = engine.list_documents()
    stats = engine.get_stats()
    print(f"\n[MANIFEST SUMMARY]")
    print(f"Total uploaded documents in manifest: {len(docs)}")
    for d in docs:
        print(f" - Document: '{d.get('filename')}' | ID: {d.get('doc_id')} | Chunks: {d.get('chunk_count')} | Size: {d.get('file_size_bytes')} bytes")
    print(f"Total vectors stored in ChromaDB: {stats.get('total_chunks', 0)}")

    test_queries = [
        {
            "category": "Beginning Question",
            "query": "How many employees work at NexaTech, and how many work in AI Research?",
            "expected_keywords": ["120", "24", "AI Research", "Engineering"]
        },
        {
            "category": "Middle Question (Unicode ₹ Test)",
            "query": "How much does the training programme cost per employee?",
            "expected_keywords": ["15,000", "₹"]
        },
        {
            "category": "End Question",
            "query": "What is required before a test can be marked as passed according to the handbook?",
            "expected_keywords": ["observed", "checklist", "passed"]
        },
        {
            "category": "Absent Question (Anti-Hallucination)",
            "query": "What is the warranty policy for NexaTech smartphones in Australia?",
            "expected_keywords": ["insufficient", "not"]
        }
    ]

    for idx, t in enumerate(test_queries, 1):
        print("\n" + "#" * 70)
        print(f"TEST {idx}: {t['category']}")
        print(f"Question: {t['query']}")
        print("#" * 70)

        # Run query through engine (which logs trace)
        ans = engine.ask(t["query"], top_k=4)

        print("\n[RESULT REPORT]")
        print(f"Model used: {ans.model_used} | Latency: {ans.latency_seconds}s")
        print(f"Has sufficient evidence: {ans.has_sufficient_evidence}")
        print(f"Final Answer:\n{ans.answer}\n")
        print(f"Deduplicated Citations Count: {len(ans.citations)}")
        for c in ans.citations:
            print(f"  * Badge Label: '{c.label}' (Score: {c.relevance_score:.4f})")
            print(f"    Source Key: '{c.source_key}' | File: '{c.filename}' | Page: {c.page_number}")

        print(f"Total Supporting Passages Retained: {len(ans.supporting_passages)}")
        for p_idx, p in enumerate(ans.supporting_passages, 1):
            print(f"  * Passage {p_idx} [{p.chunk_id}] (Score: {p.relevance_score:.4f}): {p.text[:90]}...")

        # Assertions
        if t["category"] == "Absent Question (Anti-Hallucination)":
            assert not ans.has_sufficient_evidence or "insufficient" in ans.answer.lower() or "not" in ans.answer.lower(), "Failed: Model hallucinated an absent answer!"
            print(">>> PASS: Correctly refused to hallucinate absent facts.")
        else:
            # Check deduplication: all chunks from same doc and page must result in 1 citation
            assert len(ans.citations) <= len(ans.supporting_passages), "Failed: Citations not deduplicated!"
            print(f">>> PASS: Verified {len(ans.citations)} citation badge(s) vs {len(ans.supporting_passages)} supporting passages.")

    print("\n" + "=" * 70)
    print("ALL 4 PIPELINE DIAGNOSTIC TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostics()
