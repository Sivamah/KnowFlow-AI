"""
KnowFlow AI - Intelligent Document Research Assistant
Full-featured Streamlit UI integrating persistent ChromaDB, page-aware chunking,
unit-normalized MMR retrieval, verifiable citations, and active document management.
"""

import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from src.knowflow.core.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_LAMBDA_MMR,
    CHROMA_PERSIST_DIR
)
from src.knowflow.core.document_loader import (
    DocumentLoader,
    DocumentLoaderError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    FileSizeLimitError
)
from src.knowflow.core.engine import KnowFlowEngine

load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="KnowFlow AI — Document Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
# EXECUTIVE RESEARCH ASSISTANT DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base: #0c0e14;
    --bg-surface: #141824;
    --bg-surface-elevated: #1b2032;
    --border-subtle: #272d42;
    --border-accent: #6c63ff;
    --accent: #6c63ff;
    --accent-glow: rgba(108, 99, 255, 0.2);
    --teal: #2dd4bf;
    --teal-glow: rgba(45, 212, 191, 0.15);
    --amber: #f59e0b;
    --emerald: #10b981;
    --text-primary: #f1f3f9;
    --text-secondary: #9aa1b8;
    --text-muted: #5e6682;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; box-sizing: border-box; }
.stApp { background: var(--bg-base); color: var(--text-primary); }

section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Top Header Bar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    margin-bottom: 20px;
}
.brand-group { display: flex; align-items: center; gap: 12px; }
.brand-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
}
.brand-accent { color: var(--accent); }
.badge-pro {
    background: var(--accent-glow);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid rgba(108, 99, 255, 0.4);
}
.badge-status {
    background: var(--teal-glow);
    color: var(--teal);
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    padding: 4px 10px;
    border-radius: 20px;
}

/* Card Containers */
.card-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 18px;
}

/* Document Chip */
.doc-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.doc-info { display: flex; align-items: center; gap: 10px; }
.doc-icon { font-size: 18px; }
.doc-title { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.doc-meta { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.doc-badge {
    background: rgba(16, 185, 129, 0.12);
    color: var(--emerald);
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 7px;
    border-radius: 4px;
}

/* Chat Container */
.chat-container {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px;
    min-height: 440px;
    max-height: 440px;
    overflow-y: auto;
    margin-bottom: 16px;
}

.empty-chat {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 380px;
    color: var(--text-muted);
    text-align: center;
}
.empty-chat-icon { font-size: 40px; margin-bottom: 12px; }
.empty-chat-title { font-size: 16px; font-weight: 600; color: var(--text-primary); }
.empty-chat-sub { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

/* Chat Bubbles */
.msg-user-row { display: flex; justify-content: flex-end; margin-bottom: 14px; }
.msg-bot-row  { display: flex; justify-content: flex-start; margin-bottom: 18px; }

.bubble-user {
    background: var(--accent);
    color: #ffffff;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.55;
    box-shadow: 0 4px 12px rgba(108, 99, 255, 0.25);
}

.bubble-bot {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: 14px 20px;
    border-radius: 18px 18px 18px 4px;
    max-width: 82%;
    font-size: 14px;
    line-height: 1.6;
}

.cite-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    color: var(--teal);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 8px;
    border-radius: 6px;
    margin: 6px 4px 0 0;
}

.evidence-badge-verified {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: var(--emerald);
    border: 1px solid rgba(16, 185, 129, 0.3);
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 7px;
    border-radius: 4px;
    margin-bottom: 8px;
}

.evidence-badge-insufficient {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    color: var(--amber);
    border: 1px solid rgba(245, 158, 11, 0.3);
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 7px;
    border-radius: 4px;
    margin-bottom: 8px;
}

.msg-meta-row {
    font-size: 10px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Streamlit Button Tweaks */
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 9px 18px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 14px var(--accent-glow) !important;
    transform: translateY(-1px);
}

/* Text Area and Input Controls */
div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

div[data-testid="stTextInput"] input {
    background: var(--bg-surface-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}

/* Expander custom style */
div[data-testid="stExpander"] {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ENGINE INITIALIZATION (CACHED)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def get_engine(api_key: str):
    """Instantiate singleton KnowFlowEngine connected to persistent ChromaDB."""
    return KnowFlowEngine(groq_api_key=api_key, persist_dir=CHROMA_PERSIST_DIR)


# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_doc_filter" not in st.session_state:
    st.session_state.selected_doc_filter = "All Documents"

# ═══════════════════════════════════════════════════════════════════
# API KEY RESOLUTION & VALIDATION
# ═══════════════════════════════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.session_state.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.markdown("""
    <div class="card-panel" style="border-left: 4px solid #6c63ff;">
        <div style="font-size:18px; font-weight:700; color:#f1f3f9; margin-bottom:8px;">
            🔑 Groq API Key Required
        </div>
        <p style="font-size:13px; color:#9aa1b8; margin-bottom:14px;">
            KnowFlow AI requires a Groq API key for high-speed grounded LLM inference (LLaMA 3.3 70B).<br>
            Keys can be acquired free of charge at <a href="https://console.groq.com/keys" target="_blank" style="color:#2dd4bf;">console.groq.com/keys</a>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([5, 1])
    with c1:
        key_input = st.text_input("API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
    with c2:
        if st.button("Connect", use_container_width=True):
            if key_input.strip():
                st.session_state["GROQ_API_KEY"] = key_input.strip()
                st.rerun()
            else:
                st.error("Please provide a valid API key.")
    st.stop()

# Initialize Engine
engine = get_engine(GROQ_API_KEY)
available_models = engine.generator.get_available_models()
stats = engine.get_stats()
indexed_docs = engine.list_documents()

# ═══════════════════════════════════════════════════════════════════
# TOP NAVIGATION & METRICS BAR
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="topbar">
    <div class="brand-group">
        <span style="font-size:24px;">🧠</span>
        <div class="brand-title">KnowFlow<span class="brand-accent">.AI</span></div>
        <div class="badge-pro">RESEARCH ASSISTANT</div>
    </div>
    <div style="display:flex; align-items:center; gap:16px;">
        <div class="badge-status">
            🗃️ {stats['total_documents']} Docs · {stats['total_chunks']} Chunks (Persistent)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN 2-COLUMN LAYOUT
# ═══════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 2], gap="large")

# ─────────────────────────────────────────────────────────────────
# LEFT COLUMN: Document Management & Knowledge Ingestion
# ─────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("### 📁 Document Management")

    # Document Upload Section
    with st.expander("⚡ Upload New Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Supported: PDF (with page tracking), DOCX, TXT, MD"
        )

        if uploaded_files:
            if st.button("Index Uploaded Files", use_container_width=True):
                success_count = 0
                for f in uploaded_files:
                    with st.spinner(f"Ingesting {f.name}..."):
                        try:
                            file_bytes = f.read()
                            doc, chunk_count = engine.ingest_document_from_bytes(
                                file_bytes=file_bytes,
                                filename=f.name
                            )
                            success_count += 1
                        except DocumentLoaderError as dle:
                            st.error(f"Error in {f.name}: {dle}")
                        except Exception as e:
                            st.error(f"Unexpected error in {f.name}: {e}")

                if success_count > 0:
                    st.success(f"✅ Successfully indexed {success_count} document(s) into persistent ChromaDB!")
                    time.sleep(1)
                    st.rerun()

    # Raw Text Ingestion Section
    with st.expander("📝 Paste Text / Notes"):
        text_title = st.text_input("Document Title", placeholder="e.g. Q3 Summary Notes", key="raw_title")
        raw_text_content = st.text_area(
            "Raw text input",
            height=140,
            placeholder="Paste text notes, financial figures with ₹, or research excerpts...",
            label_visibility="collapsed",
            key="raw_text"
        )
        if st.button("Add to Knowledge Base", use_container_width=True):
            if raw_text_content.strip():
                with st.spinner("Chunking & embedding..."):
                    title = text_title.strip() or f"notes_{time.strftime('%Y%m%d_%H%M%S')}.txt"
                    _, stored_chunks = engine.ingest_raw_text(text=raw_text_content, title=title)
                    st.success(f"✅ {stored_chunks} chunks indexed from text notes!")
                    st.rerun()
            else:
                st.warning("Please enter text content before submitting.")

    st.markdown("#### 🗂️ Active Knowledge Base")

    # Scope Filter Dropdown
    doc_options = ["All Documents"] + [d["filename"] for d in indexed_docs]
    st.session_state.selected_doc_filter = st.selectbox(
        "Retrieval Scope",
        options=doc_options,
        index=0,
        help="Select 'All Documents' for cross-document research or narrow down to a specific file."
    )

    if not indexed_docs:
        st.markdown("""
        <div style="text-align:center; padding:24px 10px; color:#5e6682; border:2px dashed #272d42; border-radius:10px; font-size:13px;">
            📂 No documents in knowledge base yet.<br>Upload a PDF, TXT, or DOCX above.
        </div>
        """, unsafe_allow_html=True)
    else:
        for doc in indexed_docs:
            d_col1, d_col2 = st.columns([5, 1])
            with d_col1:
                icon = "📄" if doc["file_type"] == "PDF" else "📝"
                size_kb = round(doc["file_size_bytes"] / 1024, 1)
                st.markdown(f"""
                <div class="doc-chip">
                    <div class="doc-info">
                        <span class="doc-icon">{icon}</span>
                        <div>
                            <div class="doc-title">{doc["filename"]}</div>
                            <div class="doc-meta">{doc["total_pages"]} pgs · {doc["chunk_count"]} chunks · {size_kb} KB</div>
                        </div>
                    </div>
                    <div class="doc-badge">READY</div>
                </div>
                """, unsafe_allow_html=True)
            with d_col2:
                # True document deletion from ChromaDB and Manifest
                if st.button("🗑️", key=f"del_{doc['doc_id']}", help=f"Purge {doc['filename']} from ChromaDB"):
                    engine.delete_document(doc["doc_id"])
                    st.success(f"Removed {doc['filename']}")
                    st.rerun()

        if st.button("Reset / Clear Knowledge Base", help="Completely reset persistent collection"):
            engine.clear()
            st.warning("Cleared persistent knowledge base.")
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# RIGHT COLUMN: Verifiable Research Chat & Source Inspector
# ─────────────────────────────────────────────────────────────────
with col_right:
    chat_top1, chat_top2 = st.columns([3, 1])
    with chat_top1:
        st.markdown("### 💬 Research Assistant")
    with chat_top2:
        selected_model = st.selectbox(
            "Groq Model",
            available_models,
            index=0,
            label_visibility="collapsed"
        )

    # Chat Messages Window
    chat_html = '<div class="chat-container">'
    if not st.session_state.messages:
        chat_html += """
        <div class="empty-chat">
            <div class="empty-chat-icon">🔍</div>
            <div class="empty-chat-title">Ask Questions Across Your Documents</div>
            <div class="empty-chat-sub">Every claim is backed by verifiable document filenames and page citations.</div>
        </div>
        """
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="msg-user-row">
                    <div>
                        <div class="bubble-user">{msg["content"]}</div>
                        <div class="msg-meta-row" style="justify-content:flex-end;">{msg.get("time","")}</div>
                    </div>
                </div>
                """
            else:
                evidence_pill = (
                    '<span class="evidence-badge-verified">✓ VERIFIED CONTEXT</span>'
                    if msg.get("has_evidence", True)
                    else '<span class="evidence-badge-insufficient">⚠️ INSUFFICIENT EVIDENCE DETECTED</span>'
                )

                # Defensive deduplication of citation badges
                seen_labels = set()
                cite_chips_list = []
                for c in msg.get("citations", []):
                    label = c.get("label") or (
                        f"{c.get('filename', 'Unknown')} · Page {c.get('page_number')}"
                        if c.get("page_number")
                        else c.get("filename", "Unknown Document")
                    )
                    if label and label not in seen_labels:
                        seen_labels.add(label)
                        cite_chips_list.append(f'<span class="cite-tag">📄 {label}</span>')

                cite_chips = "".join(cite_chips_list)

                chat_html += f"""
                <div class="msg-bot-row">
                    <div>
                        <div class="bubble-bot">
                            {evidence_pill}<br>
                            <div>{msg["content"]}</div>
                            <div style="margin-top:8px;">{cite_chips}</div>
                        </div>
                        <div class="msg-meta-row">
                            <span>Latency: {msg.get("latency",0)}s · Model: {msg.get("model","")}</span>
                            <span>{msg.get("time","")}</span>
                        </div>
                    </div>
                </div>
                """
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Show Supporting Passages Expander for the last assistant response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_resp = st.session_state.messages[-1]
        passages = last_resp.get("passages") or last_resp.get("citations", [])
        if passages:
            with st.expander(f"🔍 Inspect Supporting Passages ({len(passages)} chunks retrieved)", expanded=False):
                for i, p in enumerate(passages, 1):
                    relevance_pct = round(p.get("relevance", 0.0) * 100, 1)
                    st.markdown(f"**Passage {i}: {p.get('label', 'Source')}** · `Match: {relevance_pct}%`")
                    st.markdown(f"> *\"{p.get('excerpt', '')}\"*")
                    st.divider()

    # Query Input Bar
    in_col1, in_col2 = st.columns([5, 1])
    with in_col1:
        query_input = st.text_input(
            "User query",
            placeholder="Ask a question about your indexed documents...",
            label_visibility="collapsed",
            key="chat_query_input"
        )
    with in_col2:
        send_button = st.button("Search", use_container_width=True)

    if send_button and query_input.strip():
        if not indexed_docs:
            st.warning("Please upload and index at least one document before asking questions.")
        else:
            timestamp = time.strftime("%H:%M")
            st.session_state.messages.append({
                "role": "user",
                "content": query_input.strip(),
                "time": timestamp
            })

            # Check if document scope filter is active
            doc_filter = None
            if st.session_state.selected_doc_filter != "All Documents":
                doc_filter = st.session_state.selected_doc_filter

            with st.spinner("Retrieving verified passages and synthesizing answer..."):
                gen_answer = engine.ask(
                    query=query_input.strip(),
                    doc_filter=doc_filter,
                    model=selected_model
                )

                formatted_citations = []
                seen_keys = set()
                for c in gen_answer.citations:
                    s_key = getattr(c, "source_key", f"{c.filename}::{c.page_number}")
                    if s_key not in seen_keys:
                        seen_keys.add(s_key)
                        formatted_citations.append({
                            "label": c.label,
                            "filename": c.filename,
                            "page_number": c.page_number,
                            "excerpt": c.excerpt,
                            "relevance": c.relevance_score,
                            "source_key": s_key
                        })

                # Retain all retrieved passages for the detailed evidence inspector
                all_evidence = gen_answer.supporting_passages or gen_answer.citations
                formatted_passages = [
                    {
                        "label": (
                            f"{p.filename} · Page {p.page_number}"
                            if getattr(p, "page_number", None)
                            else getattr(p, "filename", "Document")
                        ),
                        "excerpt": getattr(p, "text", getattr(p, "excerpt", ""))[:280].strip() + ("..." if len(getattr(p, "text", getattr(p, "excerpt", ""))) > 280 else ""),
                        "relevance": getattr(p, "relevance_score", 0.0)
                    }
                    for p in all_evidence
                ]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": gen_answer.answer,
                    "has_evidence": gen_answer.has_sufficient_evidence,
                    "citations": formatted_citations,
                    "passages": formatted_passages,
                    "latency": gen_answer.latency_seconds,
                    "model": gen_answer.model_used,
                    "time": timestamp
                })
                st.rerun()

    # Export & Clear Controls
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c_btn2:
        if st.session_state.messages:
            chat_export = "\n\n".join([
                f"**{m['role'].upper()}** ({m.get('time','')}):\n{m['content']}"
                for m in st.session_state.messages
            ])
            st.download_button(
                "Export Chat",
                data=chat_export,
                file_name=f"knowflow_chat_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )