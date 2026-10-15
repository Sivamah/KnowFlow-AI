"""
KnowFlow AI - Grounded Citation Generator
Generates verifiable, context-grounded answers with document and page citations,
strict hallucination avoidance, and insufficient evidence detection.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional
from groq import Groq

from src.knowflow.core.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_TEMPERATURE,
    FALLBACK_GROQ_MODELS,
    MAX_GENERATION_TOKENS,
    MIN_SIMILARITY_THRESHOLD
)
from src.knowflow.core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Verifiable reference to a specific document and page."""
    filename: str
    page_number: Optional[int] = None
    excerpt: str = ""
    relevance_score: float = 0.0

    @property
    def label(self) -> str:
        clean_name = (self.filename or "Unknown Document").strip()
        if self.page_number is not None and str(self.page_number).strip().isdigit() and int(self.page_number) > 0:
            return f"{clean_name} · Page {int(self.page_number)}"
        return clean_name

    @property
    def source_key(self) -> str:
        clean_name = (self.filename or "unknown").strip().lower()
        if self.page_number is not None and str(self.page_number).strip().isdigit() and int(self.page_number) > 0:
            return f"{clean_name}::p{int(self.page_number)}"
        return f"{clean_name}::nopage"


@dataclass
class GeneratedAnswer:
    """Structured response from the KnowFlow research assistant."""
    query: str
    answer: str
    citations: List[Citation]
    has_sufficient_evidence: bool
    model_used: str
    latency_seconds: float
    raw_prompt: Optional[str] = None
    supporting_passages: List[RetrievedChunk] = field(default_factory=list)


class AnswerGenerator:
    """
    Coordinates prompt construction, Groq LLM inference, grounded citation checking,
    and insufficient-evidence safeguards.
    """

    INSUFFICIENT_EVIDENCE_PHRASES = [
        "insufficient evidence",
        "i do not have enough information",
        "i don't have enough information",
        "not mentioned in the provided",
        "context does not contain",
        "not found in the provided",
        "provided documents do not provide",
        "provided text does not mention"
    ]

    def __init__(self, groq_api_key: str, default_model: str = DEFAULT_GROQ_MODEL):
        self.groq_api_key = groq_api_key
        self.client = Groq(api_key=groq_api_key)
        self.default_model = default_model

    def get_available_models(self) -> List[str]:
        """Fetch models available on Groq, prioritized for performance."""
        try:
            available = [m.id for m in self.client.models.list().data]
            valid = [m for m in FALLBACK_GROQ_MODELS if m in available]
            if valid:
                return valid
            # Filter non-audio models
            general = [m for m in available if "whisper" not in m and "guard" not in m]
            return general or [self.default_model]
        except Exception as e:
            logger.warning(f"Could not list Groq models: {e}. Falling back to default list.")
            return FALLBACK_GROQ_MODELS

    def generate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> GeneratedAnswer:
        """
        Synthesize an answer using ONLY the retrieved chunks.
        Inserts page and document citations and flags insufficient evidence.
        """
        start_time = time.time()
        active_model = model or self.default_model
        available = self.get_available_models()
        if active_model not in available and available:
            active_model = available[0]

        # Guard 1: Zero chunks retrieved
        if not retrieved_chunks:
            return GeneratedAnswer(
                query=query,
                answer="⚠️ **No relevant context found.** Please upload or index relevant documents before asking this question.",
                citations=[],
                has_sufficient_evidence=False,
                model_used=active_model,
                latency_seconds=round(time.time() - start_time, 3)
            )

        # Build structured context with numbered passages and explicit page citations
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            block = (
                f"[Passage {idx}]\n"
                f"Document: {chunk.filename}\n"
                f"Page: {chunk.page_number}\n"
                f"Relevance: {chunk.relevance_score:.2f}\n"
                f"Content:\n{chunk.text}"
            )
            context_blocks.append(block)

        context_str = "\n\n" + ("=" * 40) + "\n\n".join(context_blocks) + "\n\n" + ("=" * 40)

        # Construct grounded prompt
        system_instruction = (
            "You are KnowFlow AI, an intelligent, verifiable document research assistant.\n"
            "Your objective is to provide high-precision, factual answers based STRICTLY on the provided context passages.\n\n"
            "Mandatory Rules:\n"
            "1. Answer using ONLY the information stated in the context passages.\n"
            "2. For every factual claim, stat, or statement, attach an inline citation in the form: [Doc: <filename>, Page: <page>].\n"
            "3. Do not invent page numbers, filenames, or external facts under any circumstances.\n"
            "4. If the context does not contain adequate evidence to answer the question, state: "
            "'Based on the provided documents, there is insufficient evidence to answer this question.' "
            "and briefly describe what is missing.\n"
            "5. Preserve exact numbers, dates, currency signs (such as ₹, $, €), and terminology."
        )

        user_prompt = (
            f"Context Passages:\n{context_str}\n\n"
            f"Research Question: {query}\n\n"
            f"Grounded Answer (with [Doc: <filename>, Page: <page>] citations):"
        )

        try:
            response = self.client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=MAX_GENERATION_TOKENS
            )
            raw_answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return GeneratedAnswer(
                query=query,
                answer=f"❌ **Error generating response from LLM**: {str(e)}",
                citations=[],
                has_sufficient_evidence=False,
                model_used=active_model,
                latency_seconds=round(time.time() - start_time, 3)
            )

        latency = round(time.time() - start_time, 3)

        # Check for insufficient evidence declarations
        lower_answer = raw_answer.lower()
        has_sufficient_evidence = not any(
            phrase in lower_answer for phrase in self.INSUFFICIENT_EVIDENCE_PHRASES
        )

        # If highest relevance score is critically low and answer looks generic, flag it
        max_score = max(c.relevance_score for c in retrieved_chunks)
        if max_score < MIN_SIMILARITY_THRESHOLD and not has_sufficient_evidence:
            has_sufficient_evidence = False

        # Assemble deduplicated verified citations from retrieved passages
        citation_map = {}
        for c in retrieved_chunks:
            fname = (c.filename or "Unknown Document").strip()
            p_num = c.page_number if (c.page_number is not None and c.page_number > 0) else None
            key = f"{fname.lower()}::p{p_num}" if p_num is not None else f"{fname.lower()}::nopage"
            excerpt_snippet = c.text[:220].strip() + ("..." if len(c.text) > 220 else "")

            if key not in citation_map:
                citation_map[key] = Citation(
                    filename=fname,
                    page_number=p_num,
                    excerpt=excerpt_snippet,
                    relevance_score=c.relevance_score
                )
            else:
                existing = citation_map[key]
                if c.relevance_score > existing.relevance_score:
                    citation_map[key] = Citation(
                        filename=fname,
                        page_number=p_num,
                        excerpt=excerpt_snippet,
                        relevance_score=c.relevance_score
                    )

        deduped_citations = list(citation_map.values())

        return GeneratedAnswer(
            query=query,
            answer=raw_answer,
            citations=deduped_citations,
            has_sufficient_evidence=has_sufficient_evidence,
            model_used=active_model,
            latency_seconds=latency,
            raw_prompt=user_prompt,
            supporting_passages=retrieved_chunks
        )
