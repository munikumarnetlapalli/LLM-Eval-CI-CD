"""
Keyword-based retriever for the demo RAG target.
Pure Python — no embedding models, no downloads, no torch.
Uses simple word-overlap (Jaccard-like) scoring.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class KeywordRetriever:
    """Retrieves document chunks using keyword overlap scoring.

    This is a test-only retriever. It is intentionally simple —
    the goal is to enable local integration tests, not to demonstrate
    production retrieval quality.
    """

    def __init__(
        self,
        knowledge_base_path: str | Path = "demo_rag/knowledge_base",
        top_k: int = 3,
    ):
        self._top_k = top_k
        self._chunks: list[dict] = []
        self._load_knowledge_base(Path(knowledge_base_path))

    def _load_knowledge_base(self, kb_path: Path) -> None:
        """Load chunks from JSON files in the knowledge base directory."""
        if not kb_path.exists():
            # Fall back to in-memory demo data if KB path doesn't exist
            self._chunks = self._demo_chunks()
            return

        for json_file in kb_path.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for chunk in data.get("chunks", []):
                        self._chunks.append({
                            "text": chunk.get("text", ""),
                            "source": chunk.get("source", json_file.stem),
                        })
            except (json.JSONDecodeError, KeyError):
                continue

        if not self._chunks:
            self._chunks = self._demo_chunks()

    def _demo_chunks(self) -> list[dict]:
        """Built-in demo knowledge base — used when no KB files exist."""
        return [
            {
                "text": "Customer records are retained for 7 years after the end of the customer relationship, in compliance with financial regulations.",
                "source": "data-retention-policy.pdf",
            },
            {
                "text": "New employees receive 15 vacation days in their first year, increasing to 20 days after 3 years of service.",
                "source": "employee-handbook.pdf",
            },
            {
                "text": "Capital expenditures over $50,000 require approval from the department head, CFO, and CEO, with a mandatory 10-business-day review period.",
                "source": "finance-policy.pdf",
            },
            {
                "text": "After 48 hours, unresolved complaints are escalated to the Customer Success Manager, then to the VP of Customer Experience if still unresolved after 72 hours total.",
                "source": "customer-support-sop.pdf",
            },
            {
                "text": "The corporate email system allows a maximum attachment size of 25 MB per email.",
                "source": "it-acceptable-use-policy.pdf",
            },
            {
                "text": "Production database credentials must be rotated every 90 days and may not be reused for at least 12 password cycles.",
                "source": "security-policy.pdf",
            },
            {
                "text": "Priority 1 incidents in the production environment have a 15-minute response SLA and a 4-hour resolution SLA.",
                "source": "incident-management-policy.pdf",
            },
            {
                "text": "All unvested stock options are immediately forfeited upon termination for cause, with no grace period or accelerated vesting.",
                "source": "equity-compensation-plan.pdf",
            },
            {
                "text": "The primary caregiver receives 16 weeks (112 days) of fully paid parental leave.",
                "source": "benefits-guide-2026.pdf",
            },
            {
                "text": "All vendor contracts with a total value exceeding $100,000 or a term exceeding 24 months require mandatory review by the Legal department.",
                "source": "vendor-management-policy.pdf",
            },
        ]

    def _tokenize(self, text: str) -> set[str]:
        """Simple lowercase word tokenization."""
        return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))

    def _score(self, query_tokens: set[str], chunk_text: str) -> float:
        """Jaccard-like overlap score between query and chunk."""
        chunk_tokens = self._tokenize(chunk_text)
        if not chunk_tokens:
            return 0.0
        intersection = query_tokens & chunk_tokens
        return len(intersection) / (len(query_tokens | chunk_tokens) or 1)

    def retrieve(self, query: str) -> tuple[list[str], list[str]]:
        """Return top-k chunk texts and their source filenames."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [], []

        scored = [
            (self._score(query_tokens, c["text"]), c["text"], c["source"])
            for c in self._chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._top_k]

        texts = [t for _, t, _ in top if t]
        sources = list(dict.fromkeys(s for _, _, s in top if s))  # deduplicate, preserve order
        return texts, sources
