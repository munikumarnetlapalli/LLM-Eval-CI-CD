"""
Golden dataset schema and Pydantic models.
Every evaluation case must conform to one of these shapes.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class CaseCategory(str, Enum):
    STRAIGHTFORWARD = "straightforward"
    DIFFICULT = "difficult"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"
    CITATION_REQUIRED = "citation_required"
    LONG_CONTEXT = "long_context"
    EDGE_CASE = "edge_case"
    NUMERICAL = "numerical"
    POLICY = "policy"


class CaseDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DatasetType(str, Enum):
    LLM = "llm"
    RAG = "rag"


class EvaluationCase(BaseModel):
    """Base evaluation case for LLM-only evaluation."""

    id: str = Field(..., description="Unique case identifier, e.g. llm_001")
    question: str = Field(..., min_length=1, description="The input question or prompt")
    expected_answer: str = Field(..., min_length=1, description="Ground-truth expected answer")
    category: CaseCategory
    difficulty: CaseDifficulty
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("id")
    @classmethod
    def id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Case ID must not be empty or whitespace")
        return v


class RAGEvaluationCase(EvaluationCase):
    """Evaluation case for RAG pipeline evaluation.

    Extends EvaluationCase with retrieval-specific fields.
    """

    required_sources: list[str] = Field(
        default_factory=list,
        description="Source documents that must appear in retrieved context",
    )
    context_snippet: Optional[str] = Field(
        None,
        description="Optional ground-truth context excerpt for faithfulness checking",
    )
    rag_query_override: Optional[str] = Field(
        None,
        description="Alternative query string for the retriever (if different from question)",
    )

    @field_validator("required_sources")
    @classmethod
    def sources_must_be_nonempty_strings(cls, v: list[str]) -> list[str]:
        for s in v:
            if not s.strip():
                raise ValueError("Each required_source must be a non-empty string")
        return v


class DatasetManifest(BaseModel):
    """Metadata for a versioned golden dataset.

    case_count and categories are always derived from the cases list.
    They are stored for serialisation but recomputed from cases on load.
    """

    version: str = Field(..., description="Dataset version, e.g. v1")
    dataset_type: DatasetType
    description: str
    created_at: str = Field(..., description="ISO-8601 creation timestamp")
    created_by: str = "system"
    git_commit: Optional[str] = Field(None, description="Git SHA of the dataset commit")
    # Stored but always derived from cases — not validated against cases
    case_count: int = Field(default=0, ge=0)
    categories: list[str] = Field(default_factory=list)
    cases: list[EvaluationCase | RAGEvaluationCase]

    def model_post_init(self, __context) -> None:
        """Recompute case_count and categories from the actual cases list."""
        # Use object.__setattr__ because Pydantic models are immutable by default
        object.__setattr__(self, "case_count", len(self.cases))
        derived_cats = sorted({c.category.value for c in self.cases})
        object.__setattr__(self, "categories", derived_cats)

    @property
    def dataset_type_str(self) -> str:
        return self.dataset_type.value
