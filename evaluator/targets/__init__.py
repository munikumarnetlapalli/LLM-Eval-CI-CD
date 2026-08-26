"""Targets package — EvaluationTarget abstraction."""
from .base import EvaluationTarget, TargetResponse
from .mock import MockTarget
from .http_target import HTTPRAGTarget, LocalLLMTarget

__all__ = [
    "EvaluationTarget",
    "TargetResponse",
    "MockTarget",
    "HTTPRAGTarget",
    "LocalLLMTarget",
]
