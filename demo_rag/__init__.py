"""
demo_rag — Lightweight keyword-search RAG for local integration testing.

THIS IS A TEST TARGET ONLY — not the main product.
Uses simple TF-IDF-style keyword search. No embedding model downloads.
No PyTorch, TensorFlow, or CUDA. Pure Python only.
"""
from .retriever import KeywordRetriever
from .generator import DemoGenerator

__all__ = ["KeywordRetriever", "DemoGenerator"]
