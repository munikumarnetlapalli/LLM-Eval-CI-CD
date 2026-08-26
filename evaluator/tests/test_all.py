"""
Comprehensive unit and integration tests for LLM Eval CI/CD.

Covers: providers, targets, datasets, metrics, runner, regression,
quality gate, comparison, CLI exit codes, storage, configuration.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ─── Provider Tests ───────────────────────────────────────────────────────────

class TestMockProvider:
    def test_complete_returns_response(self):
        from evaluator.providers.mock import MockProvider
        from evaluator.providers.base import Message
        p = MockProvider(response_text="test answer", input_tokens=100, output_tokens=50)
        resp = p.complete([Message(role="user", content="What is the policy?")])
        assert resp.content == "test answer"
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50
        assert resp.latency_ms > 0

    def test_complete_fail_raises(self):
        from evaluator.providers.mock import MockProvider
        from evaluator.providers.base import Message
        p = MockProvider(fail=True, fail_message="provider down")
        with pytest.raises(RuntimeError, match="provider down"):
            p.complete([Message(role="user", content="test")])

    def test_embed_returns_vector(self):
        from evaluator.providers.mock import MockProvider
        p = MockProvider()
        embedding = p.embed("test text")
        assert len(embedding) == 1536
        assert all(isinstance(v, float) for v in embedding)

    def test_call_count(self):
        from evaluator.providers.mock import MockProvider
        from evaluator.providers.base import Message
        p = MockProvider()
        p.complete([Message(role="user", content="q1")])
        p.complete([Message(role="user", content="q2")])
        assert p.call_count == 2

    def test_cost_estimation(self):
        from evaluator.providers.mock import MockProvider
        from evaluator.providers.base import Message, LLMResponse
        p = MockProvider(input_tokens=1000, output_tokens=500)
        resp = p.complete([Message(role="user", content="test")])
        cost = resp.estimated_cost(p.pricing)
        assert cost > 0
        assert isinstance(cost, float)

    def test_factory_builds_mock(self):
        from evaluator.providers.factory import build_provider
        p = build_provider({"provider": "mock"})
        from evaluator.providers.mock import MockProvider
        assert isinstance(p, MockProvider)

    def test_factory_invalid_provider_raises(self):
        from evaluator.providers.factory import build_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            build_provider({"provider": "invalid-xyz"})


# ─── Target Tests ─────────────────────────────────────────────────────────────

class TestMockTarget:
    def test_query_returns_response(self):
        from evaluator.targets.mock import MockTarget
        t = MockTarget(answer="The policy is X", latency_ms=100.0)
        resp = t.query("What is the policy?")
        assert resp.answer == "The policy is X"
        assert resp.latency_ms == 100.0
        assert not resp.failed

    def test_query_fail_returns_error(self):
        from evaluator.targets.mock import MockTarget
        t = MockTarget(fail=True, fail_message="service down")
        resp = t.query("test")
        assert resp.failed
        assert resp.error == "service down"
        assert resp.answer == ""

    def test_has_context(self):
        from evaluator.targets.mock import MockTarget
        t = MockTarget(context=["chunk1", "chunk2"])
        resp = t.query("question")
        assert resp.has_context
        assert len(resp.retrieved_context) == 2

    def test_call_count(self):
        from evaluator.targets.mock import MockTarget
        t = MockTarget()
        t.query("q1")
        t.query("q2")
        assert t.call_count == 2


# ─── Dataset Tests ────────────────────────────────────────────────────────────

class TestDatasetSchema:
    def test_rag_case_valid(self):
        from evaluator.datasets.schema import RAGEvaluationCase, CaseCategory, CaseDifficulty
        case = RAGEvaluationCase(
            id="rag_001",
            question="What is the retention policy?",
            expected_answer="7 years",
            required_sources=["policy.pdf"],
            category=CaseCategory.POLICY,
            difficulty=CaseDifficulty.MEDIUM,
        )
        assert case.id == "rag_001"
        assert len(case.required_sources) == 1

    def test_empty_id_raises(self):
        from evaluator.datasets.schema import EvaluationCase, CaseCategory, CaseDifficulty
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            EvaluationCase(
                id="   ",
                question="test",
                expected_answer="answer",
                category=CaseCategory.STRAIGHTFORWARD,
                difficulty=CaseDifficulty.EASY,
            )


class TestDatasetLoader:
    def test_load_golden_dataset(self):
        from evaluator.datasets.loader import DatasetLoader
        loader = DatasetLoader()
        manifest = loader.load("datasets/v1/golden.json")
        assert manifest.version == "v1"
        assert manifest.case_count == 30
        assert len(manifest.cases) == 30

    def test_load_nonexistent_raises(self):
        from evaluator.datasets.loader import DatasetLoader
        with pytest.raises(FileNotFoundError):
            DatasetLoader().load("datasets/nonexistent/golden.json")


class TestDatasetValidator:
    def test_valid_dataset_passes(self):
        from evaluator.datasets.loader import DatasetLoader
        from evaluator.datasets.validator import DatasetValidator
        manifest = DatasetLoader().load("datasets/v1/golden.json")
        result = DatasetValidator().validate(manifest)
        assert result.valid, f"Validation errors: {result.errors}"
        assert len(result.errors) == 0

    def test_duplicate_ids_fail(self):
        from evaluator.datasets.schema import (
            EvaluationCase, CaseCategory, CaseDifficulty,
            DatasetManifest, DatasetType
        )
        from evaluator.datasets.validator import DatasetValidator
        case = EvaluationCase(
            id="dup_001", question="q", expected_answer="a",
            category=CaseCategory.STRAIGHTFORWARD, difficulty=CaseDifficulty.EASY
        )
        manifest = DatasetManifest(
            version="v-test",
            dataset_type=DatasetType.LLM,
            description="test",
            created_at="2026-01-01T00:00:00Z",
            case_count=2,
            cases=[case, case],
        )
        result = DatasetValidator().validate(manifest)
        assert not result.valid
        assert any("Duplicate" in e for e in result.errors)


# ─── Metric Tests ─────────────────────────────────────────────────────────────

class TestDeterministicMetrics:
    def test_latency_p95_passes(self):
        from evaluator.metrics.llm.metrics import LatencyMetric
        m = LatencyMetric()
        latencies = [100.0] * 95 + [900.0] * 5
        result = m.evaluate_p95(latencies, threshold_ms=1000)
        assert result.metric == "p95_latency"
        assert result.passed
        assert result.score <= 1000

    def test_latency_p95_fails(self):
        from evaluator.metrics.llm.metrics import LatencyMetric
        m = LatencyMetric()
        result = m.evaluate_p95([1500.0] * 100, threshold_ms=1000)
        assert not result.passed

    def test_cost_metric_passes(self):
        from evaluator.metrics.llm.metrics import CostMetric
        m = CostMetric()
        result = m.evaluate(100, 50, input_per_1k=0.01, output_per_1k=0.03, max_cost_usd=0.02)
        assert result.metric == "cost_per_query"
        assert result.score > 0
        assert isinstance(result.passed, bool)

    def test_retrieval_quality_full_recall(self):
        from evaluator.metrics.rag.metrics import RetrievalQualityMetric
        m = RetrievalQualityMetric(threshold=0.8)
        result = m.evaluate(
            retrieved_sources=["policy.pdf", "handbook.pdf"],
            required_sources=["policy.pdf"],
        )
        assert result.passed
        assert result.score == 1.0

    def test_retrieval_quality_zero_recall(self):
        from evaluator.metrics.rag.metrics import RetrievalQualityMetric
        m = RetrievalQualityMetric(threshold=0.8)
        result = m.evaluate(
            retrieved_sources=["unrelated.pdf"],
            required_sources=["policy.pdf", "handbook.pdf"],
        )
        assert not result.passed
        assert result.score == 0.0


class TestModelBasedMetrics:
    """Model-based metrics tested with MockProvider — no API calls."""

    def test_answer_relevance_with_mock(self):
        from evaluator.metrics.llm.metrics import AnswerRelevanceMetric
        from evaluator.providers.mock import MockProvider
        p = MockProvider(response_text='{"score": 0.92, "explanation": "Very relevant"}')
        m = AnswerRelevanceMetric(provider=p, threshold=0.80)
        result = m.evaluate("What is the retention policy?", "Records are kept 7 years.")
        assert result.metric == "answer_relevance"
        assert result.score == 0.92
        assert result.passed

    def test_answer_relevance_empty_answer_fails(self):
        from evaluator.metrics.llm.metrics import AnswerRelevanceMetric
        from evaluator.providers.mock import MockProvider
        m = AnswerRelevanceMetric(provider=MockProvider(), threshold=0.80)
        result = m.evaluate("question", "")
        assert not result.passed
        assert result.score == 0.0

    def test_faithfulness_with_mock(self):
        from evaluator.metrics.rag.metrics import FaithfulnessMetric
        from evaluator.providers.mock import MockProvider
        p = MockProvider(response_text='{"faithfulness_score": 0.95, "explanation": "Fully supported"}')
        m = FaithfulnessMetric(provider=p, threshold=0.90)
        result = m.evaluate(answer="7 years retention", context=["Records kept for 7 years."])
        assert result.metric == "faithfulness"
        assert result.score == 0.95
        assert result.passed

    def test_faithfulness_no_context_fails(self):
        from evaluator.metrics.rag.metrics import FaithfulnessMetric
        from evaluator.providers.mock import MockProvider
        m = FaithfulnessMetric(provider=MockProvider(), threshold=0.90)
        result = m.evaluate(answer="some answer", context=[])
        assert not result.passed

    def test_hallucination_passes_low_rate(self):
        from evaluator.metrics.llm.metrics import HallucinationMetric
        from evaluator.providers.mock import MockProvider
        p = MockProvider(response_text='{"hallucination_rate": 0.03, "explanation": "Minimal hallucination"}')
        m = HallucinationMetric(provider=p, threshold=0.10)
        result = m.evaluate("q", "answer", ["context chunk"])
        assert result.passed  # 0.03 <= 0.10

    def test_hallucination_fails_high_rate(self):
        from evaluator.metrics.llm.metrics import HallucinationMetric
        from evaluator.providers.mock import MockProvider
        p = MockProvider(response_text='{"hallucination_rate": 0.72, "explanation": "Severe hallucination"}')
        m = HallucinationMetric(provider=p, threshold=0.10)
        result = m.evaluate("q", "answer", ["context"])
        assert not result.passed

    def test_metric_result_structured_output(self):
        """Every metric must return structured output with required fields."""
        from evaluator.metrics.result import MetricResult
        r = MetricResult(
            metric="faithfulness", score=0.94, threshold=0.90,
            passed=True, explanation="Supported by context"
        )
        d = r.to_dict()
        assert set(d.keys()) >= {"metric", "score", "threshold", "passed", "explanation", "category"}
        assert d["metric"] == "faithfulness"
        assert d["score"] == 0.94


# ─── Runner Tests ─────────────────────────────────────────────────────────────

class TestEvaluationRunner:
    def _make_runner(self):
        from evaluator.providers.mock import MockProvider
        from evaluator.targets.mock import MockTarget
        from evaluator.runners.runner import EvaluationRunner
        return EvaluationRunner(
            target=MockTarget(),
            judge_provider=MockProvider(
                response_text='{"score": 0.90, "explanation": "good",'
                              '"faithfulness_score": 0.93, "hallucination_rate": 0.03,'
                              '"relevance_score": 0.88}'
            ),
            config={
                "quality_gate": {
                    "faithfulness": {"minimum": 0.90},
                    "hallucination_rate": {"maximum": 0.10},
                }
            },
        )

    def _make_small_dataset(self):
        from evaluator.datasets.loader import DatasetLoader
        manifest = DatasetLoader().load("datasets/v1/golden.json")
        # Use just 3 cases for speed
        from dataclasses import replace
        manifest = manifest.model_copy(update={"cases": manifest.cases[:3], "case_count": 3})
        return manifest

    def test_run_completes_with_mock(self):
        runner = self._make_runner()
        dataset = self._make_small_dataset()
        result = runner.run(dataset, experiment_id="exp-test")
        assert result.run_id.startswith("run-")
        assert result.total_cases == 3
        assert result.aggregate_metrics  # not empty

    def test_run_records_metadata(self):
        runner = self._make_runner()
        dataset = self._make_small_dataset()
        result = runner.run(dataset, rag_version="rag-v2", kb_version="KB-005", environment="ci")
        assert result.rag_version == "rag-v2"
        assert result.kb_version == "KB-005"
        assert result.environment == "ci"
        assert result.dataset_version == "v1"

    def test_result_serializable(self):
        runner = self._make_runner()
        dataset = self._make_small_dataset()
        result = runner.run(dataset)
        d = result.to_dict()
        # Ensure JSON-serializable
        json.dumps(d, default=str)


# ─── Regression Tests ─────────────────────────────────────────────────────────

class TestRegressionEngine:
    def _make_baseline(self, metrics):
        from evaluator.regression.baseline import BaselineSnapshot
        return BaselineSnapshot(
            baseline_id="baseline-001",
            run_id="run-001",
            experiment_id="exp-001",
            model="gpt-4o",
            prompt_version="v1",
            rag_version="v1",
            kb_version="KB-001",
            dataset_version="v1",
            git_sha="abc123",
            timestamp="2026-08-01T00:00:00Z",
            metrics=metrics,
        )

    def test_no_regression_detected(self):
        from evaluator.regression.baseline import RegressionAnalyzer
        baseline = self._make_baseline({"faithfulness": 0.94, "answer_relevance": 0.92})
        candidate = {"faithfulness": 0.93, "answer_relevance": 0.91}
        report = RegressionAnalyzer().analyze(baseline, candidate, "run-002")
        assert not report.is_regression

    def test_regression_detected_faithfulness_drop(self):
        from evaluator.regression.baseline import RegressionAnalyzer
        baseline = self._make_baseline({"faithfulness": 0.94, "hallucination_rate": 0.038})
        candidate = {"faithfulness": 0.887, "hallucination_rate": 0.072}
        report = RegressionAnalyzer().analyze(baseline, candidate, "run-003")
        assert report.is_regression
        reg_metrics = [r.metric for r in report.regressions]
        assert "faithfulness" in reg_metrics

    def test_regression_report_uses_careful_language(self):
        """Regression analysis must say 'potential' not assert causality."""
        from evaluator.regression.baseline import RegressionAnalyzer
        baseline = self._make_baseline({"faithfulness": 0.95})
        candidate = {"faithfulness": 0.80}
        report = RegressionAnalyzer().analyze(baseline, candidate, "run-004")
        for reg in report.regressions:
            assert "potential" in reg.explanation.lower() or "area" in reg.explanation.lower()


# ─── Quality Gate Tests ───────────────────────────────────────────────────────

class TestQualityGate:
    GATE_CONFIG = {
        "faithfulness": {"minimum": 0.90},
        "answer_relevance": {"minimum": 0.80},
        "hallucination_rate": {"maximum": 0.05},
        "p95_latency_ms": {"maximum_ms": 1000},
    }

    def test_gate_passes_good_metrics(self):
        from evaluator.regression.quality_gate import QualityGate
        gate = QualityGate(self.GATE_CONFIG)
        metrics = {
            "faithfulness": 0.942,
            "answer_relevance": 0.928,
            "hallucination_rate": 0.038,
            "p95_latency_ms": 820,
        }
        result = gate.evaluate("run-pass", metrics)
        assert result.passed
        assert result.verdict == "PASSED"
        assert len(result.failing_rules) == 0

    def test_gate_fails_faithfulness_below_threshold(self):
        from evaluator.regression.quality_gate import QualityGate
        gate = QualityGate(self.GATE_CONFIG)
        metrics = {
            "faithfulness": 0.887,      # below 0.90
            "answer_relevance": 0.90,
            "hallucination_rate": 0.038,
            "p95_latency_ms": 820,
        }
        result = gate.evaluate("run-fail", metrics)
        assert not result.passed
        assert result.verdict == "FAILED"
        failing_names = [r.metric for r in result.failing_rules]
        assert "faithfulness" in failing_names

    def test_gate_fails_hallucination_above_threshold(self):
        from evaluator.regression.quality_gate import QualityGate
        gate = QualityGate(self.GATE_CONFIG)
        metrics = {
            "faithfulness": 0.94,
            "answer_relevance": 0.90,
            "hallucination_rate": 0.072,   # above 0.05
            "p95_latency_ms": 820,
        }
        result = gate.evaluate("run-hall", metrics)
        assert not result.passed
        assert "hallucination_rate" in [r.metric for r in result.failing_rules]

    def test_gate_names_all_failing_rules(self):
        """Gate must name specific failing rules, not just pass/fail."""
        from evaluator.regression.quality_gate import QualityGate
        gate = QualityGate(self.GATE_CONFIG)
        metrics = {
            "faithfulness": 0.80,  # fails
            "hallucination_rate": 0.12,  # fails
            "p95_latency_ms": 1240,  # fails
        }
        result = gate.evaluate("run-multi-fail", metrics)
        assert not result.passed
        assert len(result.failing_rules) == 3

    def test_gate_format_report_contains_failing_metrics(self):
        from evaluator.regression.quality_gate import QualityGate
        gate = QualityGate(self.GATE_CONFIG)
        result = gate.evaluate("run-x", {"faithfulness": 0.80})
        report = result.format_report()
        assert "faithfulness" in report
        assert "FAILED" in report


# ─── Storage Tests ────────────────────────────────────────────────────────────

class TestLocalStorageBackend:
    def test_save_and_load(self, tmp_path):
        from evaluator.storage.backends import LocalStorageBackend
        backend = LocalStorageBackend(base_dir=tmp_path)
        data = {"run_id": "run-001", "metrics": {"faithfulness": 0.94}}
        backend.save("runs/run-001", data)
        loaded = backend.load("runs/run-001")
        assert loaded["run_id"] == "run-001"
        assert loaded["metrics"]["faithfulness"] == 0.94

    def test_exists(self, tmp_path):
        from evaluator.storage.backends import LocalStorageBackend
        backend = LocalStorageBackend(base_dir=tmp_path)
        assert not backend.exists("runs/run-999")
        backend.save("runs/run-999", {"x": 1})
        assert backend.exists("runs/run-999")

    def test_list_keys(self, tmp_path):
        from evaluator.storage.backends import LocalStorageBackend
        backend = LocalStorageBackend(base_dir=tmp_path)
        backend.save("runs/run-001", {})
        backend.save("runs/run-002", {})
        backend.save("experiments/exp-001", {})
        run_keys = backend.list_keys(prefix="runs/")
        assert len(run_keys) == 2


# ─── Model Comparison Tests ───────────────────────────────────────────────────

class TestModelComparison:
    def _fake_run(self, model, faithfulness, relevance, hallucination, p95, cost, cases=10):
        from dataclasses import dataclass
        @dataclass
        class FakeRun:
            model: str
            rag_version: str = "v1"
            aggregate_metrics: dict = None
            total_cost_usd: float = 0.0
            total_cases: int = 10
            def __post_init__(self):
                if self.aggregate_metrics is None:
                    self.aggregate_metrics = {}
        r = FakeRun(model=model, total_cost_usd=cost * cases, total_cases=cases)
        r.aggregate_metrics = {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "hallucination_rate": hallucination,
            "p95_latency_ms": p95,
        }
        return r

    def test_comparison_identifies_best_quality(self):
        from evaluator.comparison.comparison import ModelComparison
        runs = [
            self._fake_run("ModelA", 0.94, 0.92, 0.04, 820, 0.021),
            self._fake_run("ModelB", 0.89, 0.88, 0.05, 410, 0.009),
        ]
        result = ModelComparison().compare(runs)
        assert result.best_quality == "ModelA"

    def test_comparison_identifies_lowest_cost(self):
        from evaluator.comparison.comparison import ModelComparison
        runs = [
            self._fake_run("ModelA", 0.94, 0.92, 0.04, 820, 0.021),
            self._fake_run("ModelB", 0.89, 0.88, 0.05, 410, 0.009),
        ]
        result = ModelComparison().compare(runs)
        assert result.lowest_cost == "ModelB"


# ─── Demo Scenario Tests ──────────────────────────────────────────────────────

class TestDemoScenario:
    """Validates the canonical 3-run demo scenario from build-phases-and-acceptance.md."""

    GATE_CONFIG = {
        "faithfulness": {"minimum": 0.90},
        "hallucination_rate": {"maximum": 0.05},
        "p95_latency_ms": {"maximum_ms": 1000},
    }

    def test_passing_scenario(self):
        from evaluator.regression.quality_gate import QualityGate
        gate = QualityGate(self.GATE_CONFIG)
        # Baseline run (RAG V1 + Model A)
        result = gate.evaluate("run-baseline", {
            "faithfulness": 0.942,
            "answer_relevance": 0.928,
            "hallucination_rate": 0.038,
            "p95_latency_ms": 820,
        })
        assert result.passed, f"Baseline should pass: {result.failing_rules}"

    def test_failing_scenario(self):
        from evaluator.regression.quality_gate import QualityGate
        gate = QualityGate(self.GATE_CONFIG)
        # Deliberately worse RAG V2
        result = gate.evaluate("run-degraded", {
            "faithfulness": 0.887,
            "answer_relevance": 0.901,
            "hallucination_rate": 0.072,
            "p95_latency_ms": 910,
        })
        assert not result.passed, "Degraded run should FAIL the gate"
        failing = [r.metric for r in result.failing_rules]
        assert "faithfulness" in failing
        assert "hallucination_rate" in failing

    def test_restored_scenario_passes(self):
        from evaluator.regression.quality_gate import QualityGate
        gate = QualityGate(self.GATE_CONFIG)
        result = gate.evaluate("run-restored", {
            "faithfulness": 0.951,
            "answer_relevance": 0.935,
            "hallucination_rate": 0.029,
            "p95_latency_ms": 780,
        })
        assert result.passed, "Restored config should pass"
