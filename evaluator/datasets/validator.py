"""Dataset validator — validates golden datasets for completeness and correctness."""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import DatasetManifest, EvaluationCase, RAGEvaluationCase


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    case_count: int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class DatasetValidator:
    """Validates a DatasetManifest for structural and content correctness."""

    MIN_CASES = 1
    RECOMMENDED_MIN_CASES = 10

    def validate(self, manifest: DatasetManifest) -> ValidationResult:
        result = ValidationResult(valid=True, case_count=len(manifest.cases))

        # Basic structural checks
        if not manifest.version.strip():
            result.add_error("Dataset version must not be empty")
        if not manifest.cases:
            result.add_error("Dataset must contain at least one case")
            return result

        # Case count integrity
        if manifest.case_count != len(manifest.cases):
            result.add_error(
                f"case_count={manifest.case_count} does not match "
                f"actual cases={len(manifest.cases)}"
            )

        # Warnings for small datasets
        if len(manifest.cases) < self.RECOMMENDED_MIN_CASES:
            result.add_warning(
                f"Dataset has only {len(manifest.cases)} cases; "
                f"recommend at least {self.RECOMMENDED_MIN_CASES} for meaningful evaluation"
            )

        # Check for duplicate IDs
        ids = [c.id for c in manifest.cases]
        seen: set[str] = set()
        duplicates: list[str] = []
        for cid in ids:
            if cid in seen:
                duplicates.append(cid)
            seen.add(cid)
        if duplicates:
            result.add_error(f"Duplicate case IDs found: {duplicates}")

        # Per-case checks
        for case in manifest.cases:
            self._validate_case(case, result)

        # Category coverage warnings
        categories = {c.category for c in manifest.cases}
        if len(categories) < 3:
            result.add_warning(
                f"Dataset covers only {len(categories)} categories; "
                "aim for broader coverage"
            )

        return result

    def _validate_case(
        self, case: EvaluationCase | RAGEvaluationCase, result: ValidationResult
    ) -> None:
        prefix = f"Case {case.id}"

        if not case.question.strip():
            result.add_error(f"{prefix}: question must not be empty")
        if not case.expected_answer.strip():
            result.add_error(f"{prefix}: expected_answer must not be empty")

        if isinstance(case, RAGEvaluationCase):
            for src in case.required_sources:
                if not src.strip():
                    result.add_error(f"{prefix}: required_sources contains empty string")

            if case.category.value == "unanswerable" and case.required_sources:
                result.add_warning(
                    f"{prefix}: unanswerable case has required_sources — "
                    "consider leaving it empty"
                )
