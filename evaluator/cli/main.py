"""
LLM Eval CI/CD — CLI
Professional CLI with CI-friendly exit codes.

Commands:
  llm-eval run      -- run evaluation
  llm-eval compare  -- compare two runs
  llm-eval gate     -- check quality gate (exits 1 on failure)
  llm-eval report   -- print run report
  llm-eval dataset  -- dataset utilities (validate)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(2)
    with open(path) as f:
        return yaml.safe_load(f)


@click.group()
@click.version_option(version="1.0.0", prog_name="llm-eval")
def cli():
    """LLM Eval CI/CD — Automated LLM + RAG Evaluation and Quality Gates."""
    pass


# ─── run ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", required=True, help="Path to evaluation YAML config")
@click.option("--experiment-id", default=None, help="Experiment ID to group runs under")
@click.option("--environment", default="local", show_default=True, help="Environment tag")
@click.option("--output", "-o", default=None, help="Write run result JSON to this file")
@click.option("--set-baseline", is_flag=True, default=False, help="Promote this run to baseline after completion")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress progress output")
def run(config, experiment_id, environment, output, set_baseline, quiet):
    """Run evaluation against a golden dataset.

    Exit codes: 0=success, 1=evaluation error, 2=config error
    """
    cfg = load_config(config)

    if not quiet:
        console.rule("[bold blue]LLM Eval CI/CD — Evaluation Run[/bold blue]")
        console.print(f"Config: [cyan]{config}[/cyan]")

    try:
        from evaluator.providers.factory import build_provider
        from evaluator.targets.mock import MockTarget
        from evaluator.datasets.loader import DatasetLoader
        from evaluator.datasets.validator import DatasetValidator
        from evaluator.runners.runner import EvaluationRunner
        from evaluator.storage.backends import build_storage_backend
        from evaluator.regression.baseline import BaselineManager

        # Build provider
        provider_cfg = cfg.get("model", {})
        provider = build_provider(provider_cfg)

        # Build target
        target_cfg = cfg.get("target", {})
        target_type = target_cfg.get("type", "mock")
        if target_type == "mock":
            target = MockTarget()
        elif target_type == "local_rag":
            from evaluator.targets.local_rag import LocalRAGTarget
            target = LocalRAGTarget()
        elif target_type in ("http_rag", "http_model"):
            from evaluator.targets.http_target import HTTPRAGTarget
            target = HTTPRAGTarget(base_url=target_cfg["url"])
        else:
            target = MockTarget()

        # Load dataset
        dataset_cfg = cfg.get("dataset", {})
        dataset_path = dataset_cfg.get("path", "datasets/v1/golden.json")
        loader = DatasetLoader()
        dataset = loader.load(dataset_path)

        # Validate dataset
        validator = DatasetValidator()
        val_result = validator.validate(dataset)
        if not val_result.valid:
            console.print("[red]Dataset validation failed:[/red]")
            for err in val_result.errors:
                console.print(f"  [red]✗ {err}[/red]")
            sys.exit(2)

        if not quiet:
            console.print(f"Dataset: [cyan]{dataset.version}[/cyan] — {dataset.case_count} cases")
            console.print(f"Target:  [cyan]{target.target_id}[/cyan]")
            console.print(f"Model:   [cyan]{provider.model_name}[/cyan]")
            console.print("")

        # Run evaluation
        runner = EvaluationRunner(
            target=target,
            judge_provider=provider,
            config=cfg,
        )

        with console.status("[bold green]Running evaluation...[/bold green]"):
            result = runner.run(
                dataset=dataset,
                experiment_id=experiment_id,
                rag_version=cfg.get("rag", {}).get("version", "v1"),
                kb_version=cfg.get("kb", {}).get("version", "KB-001"),
                environment=environment,
            )

        # Print summary
        _print_run_summary(result, quiet)

        # Save result
        storage = build_storage_backend()
        storage.save(f"runs/{result.run_id}", result.to_dict())

        if output:
            Path(output).write_text(json.dumps(result.to_dict(), indent=2, default=str))
            if not quiet:
                console.print(f"\nResult saved to: [cyan]{output}[/cyan]")

        if set_baseline:
            bm = BaselineManager(storage)
            snap = bm.set_as_baseline(result, notes="Promoted via CLI --set-baseline")
            console.print(f"[green]Baseline set: {snap.baseline_id}[/green]")

        if not quiet:
            console.print(f"\n[bold]Run ID:[/bold] [cyan]{result.run_id}[/cyan]")

    except Exception as e:
        console.print(f"[red]Evaluation error: {e}[/red]")
        sys.exit(1)


def _print_run_summary(result, quiet: bool):
    if quiet:
        return
    m = result.aggregate_metrics
    table = Table(title=f"Run Results — {result.run_id}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold")
    table.add_column("Status")

    for metric, score in sorted(m.items()):
        if "latency" in metric:
            table.add_row(metric, f"{score:.0f}ms", "")
        elif "cost" in metric:
            table.add_row(metric, f"${score:.4f}", "")
        elif "token" in metric:
            table.add_row(metric, f"{int(score):,}", "")
        else:
            icon = "✓" if score >= 0.85 else "✗"
            color = "green" if score >= 0.85 else "red"
            table.add_row(metric, f"{score:.1%}", f"[{color}]{icon}[/{color}]")

    console.print(table)
    console.print(f"\nTotal cases: {result.total_cases} | "
                  f"Passed: [green]{result.passed_cases}[/green] | "
                  f"Errors: [red]{result.error_count}[/red] | "
                  f"Total cost: [yellow]${result.total_cost_usd:.4f}[/yellow]")


# ─── compare ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--baseline", "-b", required=True, help="Baseline run ID or baseline ID")
@click.option("--candidate", "-c", required=True, help="Candidate run ID to compare against baseline")
def compare(baseline, candidate):
    """Compare a candidate run against a baseline.

    Exit codes: 0=no regression, 1=regression detected, 2=error
    """
    try:
        from evaluator.storage.backends import build_storage_backend
        from evaluator.regression.baseline import BaselineManager, BaselineSnapshot, RegressionAnalyzer

        storage = build_storage_backend()

        # Load baseline
        bm = BaselineManager(storage)
        try:
            baseline_snap = bm.load_baseline(baseline)
        except FileNotFoundError:
            # Try loading as a run result and converting on-the-fly
            run_data = storage.load(f"runs/{baseline}")
            baseline_snap = BaselineSnapshot(
                baseline_id=baseline,
                run_id=run_data["run_id"],
                experiment_id=run_data.get("experiment_id", ""),
                model=run_data.get("model", ""),
                prompt_version=run_data.get("prompt_version", "v1"),
                rag_version=run_data.get("rag_version", "v1"),
                kb_version=run_data.get("kb_version", "KB-001"),
                dataset_version=run_data.get("dataset_version", "v1"),
                git_sha=run_data.get("git_sha", "unknown"),
                timestamp=run_data.get("timestamp", ""),
                metrics=run_data.get("aggregate_metrics", {}),
            )

        # Load candidate
        candidate_data = storage.load(f"runs/{candidate}")
        candidate_metrics = candidate_data.get("aggregate_metrics", {})

        analyzer = RegressionAnalyzer()
        report = analyzer.analyze(baseline_snap, candidate_metrics, candidate)

        console.rule("[bold]Regression Analysis[/bold]")
        console.print(f"Baseline:  [cyan]{baseline_snap.baseline_id}[/cyan]")
        console.print(f"Candidate: [cyan]{candidate}[/cyan]")
        console.print(f"Overall delta: [{'red' if report.is_regression else 'green'}]{report.overall_delta_pct:+.1f}%[/]")
        console.print("")

        if report.regressions:
            console.print("[bold red]⚠ Potential Regression Areas:[/bold red]")
            for r in report.regressions:
                console.print(f"  [red]✗[/red] {r.explanation}")
        if report.improvements:
            console.print("[bold green]✓ Improvements:[/bold green]")
            for i in report.improvements:
                console.print(f"  [green]↑[/green] {i.explanation}")

        console.print(f"\n[italic]{report.summary}[/italic]")

        if report.is_regression:
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Compare error: {e}[/red]")
        sys.exit(2)


# ─── gate ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--run", "run_id", required=True, help="Run ID to evaluate")
@click.option("--config", "-c", default="configs/evaluation.yaml", help="Config with quality_gate section")
@click.option("--fail-fast", is_flag=True, default=False, help="Print summary and exit immediately on first failure")
def gate(run_id, config, fail_fast):
    """Evaluate quality gate for a run. Exits 1 on failure (blocks CI).

    Exit codes: 0=PASSED, 1=FAILED, 2=error
    """
    try:
        cfg = load_config(config)
        gate_cfg = cfg.get("quality_gate", {})

        if not gate_cfg:
            console.print("[yellow]No quality_gate section found in config — skipping gate[/yellow]")
            sys.exit(0)

        from evaluator.storage.backends import build_storage_backend
        from evaluator.regression.quality_gate import QualityGate

        storage = build_storage_backend()
        run_data = storage.load(f"runs/{run_id}")
        metrics = run_data.get("aggregate_metrics", {})

        qg = QualityGate(gate_cfg)
        result = qg.evaluate(run_id, metrics)

        console.print(result.format_report())

        if not result.passed:
            console.print("\n[bold red]DEPLOYMENT BLOCKED — Quality gate failed[/bold red]")
            sys.exit(1)
        else:
            console.print("\n[bold green]DEPLOYMENT APPROVED — Quality gate passed[/bold green]")
            sys.exit(0)

    except FileNotFoundError:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(2)
    except Exception as e:
        console.print(f"[red]Gate error: {e}[/red]")
        sys.exit(2)


# ─── report ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--run", "run_id", required=True, help="Run ID")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json"]), show_default=True)
def report(run_id, fmt):
    """Print a detailed report for an evaluation run.

    Exit codes: 0=success, 2=error
    """
    try:
        from evaluator.storage.backends import build_storage_backend
        storage = build_storage_backend()
        data = storage.load(f"runs/{run_id}")

        if fmt == "json":
            click.echo(json.dumps(data, indent=2, default=str))
            return

        console.rule(f"[bold]Run Report — {run_id}[/bold]")
        meta = Table(show_header=False, box=None)
        meta.add_column("Key", style="dim")
        meta.add_column("Value", style="bold")
        for key in ["model", "prompt_version", "rag_version", "kb_version",
                    "dataset_version", "git_sha", "timestamp", "environment"]:
            meta.add_row(key, str(data.get(key, "N/A")))
        console.print(meta)
        console.print("")

        metrics = data.get("aggregate_metrics", {})
        if metrics:
            t = Table(title="Aggregate Metrics")
            t.add_column("Metric")
            t.add_column("Score", style="bold")
            for k, v in sorted(metrics.items()):
                t.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            console.print(t)

        console.print(f"\nTotal cases: {data.get('total_cases')} | "
                      f"Passed: {data.get('passed_cases')} | "
                      f"Errors: {data.get('error_count')}")
        gate_r = data.get("quality_gate_result")
        if gate_r:
            color = "green" if gate_r == "PASSED" else "red"
            console.print(f"Quality gate: [{color}]{gate_r}[/{color}]")

    except FileNotFoundError:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(2)
    except Exception as e:
        console.print(f"[red]Report error: {e}[/red]")
        sys.exit(2)


# ─── dataset ──────────────────────────────────────────────────────────────────

@cli.group()
def dataset():
    """Dataset management commands."""
    pass


@dataset.command("validate")
@click.argument("dataset_path")
def dataset_validate(dataset_path):
    """Validate a golden dataset file or directory.

    Exit codes: 0=valid, 1=invalid, 2=error
    """
    from pathlib import Path
    p = Path(dataset_path)
    # Accept either a directory (will look for golden.json) or a file
    if p.is_dir():
        p = p / "golden.json"

    try:
        from evaluator.datasets.loader import DatasetLoader
        from evaluator.datasets.validator import DatasetValidator

        loader = DatasetLoader()
        dataset_obj = loader.load(p)
        validator = DatasetValidator()
        result = validator.validate(dataset_obj)

        console.rule(f"[bold]Dataset Validation — {dataset_obj.version}[/bold]")
        console.print(f"Cases: {dataset_obj.case_count} | Type: {dataset_obj.dataset_type}")
        console.print(f"Categories: {', '.join(sorted(dataset_obj.categories))}")
        console.print("")

        if result.valid:
            console.print("[bold green]✓ Dataset is valid[/bold green]")
        else:
            console.print("[bold red]✗ Dataset validation failed[/bold red]")
            for err in result.errors:
                console.print(f"  [red]ERROR: {err}[/red]")

        for warn in result.warnings:
            console.print(f"  [yellow]WARN: {warn}[/yellow]")

        sys.exit(0 if result.valid else 1)

    except FileNotFoundError:
        console.print(f"[red]Dataset file not found: {p}[/red]")
        sys.exit(2)
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        sys.exit(2)


if __name__ == "__main__":
    cli()
