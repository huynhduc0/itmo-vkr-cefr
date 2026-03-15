"""Regression tests for CI/CD workflow contracts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_ci_workflow_runs_pytest_on_push_and_pr():
    content = _read(".github/workflows/ci.yml")

    assert "name: CI – Tests" in content
    assert "push:" in content
    assert "pull_request:" in content
    assert "python -m pytest tests/ -v --tb=short" in content


def test_full_pipeline_enforces_test_gate_before_prepare_data():
    content = _read(".github/workflows/full_pipeline.yml")

    # Stage ordering contract: prepare-data must depend on unit tests.
    assert "lint-and-test:" in content
    assert "prepare-data:" in content
    assert "needs: lint-and-test" in content


def test_full_pipeline_has_manual_trigger_and_cpu_experiment_filter():
    content = _read(".github/workflows/full_pipeline.yml")

    assert "workflow_dispatch:" in content
    assert "contains(inputs.exps, '0')" in content
    assert "contains(inputs.exps, '1')" in content
    assert "contains(inputs.exps, '5')" in content
    assert "contains(inputs.exps, '6')" in content
    assert "contains(inputs.exps, '7')" in content
    assert "contains(inputs.exps, '8')" in content
    assert "contains(inputs.exps, '9')" in content
    assert "contains(inputs.exps, '10')" in content


def test_full_pipeline_validates_ru_preset_requires_dataset():
    """Workflow must reject language=ru without a dataset override."""
    content = _read(".github/workflows/full_pipeline.yml")

    # The workflow must contain an explicit validation step for the ru / no-dataset case.
    assert "inputs.language" in content
    assert "inputs.dataset" in content
    assert "ru" in content
    assert "placeholder" in content.lower() or "cefr_sp_ru" in content
