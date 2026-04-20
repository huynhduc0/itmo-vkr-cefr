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
    assert "contains(format(' {0} ', inputs.exps), ' 0 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 1 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 5 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 6 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 7 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 8 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 9 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 10 ')" in content


def test_full_pipeline_has_gpu_filters_for_exp11_to_exp14():
    content = _read(".github/workflows/full_pipeline.yml")

    assert "contains(format(' {0} ', inputs.exps), ' 11 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 12 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 13 ')" in content
    assert "contains(format(' {0} ', inputs.exps), ' 14 ')" in content


def test_full_pipeline_validates_non_english_presets_require_dataset():
    """Workflow must validate the all/dataset and all/exp6 combinations."""
    content = _read(".github/workflows/full_pipeline.yml")

    # The workflow must contain explicit validation for language=all edge cases.
    assert "inputs.language" in content
    assert "inputs.dataset" in content
    assert "\"${{ inputs.language }}\" == \"all\"" in content
    assert "Exp 6 is not supported with language=all" in content
    assert "dataset must be left empty" in content
    assert "- all" in content
    assert "it" in content
    assert "es" in content
    assert "de" in content
    assert "fr" in content


def test_full_pipeline_loops_over_all_languages_when_requested():
    content = _read(".github/workflows/full_pipeline.yml")

    assert "python -m src.prepare_data --language all --output data/" in content
    assert "for lang in en ru it es de fr; do" in content
    assert '--data_dir    "data/$lang"' in content
