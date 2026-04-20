import json
from pathlib import Path

from src.publish_results import collect_result_artifacts, generate_visuals


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_result_artifacts_preserves_language_subdirs(tmp_path):
    download_root = tmp_path / "results-raw"
    output_dir = tmp_path / "results" / "123" / "sentence"

    _write_json(
        download_root / "artifact-a" / "results" / "ru" / "results.json",
        [{"name": "Exp 1 – TF-IDF+LR", "qwk": 0.51, "macro_f1": 0.5, "accuracy": 0.55}],
    )
    _write_json(
        download_root / "artifact-b" / "results" / "fr" / "results.json",
        [{"name": "Exp 7 – TF-IDF+LinearSVC", "qwk": 0.61, "macro_f1": 0.6, "accuracy": 0.65}],
    )
    log_path = download_root / "artifact-c" / "results_cpu_ru.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("ok", encoding="utf-8")

    copied = collect_result_artifacts(download_root, output_dir, default_language="all")

    assert output_dir.joinpath("ru", "results.json").exists()
    assert output_dir.joinpath("fr", "results.json").exists()
    assert output_dir.joinpath("logs", "results_cpu_ru.txt").exists()
    assert len(copied) == 2


def test_collect_result_artifacts_places_single_language_results_under_default_language(tmp_path):
    download_root = tmp_path / "results-raw"
    output_dir = tmp_path / "results" / "124" / "essay"

    _write_json(
        download_root / "artifact-a" / "results" / "results.json",
        [{"name": "Exp 9 – Word TF-IDF+LR", "qwk": 0.72, "macro_f1": 0.7, "accuracy": 0.71}],
    )

    collect_result_artifacts(download_root, output_dir, default_language="es")

    assert output_dir.joinpath("es", "results.json").exists()


def test_generate_visuals_emits_svg_files(tmp_path):
    results_dir = tmp_path / "results" / "555" / "sentence"
    visuals_dir = tmp_path / "visuals" / "generated" / "sentence"

    _write_json(
        results_dir / "en" / "results.json",
        [
            {"name": "Exp 1 – TF-IDF+LR", "qwk": 0.61, "macro_f1": 0.6, "accuracy": 0.62},
            {"name": "Exp 8 – Zero-shot XLM-R", "qwk": 0.67, "macro_f1": 0.66, "accuracy": 0.65},
        ],
    )
    _write_json(
        results_dir / "ru" / "results.json",
        [{"name": "Exp 1 – TF-IDF+LR", "qwk": 0.44, "macro_f1": 0.43, "accuracy": 0.45}],
    )

    generate_visuals(
        results_paths=[results_dir / "en" / "results.json", results_dir / "ru" / "results.json"],
        default_language="en",
        visuals_dir=visuals_dir,
    )

    assert visuals_dir.joinpath("qwk_heatmap.svg").exists()
    assert visuals_dir.joinpath("best_qwk_by_language.svg").exists()
    assert visuals_dir.joinpath("badges", "best-qwk.svg").exists()
    assert visuals_dir.joinpath("badges", "languages.svg").exists()
