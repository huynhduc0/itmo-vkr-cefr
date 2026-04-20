import json
from pathlib import Path

import src.update_readme_results as urr


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_results_section_uses_latest_multilingual_run(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    results = tmp_path / "results"
    visuals = tmp_path / "visuals" / "generated" / "sentence"
    visuals.mkdir(parents=True, exist_ok=True)
    (visuals / "qwk_heatmap.svg").write_text("<svg/>", encoding="utf-8")
    (visuals / "best_qwk_by_language.svg").write_text("<svg/>", encoding="utf-8")
    badges = visuals / "badges"
    badges.mkdir()
    for name in ("best-qwk.svg", "best-macro-f1.svg", "languages.svg"):
        (badges / name).write_text("<svg/>", encoding="utf-8")

    monkeypatch.setattr(urr, "README_PATH", readme)
    monkeypatch.setattr(urr, "RESULTS_ROOT", results)

    _write_json(
        results / "100" / "sentence" / "run_info.json",
        {"run_number": "10", "task": "sentence", "language": "all", "exps": "0 1 7"},
    )
    _write_json(
        results / "100" / "sentence" / "en" / "results.json",
        [{"name": "Exp 7 – TF-IDF+LinearSVC", "qwk": 0.7, "accuracy": 0.6}],
    )
    _write_json(
        results / "101" / "sentence" / "run_info.json",
        {"run_number": "11", "task": "sentence", "language": "all", "exps": "0 1 10"},
    )
    _write_json(
        results / "101" / "sentence" / "en" / "results.json",
        [{"name": "Exp 10 – Ensemble (LR+CNB)", "qwk": 0.8, "accuracy": 0.7}],
    )

    section = urr.build_results_section()

    assert "results/101/sentence" in section
    assert "`Exp 10`" in section
    assert "`ru` | `N/A`" in section


def test_update_readme_replaces_marker_block(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n<!-- AUTO-RESULTS-START -->\nold\n<!-- AUTO-RESULTS-END -->\nafter\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(urr, "README_PATH", readme)
    monkeypatch.setattr(urr, "build_results_section", lambda: "NEW SECTION")

    urr.update_readme()

    content = readme.read_text(encoding="utf-8")
    assert "NEW SECTION" in content
    assert "old" not in content
