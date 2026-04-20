"""
Refresh the results dashboard section in README.md from committed pipeline runs.

Also maintains a ``best_results/`` directory that always reflects the single
best experiment per (task, language) pair across *all* committed pipeline runs.
This is especially useful when sentence/essay experiments run asynchronously,
because it gives a stable reference view without waiting for a single combined
run to finish.

Note on data-leakage in essay results
--------------------------------------
For some learner corpora (e.g. ``UniversalCEFR/caes_es``) the essay track may
yield suspiciously high accuracy (> 0.99) that mirrors or exceeds sentence-track
numbers.  The root cause is within-document leakage: texts from the *same*
learner/document may appear in both the training and the test split, so a
TF-IDF model can reach near-perfect accuracy by memorising learner-specific
vocabulary patterns.  The proper fix is to perform *document-level* splitting
(all texts from the same source document go into the same split), which requires
per-document metadata from the dataset.  Until such splitting is implemented,
essay-track accuracy on these corpora should be treated with scepticism.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from src.config import SUPPORTED_LANGUAGES

README_PATH = Path("README.md")
RESULTS_ROOT = Path("results")
BEST_RESULTS_ROOT = Path("best_results")
MARKER_START = "<!-- AUTO-RESULTS-START -->"
MARKER_END = "<!-- AUTO-RESULTS-END -->"


def _extract_exp_id(name: str) -> int:
    match = re.search(r"Exp\s+(\d+)", name)
    return int(match.group(1)) if match else 999


def _load_run_info(path: Path) -> Dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_runs(task: str) -> List[Tuple[Path, Dict[str, str]]]:
    """Return every committed run directory for *task*, sorted by run_number."""
    candidates: List[Tuple[int, Path, Dict[str, str]]] = []
    for path in RESULTS_ROOT.glob(f"*/{task}/run_info.json"):
        info = _load_run_info(path)
        run_number = int(info.get("run_number", "0"))
        candidates.append((run_number, path.parent, info))
    candidates.sort(key=lambda item: item[0])
    return [(run_dir, info) for _, run_dir, info in candidates]


def _latest_multilingual_run(task: str) -> Optional[Tuple[Path, Dict[str, str]]]:
    candidates: List[Tuple[int, Path, Dict[str, str]]] = []
    for path in RESULTS_ROOT.glob(f"*/{task}/run_info.json"):
        info = _load_run_info(path)
        is_multilingual = info.get("language") == "all" or bool(info.get("combined_manifest"))
        if not is_multilingual:
            continue
        run_number = int(info.get("run_number", "0"))
        candidates.append((run_number, path.parent, info))
    if not candidates:
        return None
    _, run_dir, info = max(candidates, key=lambda item: item[0])
    return run_dir, info


def _best_row(results_json: Path) -> Optional[Dict]:
    if not results_json.exists():
        return None
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    best = None
    best_qwk = float("-inf")
    for row in payload:
        if not isinstance(row, dict):
            continue
        qwk = float(row.get("qwk", 0.0) or 0.0)
        if qwk > best_qwk:
            best_qwk = qwk
            best = row
    return best


# ---------------------------------------------------------------------------
# Best-results folder
# ---------------------------------------------------------------------------

def update_best_results() -> None:
    """Scan all committed runs and write ``best_results/{task}/{lang}/best_result.json``.

    The file captures the best-ever QWK experiment for each (task, language)
    combination across *all* pipeline runs committed to the repository.  Because
    sentence and essay pipelines can run independently (asynchronously), this
    folder always provides an up-to-date global view of the best achieved
    performance regardless of which run committed last.
    """
    best: Dict[Tuple[str, str], Dict] = {}

    for task in ("sentence", "essay"):
        for run_dir, info in _all_runs(task):
            for lang in SUPPORTED_LANGUAGES:
                row = _best_row(run_dir / lang / "results.json")
                if row is None:
                    continue
                key = (task, lang)
                qwk = float(row.get("qwk", 0.0) or 0.0)
                existing = best.get(key)
                if existing is None or qwk > float(existing.get("qwk", 0.0) or 0.0):
                    best[key] = {
                        **row,
                        "run_id": info.get("run_id", ""),
                        "run_number": info.get("run_number", ""),
                        "task": task,
                        "language": lang,
                    }

    for (task, lang), record in best.items():
        dest = BEST_RESULTS_ROOT / task / lang / "best_result.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    _write_best_results_readme(best)


def _write_best_results_readme(best: Dict[Tuple[str, str], Dict]) -> None:
    """Write a human-readable summary to ``best_results/README.md``."""
    lines = [
        "# Best Results",
        "",
        "Best experiment per *(task, language)* combination, selected by **QWK**,",
        "aggregated across **all** committed pipeline runs.",
        "",
        "> This file is auto-generated by `src/update_readme_results.py`.",
        "> Do not edit by hand.",
        "",
    ]

    for task in ("sentence", "essay"):
        task_rows = {lang: best[(task, lang)] for lang in SUPPORTED_LANGUAGES if (task, lang) in best}
        if not task_rows:
            continue
        lines.extend([
            f"## {task.capitalize()} track",
            "",
            "| Language | Best Experiment | QWK | Accuracy | Run |",
            "|----------|-----------------|-----|----------|-----|",
        ])
        for lang in SUPPORTED_LANGUAGES:
            record = task_rows.get(lang)
            if record is None:
                lines.append(f"| `{lang}` | — | — | — | — |")
                continue
            exp_id = _extract_exp_id(str(record.get("name", "")))
            exp_label = f"Exp {exp_id}" if exp_id != 999 else str(record.get("name", ""))
            qwk = f"{float(record.get('qwk', 0.0) or 0.0):.4f}"
            accuracy = f"{float(record.get('accuracy', 0.0) or 0.0):.4f}"
            run_ref = record.get("run_number", "?")
            lines.append(f"| `{lang}` | `{exp_label}` | `{qwk}` | `{accuracy}` | #{run_ref} |")
        lines.append("")

    dest = BEST_RESULTS_ROOT / "README.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _task_visuals(task: str) -> List[str]:
    """Return markdown lines that *embed* SVG charts for *task* directly."""
    visuals_dir = Path("visuals") / "generated" / task
    heatmap = visuals_dir / "qwk_heatmap.svg"
    best_bar = visuals_dir / "best_qwk_by_language.svg"
    lines: List[str] = []
    if heatmap.exists():
        lines.append(f"![QWK Heatmap – {task}]({heatmap.as_posix()})")
    if best_bar.exists():
        lines.append(f"![Best QWK by Language – {task}]({best_bar.as_posix()})")
    return lines


# ---------------------------------------------------------------------------
# Per-task result tables
# ---------------------------------------------------------------------------

def _task_table(task: str) -> str:
    latest = _latest_multilingual_run(task)
    title = f"### {task.capitalize()}"
    if latest is None:
        return "\n".join([title, "", "Chưa có multilingual run nào được commit cho task này."])

    run_dir, info = latest
    lines = [
        title,
        "",
        f"Run mới nhất: [`{run_dir.as_posix()}`]({run_dir.as_posix()})",
        "",
        "| Language | Best experiment by QWK | QWK | Accuracy | Note |",
        "|----------|-------------------------|-----|----------|------|",
    ]

    for lang in SUPPORTED_LANGUAGES:
        best = _best_row(run_dir / lang / "results.json")
        if best is None:
            lines.append(f"| `{lang}` | `N/A` | `N/A` | `N/A` | no committed results for this language |")
            continue
        exp_id = _extract_exp_id(str(best.get("name", "")))
        exp_label = f"Exp {exp_id}" if exp_id != 999 else str(best.get("name", ""))
        qwk = f"{float(best.get('qwk', 0.0) or 0.0):.4f}"
        accuracy = f"{float(best.get('accuracy', 0.0) or 0.0):.4f}"
        note = str(best.get("note", "") or "").replace("|", "/")
        lines.append(f"| `{lang}` | `{exp_label}` | `{qwk}` | `{accuracy}` | {note or '-'} |")

    lines.extend(
        [
            "",
            f"Inputs: `task={info.get('task', task)}`, `exps={info.get('exps', '')}`, `language={info.get('language', '')}`",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# README section builder
# ---------------------------------------------------------------------------

def build_results_section() -> str:
    parts = [
        "## Results Dashboard",
        "",
        "Visualization được cập nhật tự động từ multilingual runs mới nhất của pipeline.",
        "",
    ]

    # Embed SVG charts for each task
    sentence_visuals = _task_visuals("sentence")
    essay_visuals = _task_visuals("essay")
    if sentence_visuals or essay_visuals:
        parts.append("### Sentence track – visuals")
        parts.extend(sentence_visuals if sentence_visuals else ["*Chưa có visualization.*"])
        parts.append("")
        parts.append("### Essay track – visuals")
        parts.extend(essay_visuals if essay_visuals else ["*Chưa có visualization.*"])
        parts.append("")

    # Badges
    sentence_badges = Path("visuals/generated/sentence/badges")
    if sentence_badges.exists():
        parts.extend(
            [
                "### Badges",
                "",
                f"![Sentence Best QWK]({(sentence_badges / 'best-qwk.svg').as_posix()})",
                f"![Sentence Best Macro-F1]({(sentence_badges / 'best-macro-f1.svg').as_posix()})",
                f"![Sentence Languages]({(sentence_badges / 'languages.svg').as_posix()})",
                "",
            ]
        )

    # Per-task result tables (latest multilingual run)
    for task in ("sentence", "essay"):
        parts.extend(["", _task_table(task)])

    # Link to the best-results folder
    parts.extend(
        [
            "",
            "---",
            "",
            "### Best results across all runs",
            "",
            "Kết quả tốt nhất trên mọi run đã commit, xem chi tiết tại [`best_results/`](best_results/README.md).",
        ]
    )

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def update_readme() -> None:
    content = README_PATH.read_text(encoding="utf-8")
    section = f"{MARKER_START}\n{build_results_section()}\n{MARKER_END}"
    pattern = re.compile(f"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}", re.S)
    if pattern.search(content):
        updated = pattern.sub(section, content)
    else:
        updated = content.rstrip() + "\n\n" + section + "\n"
    README_PATH.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    update_best_results()
    update_readme()
