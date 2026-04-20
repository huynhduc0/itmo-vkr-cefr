"""
Refresh the results dashboard section in README.md from committed pipeline runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from src.config import SUPPORTED_LANGUAGES

README_PATH = Path("README.md")
RESULTS_ROOT = Path("results")
MARKER_START = "<!-- AUTO-RESULTS-START -->"
MARKER_END = "<!-- AUTO-RESULTS-END -->"


def _extract_exp_id(name: str) -> int:
    match = re.search(r"Exp\s+(\d+)", name)
    return int(match.group(1)) if match else 999


def _load_run_info(path: Path) -> Dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _best_row(results_json: Path) -> Optional[Dict[str, str]]:
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


def _task_visuals(task: str) -> List[str]:
    visuals_dir = Path("visuals") / "generated" / task
    lines = [f"- `{task}`:"]
    heatmap = visuals_dir / "qwk_heatmap.svg"
    best_bar = visuals_dir / "best_qwk_by_language.svg"
    if heatmap.exists():
        lines.append(f"  - [QWK Heatmap]({heatmap.as_posix()})")
    if best_bar.exists():
        lines.append(f"  - [Best QWK by Language]({best_bar.as_posix()})")
    if not heatmap.exists() and not best_bar.exists():
        lines.append("  - chưa có visualization được commit")
    return lines


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


def build_results_section() -> str:
    parts = [
        "## Results Dashboard",
        "",
        "Visualization được cập nhật tự động từ multilingual runs mới nhất của pipeline.",
        "",
        "### Visuals",
    ]
    for task in ("sentence", "essay"):
        parts.extend(_task_visuals(task))
    sentence_badges = Path("visuals/generated/sentence/badges")
    if sentence_badges.exists():
        parts.extend(
            [
                "",
                "### Badges",
                "",
                f"![Sentence Best QWK]({(sentence_badges / 'best-qwk.svg').as_posix()})",
                f"![Sentence Best Macro-F1]({(sentence_badges / 'best-macro-f1.svg').as_posix()})",
                f"![Sentence Languages]({(sentence_badges / 'languages.svg').as_posix()})",
            ]
        )
    for task in ("sentence", "essay"):
        parts.extend(["", _task_table(task)])
    return "\n".join(parts).strip()


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
    update_readme()
