"""
Publish pipeline results into a stable repository structure and generate SVG
visualizations/badges that can be linked from the README.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

from src.config import SUPPORTED_LANGUAGES

COPY_EXTENSIONS = {".json", ".csv", ".txt"}


@dataclass
class MetricRecord:
    language: str
    experiment_id: int
    experiment_label: str
    qwk: float
    macro_f1: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish pipeline results and visuals.")
    parser.add_argument("--download_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--default_language", default="en")
    parser.add_argument("--visuals_dir", required=True)
    return parser.parse_args()


def _extract_experiment_id(name: str) -> int:
    match = re.search(r"Exp\s+(\d+)", name)
    return int(match.group(1)) if match else 999


def _short_experiment_label(name: str) -> str:
    exp_id = _extract_experiment_id(name)
    return f"Exp {exp_id}" if exp_id != 999 else name


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def collect_result_artifacts(download_root: Path, output_dir: Path, default_language: str) -> List[Path]:
    copied_results: List[Path] = []
    if not download_root.exists():
        return copied_results

    normalized_default_language = default_language if default_language in SUPPORTED_LANGUAGES else "global"

    for src in sorted(p for p in download_root.rglob("*") if p.is_file() and p.suffix in COPY_EXTENSIONS):
        parts = src.relative_to(download_root).parts
        try:
            results_idx = parts.index("results")
        except ValueError:
            dest = output_dir / "logs" / src.name
        else:
            subparts = parts[results_idx + 1 :]
            if not subparts:
                continue
            if len(subparts) == 1:
                dest = output_dir / normalized_default_language / subparts[0]
            elif subparts[0] not in SUPPORTED_LANGUAGES:
                dest = output_dir / "extras" / Path(*subparts)
            else:
                dest = output_dir.joinpath(*subparts)

        _copy_file(src, dest)
        if dest.name == "results.json":
            copied_results.append(dest)

    return copied_results


def _load_metric_records(results_paths: Sequence[Path], default_language: str) -> List[MetricRecord]:
    records: List[MetricRecord] = []

    for path in results_paths:
        language = path.parent.name if path.parent.name in SUPPORTED_LANGUAGES else default_language
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            exp_id = _extract_experiment_id(name)
            if exp_id == 999:
                continue
            records.append(
                MetricRecord(
                    language=language,
                    experiment_id=exp_id,
                    experiment_label=_short_experiment_label(name),
                    qwk=float(item.get("qwk", 0.0) or 0.0),
                    macro_f1=float(item.get("macro_f1", 0.0) or 0.0),
                    accuracy=float(item.get("accuracy", 0.0) or 0.0),
                )
            )
    return records


def _color_for_score(score: float) -> str:
    score = min(1.0, max(0.0, score))
    red = int(239 - 135 * score)
    green = int(68 + 125 * score)
    blue = int(68 + 70 * score)
    return f"rgb({red},{green},{blue})"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _badge_svg(label: str, value: str, color: str) -> str:
    left_width = max(48, 7 * len(label) + 18)
    right_width = max(44, 7 * len(value) + 18)
    total_width = left_width + right_width
    left_center = left_width / 2
    right_center = left_width + right_width / 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{escape(label)}: {escape(value)}">
<rect width="{left_width}" height="20" fill="#555"/>
<rect x="{left_width}" width="{right_width}" height="20" fill="{color}"/>
<text x="{left_center}" y="14" fill="#fff" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11" text-anchor="middle">{escape(label)}</text>
<text x="{right_center}" y="14" fill="#fff" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11" text-anchor="middle">{escape(value)}</text>
</svg>
"""


def _render_heatmap_svg(records: Sequence[MetricRecord]) -> str:
    languages = sorted({r.language for r in records})
    experiments = sorted({(r.experiment_id, r.experiment_label) for r in records})
    metric_map: Dict[Tuple[str, int], float] = {(r.language, r.experiment_id): r.qwk for r in records}

    cell_w = 88
    cell_h = 34
    left_margin = 72
    top_margin = 64
    width = left_margin + cell_w * len(experiments) + 24
    height = top_margin + cell_h * len(languages) + 36

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-label="QWK heatmap">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="16" y="28" font-family="Arial,sans-serif" font-size="18" font-weight="bold" fill="#111827">QWK by Language and Experiment</text>',
    ]

    for col, (_, label) in enumerate(experiments):
        x = left_margin + col * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="52" font-family="Arial,sans-serif" font-size="11" text-anchor="middle" fill="#374151">{escape(label)}</text>'
        )

    for row, lang in enumerate(languages):
        y = top_margin + row * cell_h
        parts.append(
            f'<text x="{left_margin - 10}" y="{y + 22}" font-family="Arial,sans-serif" font-size="12" text-anchor="end" fill="#111827">{escape(lang.upper())}</text>'
        )
        for col, (exp_id, _) in enumerate(experiments):
            x = left_margin + col * cell_w
            score = metric_map.get((lang, exp_id))
            fill = "#e5e7eb" if score is None else _color_for_score(score)
            value = "—" if score is None else f"{score:.3f}"
            text_fill = "#111827" if score is None or score < 0.6 else "#ffffff"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="6" fill="{fill}"/>')
            parts.append(
                f'<text x="{x + (cell_w - 4) / 2}" y="{y + 22}" font-family="Arial,sans-serif" font-size="11" text-anchor="middle" fill="{text_fill}">{escape(value)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _render_bar_svg(records: Sequence[MetricRecord]) -> str:
    best_by_language: Dict[str, MetricRecord] = {}
    for record in records:
        current = best_by_language.get(record.language)
        if current is None or record.qwk > current.qwk:
            best_by_language[record.language] = record

    ordered = sorted(best_by_language.values(), key=lambda item: item.language)
    bar_w = 72
    chart_h = 220
    left_margin = 56
    bottom_margin = 48
    top_margin = 48
    width = left_margin + bar_w * len(ordered) + 32
    height = top_margin + chart_h + bottom_margin

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-label="Best QWK by language">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="16" y="28" font-family="Arial,sans-serif" font-size="18" font-weight="bold" fill="#111827">Best QWK by Language</text>',
        f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + chart_h}" stroke="#9ca3af"/>',
        f'<line x1="{left_margin}" y1="{top_margin + chart_h}" x2="{width - 16}" y2="{top_margin + chart_h}" stroke="#9ca3af"/>',
    ]

    for tick in range(0, 6):
        value = tick / 5
        y = top_margin + chart_h - value * chart_h
        parts.append(f'<line x1="{left_margin}" y1="{y}" x2="{width - 16}" y2="{y}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left_margin - 8}" y="{y + 4}" font-family="Arial,sans-serif" font-size="10" text-anchor="end" fill="#6b7280">{value:.1f}</text>'
        )

    for idx, record in enumerate(ordered):
        x = left_margin + idx * bar_w + 14
        bar_height = max(2, record.qwk * chart_h)
        y = top_margin + chart_h - bar_height
        fill = _color_for_score(record.qwk)
        parts.append(f'<rect x="{x}" y="{y}" width="44" height="{bar_height}" rx="6" fill="{fill}"/>')
        parts.append(
            f'<text x="{x + 22}" y="{y - 6}" font-family="Arial,sans-serif" font-size="11" text-anchor="middle" fill="#111827">{record.qwk:.3f}</text>'
        )
        parts.append(
            f'<text x="{x + 22}" y="{top_margin + chart_h + 18}" font-family="Arial,sans-serif" font-size="11" text-anchor="middle" fill="#111827">{escape(record.language.upper())}</text>'
        )
        parts.append(
            f'<text x="{x + 22}" y="{top_margin + chart_h + 32}" font-family="Arial,sans-serif" font-size="10" text-anchor="middle" fill="#6b7280">{escape(record.experiment_label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_visuals(results_paths: Sequence[Path], default_language: str, visuals_dir: Path) -> None:
    records = _load_metric_records(results_paths, default_language=default_language)
    if not records:
        return

    visuals_dir.mkdir(parents=True, exist_ok=True)
    _write_text(visuals_dir / "qwk_heatmap.svg", _render_heatmap_svg(records))
    _write_text(visuals_dir / "best_qwk_by_language.svg", _render_bar_svg(records))

    best_qwk = max(records, key=lambda record: record.qwk)
    best_macro_f1 = max(records, key=lambda record: record.macro_f1)
    language_count = len({record.language for record in records})

    _write_text(
        visuals_dir / "badges" / "best-qwk.svg",
        _badge_svg("best qwk", f"{best_qwk.qwk:.3f}", _color_for_score(best_qwk.qwk)),
    )
    _write_text(
        visuals_dir / "badges" / "best-macro-f1.svg",
        _badge_svg("best macro-f1", f"{best_macro_f1.macro_f1:.3f}", _color_for_score(best_macro_f1.macro_f1)),
    )
    _write_text(
        visuals_dir / "badges" / "languages.svg",
        _badge_svg("languages", str(language_count), "#2563eb"),
    )


def main() -> None:
    args = parse_args()
    results_paths = collect_result_artifacts(
        download_root=Path(args.download_root),
        output_dir=Path(args.output_dir),
        default_language=args.default_language,
    )
    generate_visuals(
        results_paths=results_paths,
        default_language=args.default_language if args.default_language in SUPPORTED_LANGUAGES else "global",
        visuals_dir=Path(args.visuals_dir),
    )


if __name__ == "__main__":
    main()
