#!/usr/bin/env python3
"""Render a lightweight dashboard for logged speaking assessments."""
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional

from progress_analysis import (
    filter_records,
    format_top_counts,
    latest_priorities,
    task_family_progress,
)
from rich.console import Console
from rich.table import Table


@dataclass
class Record:
    timestamp: datetime
    session_id: str
    schema_version: Optional[int]
    speaker_id: str
    task_family: str
    theme: str
    audio: str
    whisper: str
    llm: str
    label: str
    target_duration_sec: Optional[float]
    duration_sec: Optional[float]
    wpm: Optional[float]
    word_count: Optional[int]
    overall: Optional[float]
    final_score: Optional[float]
    band: Optional[int]
    requires_human_review: Optional[bool]
    top_priorities: tuple[str, ...]
    grammar_error_categories: tuple[str, ...]
    coherence_issue_categories: tuple[str, ...]
    report_path: str


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(round(float(value)))
    except ValueError:
        return None


def parse_bool(value: str) -> Optional[bool]:
    value = value.strip().lower()
    if not value:
        return None
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def parse_pipe_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split("|") if item.strip())


def load_history(history_path: Path) -> List[Record]:
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with history_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid timestamp '{row.get('timestamp')}'") from exc
            rows.append(
                Record(
                    timestamp=ts,
                    session_id=row.get("session_id", ""),
                    schema_version=parse_int(row.get("schema_version", "")),
                    speaker_id=row.get("speaker_id", ""),
                    task_family=row.get("task_family", ""),
                    theme=row.get("theme", ""),
                    audio=row.get("audio", ""),
                    whisper=row.get("whisper", ""),
                    llm=row.get("llm", ""),
                    label=row.get("label", ""),
                    target_duration_sec=parse_float(row.get("target_duration_sec", "")),
                    duration_sec=parse_float(row.get("duration_sec", "")),
                    wpm=parse_float(row.get("wpm", "")),
                    word_count=parse_int(row.get("word_count", "")),
                    overall=parse_float(row.get("overall", "")),
                    final_score=parse_float(row.get("final_score", "")),
                    band=parse_int(row.get("band", "")),
                    requires_human_review=parse_bool(row.get("requires_human_review", "")),
                    top_priorities=tuple(
                        item
                        for item in (
                            row.get("top_priority_1", "").strip(),
                            row.get("top_priority_2", "").strip(),
                            row.get("top_priority_3", "").strip(),
                        )
                        if item
                    ),
                    grammar_error_categories=parse_pipe_list(row.get("grammar_error_categories", "")),
                    coherence_issue_categories=parse_pipe_list(row.get("coherence_issue_categories", "")),
                    report_path=row.get("report_path", ""),
                )
            )
    return sorted(rows, key=lambda r: r.timestamp)


def summarise(records: Iterable[Record]) -> dict:
    records = list(records)
    if not records:
        return {"count": 0}
    wpm_values = [r.wpm for r in records if r.wpm is not None]
    overall_values = [r.overall for r in records if r.overall is not None]
    final_values = [r.final_score for r in records if r.final_score is not None]
    return {
        "count": len(records),
        "avg_wpm": round(mean(wpm_values), 1) if wpm_values else None,
        "avg_overall": round(mean(overall_values), 2) if overall_values else None,
        "avg_final": round(mean(final_values), 2) if final_values else None,
        "best_overall": max(overall_values) if overall_values else None,
        "best_final": max(final_values) if final_values else None,
        "latest": records[-1],
    }


def load_progress_delta(report_path: str) -> Optional[dict]:
    if not report_path:
        return None
    path = Path(report_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    report = payload.get("report", payload)
    progress_delta = report.get("progress_delta")
    return progress_delta if isinstance(progress_delta, dict) else None


def render_terminal(
    records: List[Record],
    summary: dict,
    *,
    family_rows: Optional[List[dict]] = None,
    progress_delta: Optional[dict] = None,
) -> None:
    console = Console()
    if not records:
        console.print("[bold yellow]Keine Einträge in history.csv gefunden.[/bold yellow]")
        return

    console.print("\n[bold underline]Assess Speaking – Verlauf[/bold underline]\n")
    meta_line = []
    if summary.get("count"):
        meta_line.append(f"Runs: {summary['count']}")
    if summary.get("avg_wpm") is not None:
        meta_line.append(f"∅ WPM: {summary['avg_wpm']}")
    if summary.get("avg_overall") is not None:
        meta_line.append(f"∅ Overall: {summary['avg_overall']}")
    if summary.get("avg_final") is not None:
        meta_line.append(f"∅ Final: {summary['avg_final']}")
    if summary.get("best_overall") is not None:
        meta_line.append(f"Best Overall: {summary['best_overall']}")
    if summary.get("best_final") is not None:
        meta_line.append(f"Best Final: {summary['best_final']}")
    if meta_line:
        console.print("• " + " | ".join(meta_line) + "\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim")
    table.add_column("Datum")
    table.add_column("Label")
    table.add_column("Task")
    table.add_column("Audio")
    table.add_column("WPM", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("Final", justify="right")
    table.add_column("Whisper", style="dim")
    table.add_column("LLM", style="dim")

    for idx, rec in enumerate(records, start=1):
        table.add_row(
            str(idx),
            rec.timestamp.strftime("%Y-%m-%d %H:%M"),
            rec.label or "–",
            rec.task_family or "–",
            rec.audio,
            f"{rec.wpm:.1f}" if rec.wpm is not None else "–",
            f"{rec.overall:.2f}" if rec.overall is not None else "–",
            f"{rec.final_score:.2f}" if rec.final_score is not None else "–",
            rec.whisper,
            rec.llm,
        )
    console.print(table)

    if family_rows:
        console.print("\n[bold]Task-Family Analyse[/bold]\n")
        family_table = Table(show_header=True, header_style="bold")
        family_table.add_column("Task")
        family_table.add_column("Runs", justify="right")
        family_table.add_column("Avg Final", justify="right")
        family_table.add_column("Latest Final", justify="right")
        family_table.add_column("Recurring Grammar")
        family_table.add_column("Recurring Coherence")
        for row in family_rows:
            family_table.add_row(
                row["task_family"],
                str(row["count"]),
                f"{row['avg_final']:.2f}" if row["avg_final"] is not None else "–",
                f"{row['latest_final']:.2f}" if row["latest_final"] is not None else "–",
                format_top_counts(row["grammar_counts"]),
                format_top_counts(row["coherence_counts"]),
            )
        console.print(family_table)

    if len({rec.task_family for rec in records if rec.task_family}) == 1:
        changes = latest_priorities(records)
        latest_text = ", ".join(changes["latest"]) if changes["latest"] else "–"
        previous_text = ", ".join(changes["previous"]) if changes["previous"] else "–"
        console.print("\n[bold]Prioritätenvergleich[/bold]")
        console.print(f"Neueste Prioritäten: {latest_text}")
        console.print(f"Vorherige Prioritäten: {previous_text}")
    if progress_delta:
        score_delta = progress_delta.get("score_delta", {})
        console.print("\n[bold]Progress Delta[/bold]")
        console.print(
            "Letzte Änderung: "
            f"Final {score_delta.get('final', '–')} | "
            f"Overall {score_delta.get('overall', '–')} | "
            f"WPM {score_delta.get('wpm', '–')}"
        )
        console.print(
            "Neue Prioritäten: "
            + (", ".join(progress_delta.get("new_priorities", [])) or "–")
        )
        console.print(
            "Wiederkehrende Grammatik: "
            + (", ".join(progress_delta.get("repeating_grammar_categories", [])) or "–")
        )
    console.print("\nNutze '--export-html pfad.html' für eine statische Übersicht.\n")


def render_html(
    records: List[Record],
    summary: dict,
    *,
    family_rows: Optional[List[dict]] = None,
    progress_delta: Optional[dict] = None,
) -> str:
    def fmt(value: Optional[float], digits: int = 1) -> str:
        if value is None:
            return "–"
        return f"{value:.{digits}f}"

    rows_html = []
    for idx, rec in enumerate(records, start=1):
        rows_html.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{rec.timestamp.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td>{rec.label or '&#8211;'}</td>"
            f"<td>{rec.task_family or '&#8211;'}</td>"
            f"<td>{rec.audio}</td>"
            f"<td>{fmt(rec.wpm)}</td>"
            f"<td>{fmt(rec.overall, 2)}</td>"
            f"<td>{fmt(rec.final_score, 2)}</td>"
            f"<td>{rec.whisper}</td>"
            f"<td>{rec.llm}</td>"
            f"<td><a href='{rec.report_path}'>JSON</a></td>"
            "</tr>"
        )

    summary_html = []
    if summary.get("count"):
        summary_html.append(f"<strong>Runs:</strong> {summary['count']}")
    if summary.get("avg_wpm") is not None:
        summary_html.append(f"<strong>∅ WPM:</strong> {fmt(summary['avg_wpm'])}")
    if summary.get("avg_overall") is not None:
        summary_html.append(f"<strong>∅ Overall:</strong> {fmt(summary['avg_overall'], 2)}")
    if summary.get("avg_final") is not None:
        summary_html.append(f"<strong>∅ Final:</strong> {fmt(summary['avg_final'], 2)}")
    if summary.get("best_overall") is not None:
        summary_html.append(f"<strong>Best Overall:</strong> {fmt(summary['best_overall'], 2)}")
    if summary.get("best_final") is not None:
        summary_html.append(f"<strong>Best Final:</strong> {fmt(summary['best_final'], 2)}")

    family_html = ""
    if family_rows:
        family_rows_html = []
        for row in family_rows:
            family_rows_html.append(
                "<tr>"
                f"<td>{row['task_family']}</td>"
                f"<td>{row['count']}</td>"
                f"<td>{fmt(row['avg_final'], 2)}</td>"
                f"<td>{fmt(row['latest_final'], 2)}</td>"
                f"<td>{format_top_counts(row['grammar_counts'])}</td>"
                f"<td>{format_top_counts(row['coherence_counts'])}</td>"
                "</tr>"
            )
        family_html = f"""
<h2>Task-Family Analyse</h2>
<table>
<thead>
<tr><th>Task</th><th>Runs</th><th>Avg Final</th><th>Latest Final</th><th>Recurring Grammar</th><th>Recurring Coherence</th></tr>
</thead>
<tbody>
{''.join(family_rows_html)}
</tbody>
</table>
"""

    priorities_html = ""
    if len({rec.task_family for rec in records if rec.task_family}) == 1 and records:
        changes = latest_priorities(records)
        priorities_html = f"""
<h2>Prioritätenvergleich</h2>
<div class="meta">
<strong>Neueste Prioritäten:</strong> {', '.join(changes['latest']) if changes['latest'] else '–'}
&nbsp;|&nbsp;
<strong>Vorherige Prioritäten:</strong> {', '.join(changes['previous']) if changes['previous'] else '–'}
</div>
"""

    progress_delta_html = ""
    if progress_delta:
        score_delta = progress_delta.get("score_delta", {})
        progress_delta_html = f"""
<h2>Progress Delta</h2>
<div class="meta">
<strong>Vorige Session:</strong> {progress_delta.get('previous_session_id') or '–'}
&nbsp;|&nbsp;
<strong>Final Δ:</strong> {fmt(score_delta.get('final'), 2) if score_delta.get('final') is not None else '–'}
&nbsp;|&nbsp;
<strong>Overall Δ:</strong> {fmt(score_delta.get('overall'), 2) if score_delta.get('overall') is not None else '–'}
&nbsp;|&nbsp;
<strong>WPM Δ:</strong> {fmt(score_delta.get('wpm'), 2) if score_delta.get('wpm') is not None else '–'}
</div>
<div class="meta">
<strong>Neue Prioritäten:</strong> {', '.join(progress_delta.get('new_priorities', [])) or '–'}
&nbsp;|&nbsp;
<strong>Erledigt/entfallen:</strong> {', '.join(progress_delta.get('resolved_priorities', [])) or '–'}
</div>
<div class="meta">
<strong>Wiederkehrende Grammatik:</strong> {', '.join(progress_delta.get('repeating_grammar_categories', [])) or '–'}
&nbsp;|&nbsp;
<strong>Wiederkehrende Kohärenz:</strong> {', '.join(progress_delta.get('repeating_coherence_categories', [])) or '–'}
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Assess Speaking – Verlauf</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #f8f9fb; color: #222; }}
 h1 {{ margin-bottom: 0.2rem; }}
 .meta {{ margin-bottom: 1.5rem; font-size: 0.95rem; color: #444; }}
 table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
 th, td {{ padding: 0.6rem 0.8rem; border-bottom: 1px solid #e6e8eb; text-align: left; }}
 th {{ background: #eef1f7; font-weight: 600; }}
 tr:hover {{ background: #f5f7fb; }}
 a {{ color: #0a84ff; text-decoration: none; }}
 a:hover {{ text-decoration: underline; }}
 .footer {{ margin-top: 2rem; font-size: 0.85rem; color: #666; }}
</style>
</head>
<body>
<h1>Assess Speaking – Verlauf</h1>
<div class="meta">{' &nbsp;|&nbsp; '.join(summary_html) if summary_html else 'Keine Daten.'}</div>
<table>
<thead>
<tr><th>#</th><th>Datum</th><th>Label</th><th>Task</th><th>Audio</th><th>WPM</th><th>Overall</th><th>Final</th><th>Whisper</th><th>LLM</th><th>Report</th></tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
{family_html}
{priorities_html}
{progress_delta_html}
<div class="footer">Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Show logged assessment runs as a dashboard.")
    parser.add_argument(
        "--log-dir", default="reports", help="Directory containing history.csv (default: reports)")
    parser.add_argument("--speaker-id", help="Optional speaker id filter")
    parser.add_argument("--task-family", help="Optional task family filter")
    parser.add_argument(
        "--export-html", help="Optional path to write an HTML snapshot of the dashboard")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    history_path = log_dir / "history.csv"
    records = load_history(history_path)
    filtered_records = filter_records(records, speaker_id=args.speaker_id, task_family=args.task_family)
    summary = summarise(filtered_records)
    family_rows = task_family_progress(filtered_records if args.task_family else filter_records(records, speaker_id=args.speaker_id))
    progress_delta = load_progress_delta(filtered_records[-1].report_path) if filtered_records else None

    render_terminal(filtered_records, summary, family_rows=family_rows, progress_delta=progress_delta)

    if args.export_html:
        html = render_html(filtered_records, summary, family_rows=family_rows, progress_delta=progress_delta)
        out_path = Path(args.export_html)
        out_path.write_text(html, encoding="utf-8")
        Console().print(f"[green]HTML Dashboard gespeichert unter {out_path}[/green]")


if __name__ == "__main__":  # pragma: no cover
    main()
