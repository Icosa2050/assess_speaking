#!/usr/bin/env python3
"""Render a lightweight dashboard for logged speaking assessments."""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional

from rich.console import Console
from rich.table import Table


@dataclass
class Record:
    timestamp: datetime
    audio: str
    whisper: str
    llm: str
    label: str
    duration_sec: Optional[float]
    wpm: Optional[float]
    word_count: Optional[int]
    overall: Optional[float]
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
                    audio=row.get("audio", ""),
                    whisper=row.get("whisper", ""),
                    llm=row.get("llm", ""),
                    label=row.get("label", ""),
                    duration_sec=parse_float(row.get("duration_sec", "")),
                    wpm=parse_float(row.get("wpm", "")),
                    word_count=parse_int(row.get("word_count", "")),
                    overall=parse_float(row.get("overall", "")),
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
    return {
        "count": len(records),
        "avg_wpm": round(mean(wpm_values), 1) if wpm_values else None,
        "avg_overall": round(mean(overall_values), 2) if overall_values else None,
        "best_overall": max(overall_values) if overall_values else None,
        "latest": records[-1],
    }


def render_terminal(records: List[Record], summary: dict) -> None:
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
    if summary.get("best_overall") is not None:
        meta_line.append(f"Best Overall: {summary['best_overall']}")
    if meta_line:
        console.print("• " + " | ".join(meta_line) + "\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim")
    table.add_column("Datum")
    table.add_column("Label")
    table.add_column("Audio")
    table.add_column("WPM", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("Whisper", style="dim")
    table.add_column("LLM", style="dim")

    for idx, rec in enumerate(records, start=1):
        table.add_row(
            str(idx),
            rec.timestamp.strftime("%Y-%m-%d %H:%M"),
            rec.label or "–",
            rec.audio,
            f"{rec.wpm:.1f}" if rec.wpm is not None else "–",
            f"{rec.overall:.2f}" if rec.overall is not None else "–",
            rec.whisper,
            rec.llm,
        )
    console.print(table)
    console.print("\nNutze '--export-html pfad.html' für eine statische Übersicht.\n")


def render_html(records: List[Record], summary: dict) -> str:
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
            f"<td>{rec.audio}</td>"
            f"<td>{fmt(rec.wpm)}</td>"
            f"<td>{fmt(rec.overall, 2)}</td>"
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
    if summary.get("best_overall") is not None:
        summary_html.append(f"<strong>Best Overall:</strong> {fmt(summary['best_overall'], 2)}")

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
<tr><th>#</th><th>Datum</th><th>Label</th><th>Audio</th><th>WPM</th><th>Overall</th><th>Whisper</th><th>LLM</th><th>Report</th></tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
<div class="footer">Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Show logged assessment runs as a dashboard.")
    parser.add_argument(
        "--log-dir", default="reports", help="Directory containing history.csv (default: reports)")
    parser.add_argument(
        "--export-html", help="Optional path to write an HTML snapshot of the dashboard")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    history_path = log_dir / "history.csv"
    records = load_history(history_path)
    summary = summarise(records)

    render_terminal(records, summary)

    if args.export_html:
        html = render_html(records, summary)
        out_path = Path(args.export_html)
        out_path.write_text(html, encoding="utf-8")
        Console().print(f"[green]HTML Dashboard gespeichert unter {out_path}[/green]")


if __name__ == "__main__":  # pragma: no cover
    main()
