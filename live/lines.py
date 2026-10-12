"""The schedule-line log of Phase 5 (P5.1): the sportsbook line as it stood, in its OWN table, read by nothing in the prediction path.

Append-only CSV keyed by (game_id, captured_at). A capture identical to the game's last logged line is not written again. Missing lines are empty, never 0.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import polars as pl

LINE_COLUMNS = ["spread_line", "total_line", "home_moneyline", "away_moneyline"]
COLUMNS = ["game_id", "captured_at"] + LINE_COLUMNS


def log_lines(schedule: pl.DataFrame, captured_at: datetime, path) -> int:
    """Append the lines of `schedule` (game_id plus LINE_COLUMNS) that differ from each game's last logged line; returns the number of rows written."""
    path = Path(path)
    last = {}
    if path.exists():
        for r in pl.read_csv(path, infer_schema_length=0).iter_rows(named=True):
            last[r["game_id"]] = tuple(r[c] or "" for c in LINE_COLUMNS)
    rows = []
    for r in schedule.iter_rows(named=True):
        vals = tuple("" if r.get(c) is None else _fmt(r[c]) for c in LINE_COLUMNS)
        if last.get(r["game_id"]) != vals:
            rows.append([r["game_id"], captured_at.isoformat(), *vals])
    if rows:
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(COLUMNS)
            w.writerows(rows)
    return len(rows)


def _fmt(x) -> str:
    return str(int(x)) if float(x) == int(x) else str(float(x))
