"""Save the live season's tables that cannot be pulled again as they stood (injury reports, depth charts, rosters, the schedule), 2026 prospective record.

nflverse overwrites these as the week goes on, so a later pull shows the latest state, not what was known at the time. Each run writes a new
directory records/asof/<UTC date>T<hour>/<table>.parquet (zstd) and never touches an earlier one. Only live-season rows (after the holdout season)
are kept: the raw database holds 2025 depth charts with a null season, and none of that is saved or read here.

  python -m ingest.save_asof [--from-raw STAMP]     fresh pull now; --from-raw stores the 2026 slice of the raw database (pulled 2026-10-07) once
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

import config
from features.depth_live import dt_season

ASOF_DIR = config.ROOT / "records" / "asof"
TABLES = ("injuries", "rosters_weekly", "schedules", "depth_charts")
LIVE_FIRST = config.HOLDOUT_SEASON + 1
SLOTS = ((2, 12), (5, 18))          # Wednesday 12:00 UTC (after the Monday game, week's first reports), Saturday 18:00 UTC (the final reports before Sunday)


def is_asof_slot(now: datetime, tolerance_hours: float = 1.0) -> bool:
    """Whether `now` (UTC) is within the tolerance of a weekly as-of slot."""
    minutes = now.hour * 60 + now.minute
    return any(now.weekday() == d and abs(minutes - h * 60) <= tolerance_hours * 60 for d, h in SLOTS)


def live_rows(frames: dict) -> dict:
    """Each table limited to live-season rows: by season where the table has one, by the season of the snapshot timestamp `dt` (the season starts in March, so January and February 2026 charts are the holdout season) for the new depth-chart layout."""
    out = {}
    for name, f in frames.items():
        if "dt" in f.columns:
            f = f.filter(dt_season(pl.col("dt")) >= LIVE_FIRST)
        elif "season" in f.columns:
            f = f.filter(pl.col("season") >= LIVE_FIRST)
        out[name] = f
    return out


def save(frames: dict, out_dir: Path, now: datetime) -> Path:
    """Write the live-season rows of each table to out_dir/<date>T<hour>/; an existing snapshot of that hour is left untouched."""
    d = out_dir / now.strftime("%Y-%m-%dT%H")
    if d.exists():
        return d
    d.mkdir(parents=True)
    for name, f in live_rows(frames).items():
        f.write_parquet(d / f"{name}.parquet", compression="zstd")
    return d


def fresh_frames() -> dict:
    import nflreadpy
    load = {"injuries": nflreadpy.load_injuries, "rosters_weekly": nflreadpy.load_rosters_weekly, "schedules": nflreadpy.load_schedules,
            "depth_charts": nflreadpy.load_depth_charts}
    out = {}
    for name, fn in load.items():
        d = fn([LIVE_FIRST])
        out[name] = d.to_polars() if hasattr(d, "to_polars") else d
    return out


def raw_frames(raw_db=config.RAW_DUCKDB_PATH) -> dict:
    import duckdb
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        out = {t: con.execute(f"SELECT * FROM {t} WHERE season >= {LIVE_FIRST}").pl() for t in ("injuries", "rosters_weekly", "schedules")}
        out["depth_charts"] = con.execute("SELECT * FROM depth_charts WHERE season IS NULL AND dt >= ?", [f"{LIVE_FIRST}-01-01"]).pl()
    finally:
        con.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", default=None, help="store the live slice of the raw database under this stamp (e.g. 2026-10-07T00)")
    a = ap.parse_args()
    if a.from_raw:
        d = save(raw_frames(), ASOF_DIR, datetime.strptime(a.from_raw, "%Y-%m-%dT%H"))
    else:
        d = save(fresh_frames(), ASOF_DIR, datetime.now(timezone.utc))
    print(f"snapshot {d.relative_to(config.ROOT)}: " + ", ".join(f"{p.stem} {pl.read_parquet(p).height}" for p in sorted(d.iterdir())))
