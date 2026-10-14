"""Import a FanDuel player-prop export (BettingPros public odds feed, supplied by the project manager) into the prop-line log.

Rules: docs/overnight_plan.md, "BettingPros FanDuel export, import and week-5 comparison". Only Over/Under rows of the four markets the model predicts; each selection gives an OPENING
row (stamped with the file's opening time) and a LAST row (stamped with the file's line-updated time, accepted up to 300 s after kickoff and noted).

  python scripts/import_bettingpros.py FanDuel_NFL_2026_Week5_Player_Props.csv [--push]
"""
import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from live import prop_lines as PL
from live import publish
from live import targets as TG

MARKETS = {"passing-yards": "Passing Yards", "rushing-yards": "Rushing Yards", "receiving-yards": "Receiving Yards", "receptions": "Receptions"}
TEAM_FIX = {"JAC": "JAX", "LAR": "LA"}
GRACE_SECONDS = 300


def load(path: Path) -> pl.DataFrame:
    d = pl.read_csv(path, infer_schema_length=0, encoding="utf8-lossy")
    return d.rename({c: c.lstrip("﻿") for c in d.columns})


def to_rows(d: pl.DataFrame, sched: pl.DataFrame) -> tuple:
    """(rows for log_prop_lines, counts) from the export; the game is resolved on season + week + the two teams."""
    fix = lambda c: pl.col(c).replace(TEAM_FIX)
    d = d.with_columns(away=fix("away_team"), home=fix("home_team"))
    keep = d.filter(pl.col("market").is_in(list(MARKETS)) & pl.col("selection").is_in(["Over", "Under"]))
    games = sched.select("game_id", "season", "week", "home_team", "away_team").unique().with_columns(season=pl.col("season").cast(pl.Utf8), week=pl.col("week").cast(pl.Utf8))
    keep = keep.join(games, left_on=["season", "week", "home", "away"], right_on=["season", "week", "home_team", "away_team"], how="left")
    base = dict(book=pl.lit("FanDuel"), game_id=pl.col("game_id"), player_name=pl.col("player"), book_market=pl.col("market").replace(MARKETS), side=pl.col("selection").str.to_lowercase())
    opens = keep.select(**base, line=pl.col("fanduel_opening_line"), price=pl.col("fanduel_opening_odds"), captured_at=pl.col("fanduel_opening_timestamp_utc"), note=pl.lit("open"))
    lasts = keep.select(**base, line=pl.col("line"), price=pl.col("american_odds"), captured_at=pl.col("line_updated_utc"), note=pl.lit("last"))
    counts = dict(file_rows=d.height, skipped_other_markets=d.height - keep.height, selections=keep.height, no_game=keep["game_id"].null_count())
    return pl.concat([opens, lasts]).filter(pl.col("game_id").is_not_null()), counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--log", type=Path, default=config.ROOT / "records" / "lines" / "prop_lines_log.csv")
    a = ap.parse_args(argv)
    if a.push:
        publish.sync_records(config.ROOT, ["records/lines"])
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    pcon = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    players = pcon.execute("SELECT DISTINCT gsis_id, display_name, latest_team, position FROM players WHERE gsis_id IS NOT NULL").pl()
    pcon.close()
    rows, counts = to_rows(load(a.csv), sched)
    res = PL.log_prop_lines(rows, datetime.now(timezone.utc), a.log, TG.schedule_frame(sched), players, source="bettingpros_import", grace_seconds=GRACE_SECONDS)
    keep_copy = a.log.parent / "imports" / a.csv.name
    keep_copy.parent.mkdir(parents=True, exist_ok=True)
    if not keep_copy.exists():
        shutil.copy(a.csv, keep_copy)                                           # the raw file as supplied, kept once, never changed
    print(counts, f"written {res.written}, unchanged {res.unchanged}, refused {res.refused}")
    if a.push:
        files = [p for p in (a.log, a.log.with_name("prop_lines_rejects.csv"), keep_copy) if p.exists()]
        print("published:", publish.publish(config.ROOT, files, f"prop lines: BettingPros FanDuel export {a.csv.name}: {res.written} written, {res.refused} refused"))
    return res


if __name__ == "__main__":
    main()
