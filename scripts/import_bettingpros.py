"""Import a FanDuel player-prop export (BettingPros public odds feed, supplied by the project manager) into the prop-line log.

Rules: docs/overnight_plan.md, "BettingPros FanDuel export, import and week-5 comparison". Only Over/Under rows of the markets the model predicts (the four of the first import plus pass attempts, pass completions and rush
attempts); each selection gives an OPENING row (stamped with the file's opening time) and a LAST row (stamped with the file's line-updated time).
A LAST row stamped after kickoff is refused by default beyond 300 s. With --grace-seconds N (N large enough to keep every row) it is kept instead and carries "stamp S s after kickoff" in its note, so an analysis
can run with and without the flagged rows (the feed overwrites a line once the game starts, so such a row may already know part of the game).

  python scripts/import_bettingpros.py FanDuel_NFL_2026_Week5_Player_Props.csv [--push] [--grace-seconds N]
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

MARKETS = {"passing-yards": "Passing Yards", "rushing-yards": "Rushing Yards", "receiving-yards": "Receiving Yards", "receptions": "Receptions",
           "passing-attempts": "Passing Attempts", "passing-completions": "Passing Completions", "rushing-attempts": "Rushing Attempts"}
TEAM_FIX = {"JAC": "JAX", "LAR": "LA"}
GRACE_SECONDS = 300


def load(path: Path) -> pl.DataFrame:
    d = pl.read_csv(path, infer_schema_length=0, encoding="utf8-lossy")
    d = d.rename({c: c.lstrip("﻿") for c in d.columns})
    # all-books exports name the row's own book's opening quote book_opening_*; filtered to one book these are that book's openings
    if "fanduel_opening_line" not in d.columns and "book_opening_line" in d.columns:
        d = d.rename({c: c.replace("book_opening_", "fanduel_opening_") for c in d.columns if c.startswith("book_opening_")})
    return d


def to_rows(d: pl.DataFrame, sched: pl.DataFrame) -> tuple:
    """(rows for log_prop_lines, counts) from the export; the game is resolved on season + week + the two teams. Regular season only; the book is the row's sportsbook; alternate lines are flagged by is_main_line."""
    fix = lambda c: pl.col(c).replace(TEAM_FIX)
    if "season_type" in d.columns:
        d = d.filter(pl.col("season_type") == "REG")
    d = d.with_columns(away=fix("away_team"), home=fix("home_team"))
    keep = d.filter(pl.col("market").is_in(list(MARKETS)) & pl.col("selection").is_in(["Over", "Under"]))
    games = sched.select("game_id", "season", "week", "home_team", "away_team").unique().with_columns(season=pl.col("season").cast(pl.Utf8), week=pl.col("week").cast(pl.Utf8))
    keep = keep.join(games, left_on=["season", "week", "home", "away"], right_on=["season", "week", "home_team", "away_team"], how="left")
    book = pl.col("sportsbook") if "sportsbook" in keep.columns else pl.lit("FanDuel")
    alt = (pl.col("is_main_line") == "False") if "is_main_line" in keep.columns else pl.lit(False)
    base = dict(book=book, game_id=pl.col("game_id"), player_name=pl.col("player"), book_market=pl.col("market").replace(MARKETS), side=pl.col("selection").str.to_lowercase(), is_alt=alt)
    has_open = pl.col("fanduel_opening_line").is_not_null() & pl.col("fanduel_opening_odds").is_not_null() & pl.col("fanduel_opening_timestamp_utc").is_not_null() if "fanduel_opening_line" in keep.columns else pl.lit(False)
    opens = keep.filter(has_open).select(**base, line=pl.col("fanduel_opening_line"), price=pl.col("fanduel_opening_odds"), captured_at=pl.col("fanduel_opening_timestamp_utc"), note=pl.lit("open"))
    lasts = keep.select(**base, line=pl.col("line"), price=pl.col("american_odds"), captured_at=pl.col("line_updated_utc"), note=pl.lit("last"))
    counts = dict(file_rows=d.height, skipped_other_markets=d.height - keep.height, selections=keep.height, no_game=keep["game_id"].null_count(), with_opening=int(has_open.sum()) if False else opens.height)
    return pl.concat([opens, lasts]).filter(pl.col("game_id").is_not_null()), counts


def roster_for(season: int) -> pl.DataFrame:
    """Who played in each game of `season` (game_id, gsis_id, display_name, team, position): played log plus the player master's names."""
    from live import history as H
    plog = H.read_played_logs(season)[0].filter(pl.col("season") == season).select("game_id", gsis_id="player_id", team="team", position="family")
    pcon = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    names = pcon.execute("SELECT DISTINCT gsis_id, display_name FROM players WHERE gsis_id IS NOT NULL").pl()
    pcon.close()
    return plog.join(names, on="gsis_id", how="inner").unique(["game_id", "gsis_id"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--grace-seconds", type=int, default=GRACE_SECONDS, help="keep a LAST row stamped up to this many seconds after kickoff (noted in the row); later ones are refused")
    ap.add_argument("--roster-season", type=int, default=None, help="match names inside each game's roster of this season (archives of past seasons)")
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
    roster = roster_for(a.roster_season) if a.roster_season else None
    res = PL.log_prop_lines(rows, datetime.now(timezone.utc), a.log, TG.schedule_frame(sched), players, source="bettingpros_import", grace_seconds=a.grace_seconds, roster=roster)
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
