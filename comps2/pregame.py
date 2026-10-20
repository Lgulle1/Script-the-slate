"""Who is available for a game, from information known before kickoff only (2026-10-20, closing the look-ahead leaks of the comps and situations).

For (season, week, team): a player is UNAVAILABLE when the week's roster lists him as anything but active (inactive list, reserve/IR, cut, traded)
or the final injury report lists him Out or Doubtful. The inactive list comes out about 90 minutes before kickoff and the lines studied are near-kickoff
lines, so it is known when the bet is made; how many snaps anyone ends up playing is not, and is no longer used.

  available(season, week, team) -> (gsis ids available, gsis ids unavailable) via the tables built once by load().
"""
from __future__ import annotations

import duckdb
import polars as pl

import config

OUT_STATUS = ("Out", "Doubtful")


def load() -> pl.DataFrame:
    """One row per (season, week, team, gsis_id, pfr_id) with `available` (bool)."""
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    ro = con.execute("""SELECT * FROM (SELECT DISTINCT ON (season, week, team, gsis_id) CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS "week",
                        team, gsis_id, pfr_id, status FROM rosters_weekly WHERE gsis_id IS NOT NULL ORDER BY season, week, team, gsis_id, pulled_at DESC)""").pl()
    inj = con.execute("""SELECT * FROM (SELECT DISTINCT ON (season, week, gsis_id) CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS "week",
                         gsis_id, report_status FROM injuries WHERE gsis_id IS NOT NULL ORDER BY season, week, gsis_id, pulled_at DESC)""").pl()
    con.close()
    ro = ro.join(inj, on=["season", "week", "gsis_id"], how="left")
    return ro.with_columns(available=(pl.col("status") == "ACT") & ~pl.col("report_status").fill_null("").is_in(list(OUT_STATUS)))
