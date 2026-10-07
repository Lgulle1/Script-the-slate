"""Pregame lineup groups per team-week, from the weekly depth charts (2020-24 schema).

Used to flag, for continuity weighting, whether a past game's environment differs from the game
being predicted. Each component is a stable integer id for the SET of players in that group
(-1 = chart missing, which downstream treats as "unchanged", never as a change):
  QB   depth-1 quarterback          RB       RB depth 1-2 (fullbacks are listed as RB)
  WRTE WR depth 1-3 + TE depth 1    OL       T/G/C depth 1 (the five starters)
HC and OC changes are NOT derivable here: the coaches table only holds the current staff, and
nflverse's coach columns are not trusted. They are therefore never flagged (see features/volume_features.py).
"""
from __future__ import annotations

import hashlib

import duckdb
import polars as pl

import config

COMPONENTS = ("QB", "RB", "WRTE", "OL")


def _sid(ids) -> int:
    ids = sorted(i for i in ids if i)
    if not ids:
        return -1
    return int.from_bytes(hashlib.md5(",".join(ids).encode()).digest()[:7], "big")


def build_lineups(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """One row per (team, season, week): integer ids for QB / RB / WRTE / OL groups."""
    cap = f" AND season <= {config.cap_season(max_season)}"  # raises config.HoldoutError for 2025+
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute(
            "SELECT season, week, club_code AS team, gsis_id, position, depth_team FROM "
            "(SELECT DISTINCT ON (season, week, club_code, gsis_id, depth_team, position) * FROM depth_charts "
            f" WHERE season IS NOT NULL AND week IS NOT NULL AND game_type = 'REG' AND formation = 'Offense'{cap} "
            " ORDER BY season, week, club_code, gsis_id, depth_team, position, pulled_at DESC)").pl()
    finally:
        con.close()
    d = d.with_columns(slot=pl.col("depth_team").cast(pl.Int32, strict=False))
    comp = {
        "QB": (pl.col("position") == "QB") & (pl.col("slot") == 1),
        "RB": (pl.col("position") == "RB") & pl.col("slot").is_in([1, 2]),
        "WRTE": ((pl.col("position") == "WR") & pl.col("slot").is_in([1, 2, 3])) | ((pl.col("position") == "TE") & (pl.col("slot") == 1)),
        "OL": pl.col("position").is_in(["T", "G", "C"]) & (pl.col("slot") == 1),
    }
    out = d.group_by("season", "week", "team").agg(
        [pl.col("gsis_id").filter(cond).unique().alias(name) for name, cond in comp.items()])
    out = out.with_columns([pl.col(c).map_elements(_sid, return_dtype=pl.Int64) for c in COMPONENTS])
    return out.sort("season", "week", "team")
