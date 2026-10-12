"""Pull what the Phase 3 live run reads, append-only (every row keeps its pulled_at): schedules, player stats, depth charts, play-by-play, weekly rosters, injury reports and snap counts.

  bootstrap()   every season from 2018 (the first of the team ratings) to the current one: for a fresh container with no raw database
  refresh()     the current season only: the weekly top-up
The tables no live layer reads (advanced stats, participation) are not pulled here.
"""
from __future__ import annotations

import nflreadpy

import config
from ingest import pull_nflverse as pn

TABLES = {"schedules": pn.pull_schedules, "player_stats": pn.pull_player_stats, "depth_charts": pn.pull_depth_charts, "pbp": pn.pull_pbp,
          "rosters_weekly": pn.pull_rosters_weekly, "injuries": pn.pull_injuries, "snap_counts": pn.pull_snap_counts}
FIRST_SEASON = 2018        # team ratings (4b) start in 2018; everything else reads from FEATURE_HISTORY_START, so the bootstrap pulls the longest need for all


def current_season() -> int:
    return int(nflreadpy.get_current_season())


def _pull(seasons, db_path) -> dict:
    config.ensure_data_dirs()
    return {name: fn(seasons, db_path) for name, fn in TABLES.items()}


def bootstrap(db_path=config.RAW_DUCKDB_PATH) -> dict:
    return _pull(list(range(FIRST_SEASON, current_season() + 1)), db_path)


def refresh(db_path=config.RAW_DUCKDB_PATH) -> dict:
    return _pull([current_season()], db_path)
