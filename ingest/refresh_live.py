"""Pull what the Phase 3 live run reads, append-only (every row keeps its pulled_at): schedules, player stats, depth charts and play-by-play (QB kneels).

  bootstrap()   every season from FEATURE_HISTORY_START to the current one: for a fresh container with no raw database
  refresh()     the current season only: the weekly top-up
The tables the live run does not read (snap counts, advanced stats, injuries before layer 4a is wired) are not pulled here.
"""
from __future__ import annotations

import nflreadpy

import config
from ingest import pull_nflverse as pn

TABLES = {"schedules": pn.pull_schedules, "player_stats": pn.pull_player_stats, "depth_charts": pn.pull_depth_charts, "pbp": pn.pull_pbp}


def current_season() -> int:
    return int(nflreadpy.get_current_season())


def _pull(seasons, db_path) -> dict:
    config.ensure_data_dirs()
    return {name: fn(seasons, db_path) for name, fn in TABLES.items()}


def bootstrap(db_path=config.RAW_DUCKDB_PATH) -> dict:
    return _pull(list(range(config.FEATURE_HISTORY_START, current_season() + 1)), db_path)


def refresh(db_path=config.RAW_DUCKDB_PATH) -> dict:
    return _pull([current_season()], db_path)
