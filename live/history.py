"""The one reader of 2025+ games, as INPUT HISTORY for live targets (plan "Phase 5 gates", P5.4 guardrails 1-3; your decision of 2026-10-12).

`read_input_history(cutoff)` returns the player and team game logs for every game with gameday STRICTLY BEFORE `cutoff` (the target's week cutoff: its first
kickoff), from FEATURE_HISTORY_START on, including the 2025 season and 2026 to date when the cutoff is in those seasons. It is the only code that opens
config.input_history. The result is an `InputHistory`, not a backtest `History` / `BacktestData`: it is for building features of a target that has not
happened yet, never for fitting a model and never as a target. Models stay fit on 2020-2024 rows only until the Phase 6 benchmark has run once and its
result is recorded in the plan. A cutoff in the 2025 season (the Phase 6 benchmark path) sees nothing from 2026 and nothing on or after the cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

import config
from eval import baselines as bl
from features import lineups as lu
from features import volume_features as vf


@dataclass(frozen=True)
class InputHistory:
    player_log: pl.DataFrame
    team_log: pl.DataFrame
    cutoff: date
    through_season: int
    team_volume: pl.DataFrame | None = None     # team-game volumes of the history games (feature input)
    lineups: pl.DataFrame | None = None          # lineup group ids per (team, season, week): weekly charts to 2024, timestamped charts from 2025
    input_only: bool = True


def season_of(day: date) -> int:
    """The NFL season a date belongs to (it starts in March)."""
    return day.year if day.month >= 3 else day.year - 1


def read_input_history(cutoff: date, raw_db=config.RAW_DUCKDB_PATH) -> InputHistory:
    """Game logs, team volumes and lineups for every game strictly before `cutoff`.

    `raw_db` is normally the as-of working copy of live/derive.py, whose 2025+ weekly depth charts (made from the timestamped snapshots) let the unchanged loaders
    see 2025+ games exactly as they see 2020-2024; the target week's charts in it are as of the run clock."""
    through = season_of(cutoff) if season_of(cutoff) >= config.HOLDOUT_SEASON else max(config.BACKTEST_SEASONS)

    def load():
        tl = bl.build_team_game_log(raw_db, max_season=through)
        return tl, bl.build_player_game_log(raw_db, max_season=through), vf.build_team_volume(raw_db, max_season=through), lu.build_lineups(raw_db, max_season=through)

    if through >= config.HOLDOUT_SEASON:
        with config.input_history(through):
            team_log, player_log, tv, ln = load()
    else:
        team_log, player_log, tv, ln = load()
    team_log = team_log.filter(pl.col("gameday") < cutoff)
    player_log = player_log.filter(pl.col("gameday") < cutoff)
    tv = tv.join(team_log.select("game_id").unique(), on="game_id", how="inner")
    return InputHistory(player_log=player_log, team_log=team_log, cutoff=cutoff, through_season=through, team_volume=tv, lineups=ln)
