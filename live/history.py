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


def read_played_logs(season: int, raw_db=config.RAW_DUCKDB_PATH) -> tuple:
    """(player_log, team_log) of every game of seasons <= `season` as the raw database holds it NOW: for checking a past live run against what happened (retro checks, grading) and for
    trailing yardsticks. Outcome-side reading, never a model input; it goes through this module so that this file stays the only one that opens the holdout (config.input_history)."""
    if season >= config.HOLDOUT_SEASON:
        with config.input_history(season):
            return bl.build_player_game_log(raw_db, max_season=season), bl.build_team_game_log(raw_db, max_season=season)
    return bl.build_player_game_log(raw_db, max_season=season), bl.build_team_game_log(raw_db, max_season=season)


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


@dataclass(frozen=True)
class InjuryInputs:
    """The 4a frames for a live run: player weeks, role frame (both holding the target games), injury reports and weekly roster statuses."""
    player_weeks: pl.DataFrame
    frame: pl.DataFrame
    injuries: pl.DataFrame
    rosters: pl.DataFrame
    through_season: int
    input_only: bool = True


def read_input_injury_inputs(cutoff: date, raw_db, targets: pl.DataFrame, future_games: pl.DataFrame | None = None) -> InjuryInputs:
    """The layer-4a inputs through the scoped unlock: 2025+ games enter as input history only (plan P5.7 item 4). `targets`: season, week, team, game_id, gameday."""
    from models import injuries as ij
    through = season_of(cutoff) if season_of(cutoff) >= config.HOLDOUT_SEASON else max(config.BACKTEST_SEASONS)

    def load():
        return (ij.build_player_weeks(raw_db, through, targets=targets, future_games=future_games), ij.build_role_frame(raw_db, through, targets=targets, future_games=future_games),
                ij.load_injury_rows(raw_db, through), ij.load_roster_status(raw_db, through))

    if through >= config.HOLDOUT_SEASON:
        with config.input_history(through):
            pw, frame, inj, ros = load()
    else:
        pw, frame, inj, ros = load()
    return InjuryInputs(pw, frame, inj, ros, through)


@dataclass(frozen=True)
class GameStateInputs:
    """The 4b.2 pre-game features of every team-game (played or scheduled) from the history through the target week, and the schedule rows."""
    games: pl.DataFrame
    features: pl.DataFrame
    through_season: int
    input_only: bool = True


def read_input_game_state(cutoff: date, raw_db, fit_floor: date | None = None) -> GameStateInputs:
    """Layer-4b inputs through the scoped unlock (plan P5.7 item 4): ratings, plays, QB books, depth charts, injury statuses and rosters of 2025+ games enter as history
    only. The 4a.1 status models behind the QB-absence probability are fit on games before `fit_floor` (2020-2024 rows only) when it is given."""
    from features import team_ratings as tr
    from models import game_state as gs
    through = season_of(cutoff) if season_of(cutoff) >= config.HOLDOUT_SEASON else max(config.BACKTEST_SEASONS)

    def load():
        inp = gs.load_inputs(raw_db, through)
        ratings = tr.compute_ratings(inp.plays, n_boot=0)
        feats = gs.pregame_features(inp, ratings, gs.status_models_by_week(inp, raw_db, through, fit_floor=fit_floor))
        return inp.games, feats

    if through >= config.HOLDOUT_SEASON:
        with config.input_history(through):
            games, feats = load()
    else:
        games, feats = load()
    return GameStateInputs(games, feats, through)
