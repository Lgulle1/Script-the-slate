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


@dataclass(frozen=True)
class InputHistory:
    player_log: pl.DataFrame
    team_log: pl.DataFrame
    cutoff: date
    through_season: int
    input_only: bool = True


def season_of(day: date) -> int:
    """The NFL season a date belongs to (it starts in March)."""
    return day.year if day.month >= 3 else day.year - 1


def read_input_history(cutoff: date, raw_db=config.RAW_DUCKDB_PATH) -> InputHistory:
    through = season_of(cutoff) if season_of(cutoff) >= config.HOLDOUT_SEASON else max(config.BACKTEST_SEASONS)
    if through >= config.HOLDOUT_SEASON:
        with config.input_history(through):
            team_log = bl.build_team_game_log(raw_db, max_season=through)
            player_log = bl.build_player_game_log(raw_db, max_season=through)
    else:
        team_log = bl.build_team_game_log(raw_db, max_season=through)
        player_log = bl.build_player_game_log(raw_db, max_season=through)
    team_log = team_log.filter(pl.col("gameday") < cutoff)
    player_log = player_log.filter(pl.col("gameday") < cutoff)
    return InputHistory(player_log=player_log, team_log=team_log, cutoff=cutoff, through_season=through)


def lineups_from_snapshots(charts: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Lineup group ids (features.lineups.COMPONENTS) of 2025+ team-games from the timestamped charts (raw layout: dt, team, pos_abb, pos_rank, gsis_id, pos_grp).

    `games` has season, week, team, gameday. Each game takes the newest offensive chart strictly BEFORE its game day (the convention of eval/baselines.build_role_table),
    so nothing from the day itself leaks in; a game with no earlier chart gets no row (continuity then treats it as unchanged, never as a change)."""
    from datetime import datetime, time
    from features.depth_live import LEGACY_POSITION, OFFENSE_GROUP
    c = charts.filter(pl.col("season").is_null() & (pl.col("pos_grp") == OFFENSE_GROUP) & pl.col("gsis_id").is_not_null() & pl.col("pos_abb").is_in(list(LEGACY_POSITION)))
    c = c.with_columns(snap=pl.col("dt").str.to_datetime(time_zone="UTC", strict=False).dt.replace_time_zone(None)).drop_nulls("snap")
    snaps = c.select("team", "snap").unique().sort("snap")
    g = games.select("season", "week", "team", "gameday").unique().with_columns(game_start=pl.col("gameday").map_elements(lambda d: datetime.combine(d, time()), return_dtype=pl.Datetime("us"))).sort("game_start")
    picked = g.join_asof(snaps, left_on="game_start", right_on="snap", by="team", strategy="backward", allow_exact_matches=False, check_sortedness=False).drop_nulls("snap")
    if picked.is_empty():
        return pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32, "team": pl.String, **{k: pl.Int64 for k in lu.COMPONENTS}})
    d = picked.join(c.select("team", "snap", "gsis_id", position=pl.col("pos_abb").replace_strict(LEGACY_POSITION), depth_team=pl.col("pos_rank").cast(pl.String)),
                    on=["team", "snap"], how="inner").select("season", "week", "team", "gsis_id", "position", "depth_team")
    return lu.lineups_from_chart(d)
