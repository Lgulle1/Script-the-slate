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


def read_input_history(cutoff: date, raw_db=config.RAW_DUCKDB_PATH, target_games: pl.DataFrame | None = None, asof=None) -> InputHistory:
    """Game logs, team volumes and lineups for every game strictly before `cutoff`; `target_games` (season, week, team, gameday) adds the lineup rows of the
    games about to be played, taken from the newest chart at or before `asof` (naive UTC datetime; default: midnight of each game day)."""
    through = season_of(cutoff) if season_of(cutoff) >= config.HOLDOUT_SEASON else max(config.BACKTEST_SEASONS)

    def load():
        tl = bl.build_team_game_log(raw_db, max_season=through)
        return tl, bl.build_player_game_log(raw_db, max_season=through), vf.build_team_volume(raw_db, max_season=through)

    if through >= config.HOLDOUT_SEASON:
        with config.input_history(through):
            team_log, player_log, tv = load()
    else:
        team_log, player_log, tv = load()
    team_log = team_log.filter(pl.col("gameday") < cutoff)
    player_log = player_log.filter(pl.col("gameday") < cutoff)
    tv = tv.join(team_log.select("game_id").unique(), on="game_id", how="inner")
    ln = lu.build_lineups(raw_db, max_season=max(config.BACKTEST_SEASONS))
    if through >= config.HOLDOUT_SEASON:
        import duckdb
        games = team_log.filter(pl.col("season") >= config.HOLDOUT_SEASON).select("season", "week", "team", "gameday")
        if target_games is not None:
            tg = target_games.select("season", "week", "team", "gameday")
            games = pl.concat([games, tg.with_columns(pl.col("season").cast(games["season"].dtype), pl.col("week").cast(games["week"].dtype))])
        con = duckdb.connect(str(raw_db), read_only=True)
        try:
            charts = con.execute("SELECT DISTINCT dt, team, pos_abb, pos_rank, gsis_id, pos_grp, season FROM depth_charts WHERE season IS NULL AND dt IS NOT NULL").pl()
        finally:
            con.close()
        new = lineups_from_snapshots(charts, games, asof=asof, asof_games=None if target_games is None else target_games)
        ln = pl.concat([ln, new.select(ln.columns).cast(ln.schema)])
    return InputHistory(player_log=player_log, team_log=team_log, cutoff=cutoff, through_season=through, team_volume=tv, lineups=ln)


def lineups_from_snapshots(charts: pl.DataFrame, games: pl.DataFrame, asof=None, asof_games: pl.DataFrame | None = None) -> pl.DataFrame:
    """Lineup group ids (features.lineups.COMPONENTS) of 2025+ team-games from the timestamped charts (raw layout: dt, team, pos_abb, pos_rank, gsis_id, pos_grp).

    `games` has season, week, team, gameday. Each game takes the newest offensive chart strictly BEFORE its game day (the convention of eval/baselines.build_role_table),
    so nothing from the day itself leaks in; a game with no earlier chart gets no row (continuity then treats it as unchanged, never as a change).
    The games in `asof_games` (the targets of a live run) take the newest chart at or before `asof` instead: the run's own clock."""
    from datetime import datetime, time
    from features.depth_live import LEGACY_POSITION, OFFENSE_GROUP
    c = charts.filter(pl.col("season").is_null() & (pl.col("pos_grp") == OFFENSE_GROUP) & pl.col("gsis_id").is_not_null() & pl.col("pos_abb").is_in(list(LEGACY_POSITION)))
    c = c.with_columns(snap=pl.col("dt").str.to_datetime(time_zone="UTC", strict=False).dt.replace_time_zone(None)).drop_nulls("snap")
    snaps = c.select("team", "snap").unique().sort("snap")
    g = games.select("season", "week", "team", "gameday").unique().with_columns(game_start=pl.col("gameday").map_elements(lambda d: datetime.combine(d, time()), return_dtype=pl.Datetime("us")))
    exact = False
    if asof_games is not None and asof is not None:
        key = asof_games.select("season", "week", "team").unique().with_columns(_t=pl.lit(True))
        g = g.join(key, on=["season", "week", "team"], how="left").with_columns(
            game_start=pl.when(pl.col("_t").fill_null(False)).then(pl.lit(asof).cast(pl.Datetime("us"))).otherwise(pl.col("game_start"))).drop("_t")
    g = g.sort("game_start")
    picked = g.join_asof(snaps, left_on="game_start", right_on="snap", by="team", strategy="backward", allow_exact_matches=False, check_sortedness=False).drop_nulls("snap")
    if picked.is_empty():
        return pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32, "team": pl.String, **{k: pl.Int64 for k in lu.COMPONENTS}})
    d = picked.join(c.select("team", "snap", "gsis_id", position=pl.col("pos_abb").replace_strict(LEGACY_POSITION), depth_team=pl.col("pos_rank").cast(pl.String)),
                    on=["team", "snap"], how="inner").select("season", "week", "team", "gsis_id", "position", "depth_team")
    return lu.lineups_from_chart(d)
