"""The depth charts since the 2025 season, read back into the shape the Phase 3-4 loaders use.

nflverse changed the layout: a chart is now a snapshot (`dt`, an ISO timestamp, one per team and day) with no season and no week, positions are
`pos_abb` (LT / LG / C / RG / RT, WR, TE, QB, RB, FB ...), the offense is the group `pos_grp = '3WR 1TE'`, and `pos_rank` is the depth rank over the
position's slots (WR slots 1, 2 and 8 hold ranks 1-3, 4-6 ...). The old loaders (features/lineups.py, features/phase4_inputs.py, models/game_state.py)
filter on season and week, so the raw table's 1.17 million new-layout rows (season NULL) are invisible to them: harmless for the 2020-2024 backtest,
fatal for live predictions. This module is the live path: `chart_as_of(frame, cutoff, week)` is the latest chart per team at or before the cutoff,
with the old columns (season, week, team, gsis_id, position, depth_team, formation).

A chart belongs to the season that began in March of its year (a February 2026 chart is the 2025 season). The 2025 season is the locked holdout:
every function here refuses it (config.HoldoutError) and never returns one of its rows.
"""
from __future__ import annotations

import polars as pl

import config

OFFENSE_GROUP = "3WR 1TE"
LEGACY_POSITION = {"QB": "QB", "RB": "RB", "FB": "FB", "WR": "WR", "TE": "TE", "LT": "T", "RT": "T", "LG": "G", "RG": "G", "C": "C"}
SEASON_STARTS_IN_MONTH = 3


def dt_season(dt: pl.Expr) -> pl.Expr:
    """The season of a chart timestamp: its year from March on, the year before for January and February."""
    ts = dt.cast(pl.String).str.slice(0, 7)
    year, month = ts.str.slice(0, 4).cast(pl.Int32, strict=False), ts.str.slice(5, 2).cast(pl.Int32, strict=False)
    return pl.when(month >= SEASON_STARTS_IN_MONTH).then(year).otherwise(year - 1).cast(pl.Int32)


def live_charts(frame: pl.DataFrame) -> pl.DataFrame:
    """The new-layout rows (season NULL, `dt` set) of the live seasons only. Raises HoldoutError if the frame asks for nothing but holdout rows."""
    new = frame.filter(pl.col("season").is_null() & pl.col("dt").is_not_null()).with_columns(chart_season=dt_season(pl.col("dt")))
    live = new.filter(pl.col("chart_season") > config.HOLDOUT_SEASON)
    if new.height and not live.height:
        raise config.HoldoutError(f"every chart in the frame is from season {config.HOLDOUT_SEASON} or earlier, the locked holdout")
    return live


def chart_as_of(frame: pl.DataFrame, cutoff: str, week: int) -> pl.DataFrame:
    """The offensive depth chart of each team as of `cutoff` (an ISO timestamp, e.g. kickoff or the snapshot time): its newest chart at or before it.

    A team with no chart yet has no rows (never a later chart). `week` is the week the chart is for; the caller knows the schedule."""
    season = pl.Series([cutoff]).to_frame("c").select(dt_season(pl.col("c"))).item()
    if season <= config.HOLDOUT_SEASON:
        raise config.HoldoutError(f"a chart as of {cutoff} is from season {season}, the locked holdout ({config.HOLDOUT_SEASON})")
    d = (live_charts(frame).filter((pl.col("dt") <= cutoff) & (pl.col("pos_grp") == OFFENSE_GROUP) & pl.col("gsis_id").is_not_null()
                                   & pl.col("pos_abb").is_in(list(LEGACY_POSITION)))
         .filter(pl.col("dt") == pl.col("dt").max().over("team")))
    return (d.select(season=pl.col("chart_season"), week=pl.lit(week, dtype=pl.Int32), team="team", gsis_id="gsis_id",
                     position=pl.col("pos_abb").replace_strict(LEGACY_POSITION), depth_team=pl.col("pos_rank").cast(pl.String), formation=pl.lit("Offense"), dt="dt")
            .unique().sort("team", "position", "depth_team", "gsis_id"))
