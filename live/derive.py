"""The as-of working copy of the raw database for one live run (plan P5.2 and P5.7).

`build_live_db(raw_db, out_path, now, target_season, target_week)`:
  1. copies the raw database file (the raw file itself is never opened for writing);
  2. deletes every row pulled after `now` (all raw tables carry pulled_at), so everything a loader reads is as of the run clock;
  3. adds, for every 2025+ regular-season game, a weekly offensive depth chart in the OLD layout (season, week, club_code, gsis_id, position, depth_position, depth_team,
     formation, game_type) made from the timestamped snapshots: the newest snapshot strictly before the game day, as eval/baselines.build_role_table does; the games of the
     target week take the newest snapshot at or before `now` instead. With these rows every existing loader (lineups, chart flags, expected roles, injuries, simulation inputs)
     works unchanged on 2025+ games.
The copy holds 2025 rows and is read only through live/history.py under config.input_history; it lives under data/processed (git-ignored) and is rebuilt every run.
"""
from __future__ import annotations

import shutil
from datetime import datetime, time
from pathlib import Path

import duckdb
import polars as pl

import config
from features.depth_live import LEGACY_POSITION, OFFENSE_GROUP

FIRST_WEEKLY_SEASON = config.HOLDOUT_SEASON


def _raw_tables(con) -> list:
    return [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' AND table_type = 'BASE TABLE'").fetchall()]


def build_live_db(raw_db, out_path, now: datetime, target_season: int | None = None, target_week: int | None = None) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    shutil.copyfile(raw_db, out)
    now_naive = now.astimezone(tz=__import__("datetime").timezone.utc).replace(tzinfo=None)
    con = duckdb.connect(str(out))
    try:
        for t in _raw_tables(con):
            if "pulled_at" in {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}:
                con.execute(f"DELETE FROM {t} WHERE pulled_at > ?", [now_naive])
        _add_weekly_charts(con, now_naive, target_season, target_week)
    finally:
        con.close()
    return out


def _add_weekly_charts(con, now_naive: datetime, target_season, target_week) -> None:
    sched = con.execute("SELECT DISTINCT ON (game_id) game_id, season, week, gameday, home_team, away_team FROM schedules "
                        f"WHERE game_type = 'REG' AND season >= {FIRST_WEEKLY_SEASON} ORDER BY game_id, pulled_at DESC").pl()
    if sched.is_empty():
        return
    sched = sched.with_columns(gameday=pl.col("gameday").str.to_date())
    games = pl.concat([sched.select("season", "week", "gameday", team="home_team"), sched.select("season", "week", "gameday", team="away_team")]).unique()
    snaps = con.execute("SELECT DISTINCT dt, team, pos_abb, pos_rank, gsis_id FROM depth_charts WHERE season IS NULL AND dt IS NOT NULL AND pos_grp = ? AND gsis_id IS NOT NULL",
                        [OFFENSE_GROUP]).pl()
    snaps = snaps.filter(pl.col("pos_abb").is_in(list(LEGACY_POSITION))).with_columns(
        snap=pl.col("dt").str.to_datetime(time_zone="UTC", strict=False).dt.replace_time_zone(None)).drop_nulls("snap")
    if snaps.is_empty():
        return
    index = snaps.select("team", "snap").unique().sort("snap")
    g = games.with_columns(game_start=pl.col("gameday").map_elements(lambda d: datetime.combine(d, time()), return_dtype=pl.Datetime("us")))
    if target_season is not None:
        is_target = (pl.col("season") == target_season) & (pl.col("week") == target_week)
        g = g.with_columns(game_start=pl.when(is_target).then(pl.lit(now_naive).cast(pl.Datetime("us"))).otherwise(pl.col("game_start")))
        g = g.filter((pl.col("season") < target_season) | ((pl.col("season") == target_season) & (pl.col("week") <= target_week)))
    picked = g.sort("game_start").join_asof(index, left_on="game_start", right_on="snap", by="team", strategy="backward", allow_exact_matches=False, check_sortedness=False).drop_nulls("snap")
    rows = picked.join(snaps.select("team", "snap", "gsis_id", "pos_abb", "pos_rank"), on=["team", "snap"], how="inner").select(
        "season", "week", club_code="team", game_type=pl.lit("REG"), depth_team=pl.col("pos_rank").cast(pl.String), formation=pl.lit("Offense"), gsis_id="gsis_id",
        position=pl.col("pos_abb").replace_strict(LEGACY_POSITION), depth_position=pl.col("pos_abb").replace_strict(LEGACY_POSITION), pulled_at=pl.lit(now_naive).cast(pl.Datetime("us")))
    if rows.is_empty():
        return
    cols = {r[0]: r[1] for r in con.execute("DESCRIBE depth_charts").fetchall()}
    for c, t in cols.items():
        if c not in rows.columns:
            rows = rows.with_columns(pl.lit(None).alias(c))
    rows = rows.select(list(cols))
    con.register("derived_charts", rows.to_arrow())
    con.execute("INSERT INTO depth_charts SELECT * FROM derived_charts")
    con.unregister("derived_charts")
