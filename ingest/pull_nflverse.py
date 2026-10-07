"""Pull nflverse data into DuckDB."""
import logging
from datetime import datetime, timezone
from functools import partial

import duckdb
import nflreadpy
import polars as pl

import config

PLAYERS_TABLE = "players"
log = logging.getLogger(__name__)

FTN_MIN_SEASON = 2022
PFR_MIN_SEASON = 2018
PARTICIPATION_SEASONS = range(2016, 2026)  # load_participation raises outside this


def pull_players(db_path=config.DUCKDB_PATH) -> int:
    """Pull load_players() once and save it as the master player id table.

    Every script that joins player data must join through this table
    (``players``, keyed by ``gsis_id``), never around it. Replaces any
    existing copy. Returns the number of rows written.
    """
    config.ensure_data_dirs()
    players = nflreadpy.load_players()  # polars DataFrame
    con = duckdb.connect(str(db_path))
    try:
        con.register("players_df", players.to_arrow())
        con.execute(f"CREATE OR REPLACE TABLE {PLAYERS_TABLE} AS SELECT * FROM players_df")
        return con.execute(f"SELECT count(*) FROM {PLAYERS_TABLE}").fetchone()[0]
    finally:
        con.close()


def pull_seasons() -> list[int]:
    """Backtest seasons + locked holdout + the current season (deduplicated)."""
    seasons = {*config.BACKTEST_SEASONS, config.HOLDOUT_SEASON, nflreadpy.get_current_season()}
    return sorted(seasons)


def _append_raw(table: str, df: pl.DataFrame, db_path) -> int:
    """Append df (plus a pulled_at UTC column) to a raw table; never overwrites.

    New columns upstream are added to the table; columns missing from df are NULL.
    Re-pulling the same season appends a second copy, distinguished by pulled_at.
    """
    config.ensure_data_dirs()
    df = df.with_columns(pl.lit(datetime.now(timezone.utc).replace(tzinfo=None)).alias("pulled_at"))  # naive UTC
    con = duckdb.connect(str(db_path))
    try:
        con.register("incoming", df.to_arrow())
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()[0]
        if not exists:
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM incoming')
        else:
            have = {r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()}
            for name, typ, *_ in con.execute("DESCRIBE incoming").fetchall():
                if name not in have:
                    con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {typ}')
            con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM incoming')
        return df.height
    finally:
        con.close()


def _pull(table: str, loader, seasons, db_path, min_season=None) -> int:
    seasons = list(seasons) if seasons is not None else pull_seasons()
    if min_season is not None:
        seasons = [s for s in seasons if s >= min_season]
    if not seasons:
        return 0
    return _append_raw(table, loader(seasons), db_path)


def pull_schedules(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("schedules", nflreadpy.load_schedules, seasons, db_path)


def pull_pbp(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("pbp", nflreadpy.load_pbp, seasons, db_path)


def pull_player_stats(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("player_stats", nflreadpy.load_player_stats, seasons, db_path)


def pull_snap_counts(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("snap_counts", nflreadpy.load_snap_counts, seasons, db_path)


def pull_rosters_weekly(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("rosters_weekly", nflreadpy.load_rosters_weekly, seasons, db_path)


# --- 1.3 weekly-snapshot tables -------------------------------------------
# Re-pulled on a Wed/Fri/Sun cadence once live. Every pull is kept: never
# deduplicate or overwrite past pulls -- pulled_at is what lets us reconstruct
# what was known at a given moment.


def pull_injuries(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("injuries", nflreadpy.load_injuries, seasons, db_path)


def pull_depth_charts(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("depth_charts", nflreadpy.load_depth_charts, seasons, db_path)


# --- 1.4 advanced-stats tables ---------------------------------------------


def pull_ftn_charting(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("ftn_charting", nflreadpy.load_ftn_charting, seasons, db_path, FTN_MIN_SEASON)


def _nextgen(stat_type):
    def pull(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
        loader = partial(nflreadpy.load_nextgen_stats, stat_type=stat_type)
        return _pull(f"nextgen_stats_{stat_type}", loader, seasons, db_path)

    pull.__name__ = f"pull_nextgen_stats_{stat_type}"
    return pull


def _pfr(stat_type):
    def pull(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
        loader = partial(nflreadpy.load_pfr_advstats, stat_type=stat_type, summary_level="week")
        return _pull(f"pfr_advstats_{stat_type}", loader, seasons, db_path, PFR_MIN_SEASON)

    pull.__name__ = f"pull_pfr_advstats_{stat_type}"
    return pull


pull_nextgen_stats_rushing = _nextgen("rushing")
pull_nextgen_stats_receiving = _nextgen("receiving")
pull_nextgen_stats_passing = _nextgen("passing")
pull_pfr_advstats_rush = _pfr("rush")
pull_pfr_advstats_rec = _pfr("rec")
pull_pfr_advstats_pass = _pfr("pass")


def pull_ff_opportunity(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    """Weekly-level expected-fantasy-points (opportunity) table."""
    loader = partial(nflreadpy.load_ff_opportunity, stat_type="weekly")
    return _pull("ff_opportunity", loader, seasons, db_path)


def pull_officials(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    return _pull("officials", nflreadpy.load_officials, seasons, db_path)


# --- 1.5 participation: TRAINING LABELS ONLY --------------------------------
# Not available for the current season, so it must never be used as a
# pregame feature -- only as a training label (who actually played / formation
# personnel). Rows carry training_labels_only=True as a schema-level tag.


def pull_participation(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> int:
    """Pull load_participation for 2016-2025 only; unavailable seasons are skipped, not fatal."""
    seasons = list(seasons) if seasons is not None else list(PARTICIPATION_SEASONS)
    total = 0
    for season in seasons:
        if season not in PARTICIPATION_SEASONS:
            log.warning("participation: skipping %s (outside 2016-2025)", season)
            continue
        try:
            df = nflreadpy.load_participation([season])
        except Exception as e:
            log.warning("participation: skipping %s (%s)", season, e)
            continue
        df = df.with_columns(pl.lit(True).alias("training_labels_only"))
        total += _append_raw("participation", df, db_path)
    return total


RAW_PULLS = {
    "schedules": pull_schedules,
    "pbp": pull_pbp,
    "player_stats": pull_player_stats,
    "snap_counts": pull_snap_counts,
    "rosters_weekly": pull_rosters_weekly,
}


SNAPSHOT_PULLS = {"injuries": pull_injuries, "depth_charts": pull_depth_charts}

ADVANCED_PULLS = {
    "ftn_charting": pull_ftn_charting,
    "nextgen_stats_rushing": pull_nextgen_stats_rushing,
    "nextgen_stats_receiving": pull_nextgen_stats_receiving,
    "nextgen_stats_passing": pull_nextgen_stats_passing,
    "pfr_advstats_rush": pull_pfr_advstats_rush,
    "pfr_advstats_rec": pull_pfr_advstats_rec,
    "pfr_advstats_pass": pull_pfr_advstats_pass,
    "ff_opportunity": pull_ff_opportunity,
    "officials": pull_officials,
    "participation": pull_participation,
}


def pull_all_raw(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> dict[str, int]:
    """Run every raw pull; returns rows appended per table."""
    return {name: fn(seasons, db_path) for name, fn in RAW_PULLS.items()}


if __name__ == "__main__":
    print(f"wrote {pull_players()} rows to {PLAYERS_TABLE}")
    print(pull_all_raw())
    for name, fn in {**SNAPSHOT_PULLS, **ADVANCED_PULLS}.items():
        print(name, fn())
