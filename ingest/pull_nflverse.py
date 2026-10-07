"""Pull nflverse data into DuckDB."""
import logging
from datetime import datetime, timezone
from functools import partial

import duckdb
import nflreadpy
import polars as pl

import config

PLAYERS_TABLE = "players"
PLAYERS_CURRENT = "players_current"  # view: latest pull per player; the only thing master-id joins read
log = logging.getLogger(__name__)

FTN_MIN_SEASON = 2022
PFR_MIN_SEASON = 2018
PARTICIPATION_SEASONS = range(2016, 2026)  # load_participation raises outside this


def pull_players(db_path=config.DUCKDB_PATH) -> int:
    """Pull load_players() and APPEND it to the master player table (every pull keeps its pulled_at).

    Every master-id join reads the `players_current` view (the latest pull per gsis_id), never the raw
    table and never around it. Returns the number of rows appended.
    """
    config.ensure_data_dirs()
    n = _append_raw(PLAYERS_TABLE, nflreadpy.load_players(), db_path)
    ensure_players_view(db_path)
    return n


def ensure_players_view(db_path=config.DUCKDB_PATH):
    """(Re)create players_current, adding pulled_at to a legacy table that predates append-only pulls.

    Legacy rows keep a NULL pulled_at and lose to any dated pull, so the view equals the old table until the
    first new pull arrives. Players without a gsis_id cannot be keyed and are excluded.
    """
    con = duckdb.connect(str(db_path))
    try:
        if "pulled_at" not in {r[0] for r in con.execute(f"DESCRIBE {PLAYERS_TABLE}").fetchall()}:
            con.execute(f"ALTER TABLE {PLAYERS_TABLE} ADD COLUMN pulled_at TIMESTAMP")
        con.execute(f"CREATE OR REPLACE VIEW {PLAYERS_CURRENT} AS SELECT * FROM {PLAYERS_TABLE} WHERE gsis_id IS NOT NULL "
                    "QUALIFY row_number() OVER (PARTITION BY gsis_id ORDER BY pulled_at DESC NULLS LAST) = 1")
    finally:
        con.close()


def pull_seasons() -> list[int]:
    """Every season from config.DATA_START_SEASON (2016) through the current season. This is what pulls cover;
    which of those seasons models and features READ is config.FEATURE_HISTORY_START."""
    return list(range(config.DATA_START_SEASON, nflreadpy.get_current_season() + 1))


def _append_raw(table: str, df: pl.DataFrame, db_path, pulled_at: datetime | None = None) -> int:
    """Append df (plus a pulled_at UTC column) to a raw table; never overwrites.

    New columns upstream are added to the table; columns missing from df are NULL.
    Re-pulling the same season appends a second copy, distinguished by pulled_at.
    """
    config.ensure_data_dirs()
    stamp = pulled_at if pulled_at is not None else datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC
    df = df.with_columns(pl.lit(stamp).alias("pulled_at"))
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


def pull_pre_feature_history(db_path=config.RAW_DUCKDB_PATH, tables=None) -> dict:
    """One-off backfill: every table with data for DATA_START_SEASON .. FEATURE_HISTORY_START-1 (2016-2019),
    appended with pulled_at into the same raw database. Data only: nothing reads these seasons until
    FEATURE_HISTORY_START moves. FTN (2022+) and PFR (2018+) clip themselves via _pull; participation already
    holds 2016-2025 so it is not re-pulled (it would only duplicate rows). Returns rows appended per table,
    or the exception text for a table that failed (the others still run).
    """
    seasons = list(range(config.DATA_START_SEASON, config.FEATURE_HISTORY_START))
    pulls = {**RAW_PULLS, **SNAPSHOT_PULLS, **{k: v for k, v in ADVANCED_PULLS.items() if k != "participation"}}
    out = {}
    for name, fn in pulls.items():
        if tables and name not in tables:
            continue
        try:
            out[name] = fn(seasons, db_path)
        except Exception as e:  # noqa: BLE001 - report per table, keep going
            out[name] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
    return out


if __name__ == "__main__":
    print(f"wrote {pull_players()} rows to {PLAYERS_TABLE}")
    print(pull_all_raw())
    for name, fn in {**SNAPSHOT_PULLS, **ADVANCED_PULLS}.items():
        print(name, fn())
