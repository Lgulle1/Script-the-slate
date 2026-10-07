"""Pull nflverse data into DuckDB."""
from datetime import datetime, timezone

import duckdb
import nflreadpy
import polars as pl

import config

PLAYERS_TABLE = "players"


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


def _pull(table: str, loader, seasons, db_path) -> int:
    seasons = list(seasons) if seasons is not None else pull_seasons()
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


RAW_PULLS = {
    "schedules": pull_schedules,
    "pbp": pull_pbp,
    "player_stats": pull_player_stats,
    "snap_counts": pull_snap_counts,
    "rosters_weekly": pull_rosters_weekly,
}


def pull_all_raw(seasons=None, db_path=config.RAW_DUCKDB_PATH) -> dict[str, int]:
    """Run every raw pull; returns rows appended per table."""
    return {name: fn(seasons, db_path) for name, fn in RAW_PULLS.items()}


if __name__ == "__main__":
    print(f"wrote {pull_players()} rows to {PLAYERS_TABLE}")
    print(pull_all_raw())
