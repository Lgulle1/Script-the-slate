"""Pull nflverse data into DuckDB."""
import duckdb
import nflreadpy

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


if __name__ == "__main__":
    print(f"wrote {pull_players()} rows to {PLAYERS_TABLE}")
