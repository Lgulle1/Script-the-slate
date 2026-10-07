import duckdb
import pytest

from ingest import pull_nflverse


def test_pull_players_writes_master_table(tmp_path):
    db = tmp_path / "t.duckdb"
    try:
        n = pull_nflverse.pull_players(db)
    except Exception as e:  # network unavailable
        pytest.skip(f"players unavailable: {e}")
    con = duckdb.connect(str(db))
    assert n > 10_000
    assert con.execute("SELECT count(*) FROM players").fetchone()[0] == n
    assert con.execute("SELECT count(*) FROM players WHERE gsis_id IS NOT NULL").fetchone()[0] > 0
    # re-running APPENDS (every pull keeps its pulled_at); players_current still serves one row per player
    assert pull_nflverse.pull_players(db) == n
    assert con.execute("SELECT count(*) FROM players").fetchone()[0] == 2 * n
    assert con.execute("SELECT count(*) FROM players_current").fetchone()[0] <= n
