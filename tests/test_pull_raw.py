import duckdb
import polars as pl
import pytest

import config
from ingest import pull_nflverse as pn


def test_pull_seasons_includes_config_and_current():
    s = pn.pull_seasons()
    assert set(config.BACKTEST_SEASONS) <= set(s)
    assert config.HOLDOUT_SEASON in s
    assert s == sorted(set(s))


def test_append_never_overwrites_and_handles_new_columns(tmp_path):
    db = tmp_path / "r.duckdb"
    assert pn._append_raw("t", pl.DataFrame({"a": [1, 2]}), db) == 2
    assert pn._append_raw("t", pl.DataFrame({"a": [3], "b": ["x"]}), db) == 1
    con = duckdb.connect(str(db))
    assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM t WHERE pulled_at IS NOT NULL").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM t WHERE b IS NULL").fetchone()[0] == 2


def test_live_small_pull_appends(tmp_path):
    db = tmp_path / "r.duckdb"
    try:
        n1 = pn.pull_schedules([2024], db)
    except Exception as e:  # network unavailable
        pytest.skip(f"schedules unavailable: {e}")
    n2 = pn.pull_schedules([2024], db)
    con = duckdb.connect(str(db))
    assert con.execute("SELECT count(*) FROM schedules").fetchone()[0] == n1 + n2
