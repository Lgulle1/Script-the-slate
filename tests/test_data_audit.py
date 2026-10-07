import json

import duckdb

from ingest import data_audit as da


def test_audit_report(tmp_path):
    raw = tmp_path / "raw.duckdb"
    con = duckdb.connect(str(raw))
    con.execute("CREATE TABLE injuries AS SELECT * FROM (VALUES (2024, 1, 'a', TIMESTAMP '2024-09-01'),"
                " (2024, 3, NULL, TIMESTAMP '2024-09-01'), (2025, 2, 'b', TIMESTAMP '2025-09-01'))"
                " t(season, week, gsis_id, pulled_at)")
    con.execute("CREATE TABLE participation AS SELECT '2023_01_A_B' nflverse_game_id, TIMESTAMP '2024-01-01' pulled_at")
    con.close()
    out = tmp_path / "audit.json"
    rep = da.run_audit(raw, tmp_path / "missing.duckdb", out)
    inj = rep["tables"]["injuries"]
    assert inj["by_season"] == {"2024": {"rows": 2, "max_week": 3}, "2025": {"rows": 1, "max_week": 2}}
    assert inj["pct_filled"]["gsis_id"] == round(200 / 3, 2)
    assert "team" in inj["columns_not_found"]
    assert rep["tables"]["participation"]["by_season"] == {"2023": {"rows": 1}}
    assert json.loads(out.read_text())["tables"].keys() == rep["tables"].keys()


def test_important_columns_exist_in_live_raw_db():
    import pytest
    import config
    if not config.RAW_DUCKDB_PATH.exists():
        pytest.skip("no local raw db")
    rep = da.run_audit(out_path=config.DATA_DIR / "data_audit.json")
    missing = {t: i["columns_not_found"] for t, i in rep["tables"].items() if i.get("columns_not_found")}
    assert not missing, missing
