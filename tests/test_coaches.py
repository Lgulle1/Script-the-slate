import duckdb
import pytest

from ingest import load_coaches as lc

CSV = """team,role,name,start_year_with_team,prior_stops,source_url,confidence
KC,HC,Andy Reid,2013,PHI,http://x,high
KC,OC,Someone,2023,"NYG;DEN",http://y,low
"""


def test_load_and_hand_corrections_survive_reload(tmp_path):
    csv, db = tmp_path / "c.csv", tmp_path / "d.duckdb"
    csv.write_text(CSV)
    assert lc.load_coaches(csv, db) == 2
    con = duckdb.connect(str(db))
    assert con.execute("SELECT count(*) FROM coaches WHERE hand_corrected").fetchone()[0] == 0
    con.close()
    lc.mark_hand_corrected("KC", "OC", "Someone", db, confidence="high")
    csv.write_text(CSV)  # re-import the same (stale) seed
    assert lc.load_coaches(csv, db) == 2
    con = duckdb.connect(str(db))
    assert con.execute("SELECT confidence, hand_corrected FROM coaches WHERE role='OC'").fetchone() == ("high", True)


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        lc.load_coaches(tmp_path / "nope.csv", tmp_path / "d.duckdb")
    bad = tmp_path / "b.csv"
    bad.write_text("team,role\nKC,HC\n")
    with pytest.raises(ValueError, match="missing columns"):
        lc.load_coaches(bad, tmp_path / "d.duckdb")


def test_start_year_notes_are_kept_and_parsed(tmp_path):
    csv, db = tmp_path / "c.csv", tmp_path / "d.duckdb"
    csv.write_text(CSV.replace("2013", "2013 (with team since 2010)").replace("2023", "unknown"))
    lc.load_coaches(csv, db)
    rows = duckdb.connect(str(db)).execute(
        "SELECT role, start_year_with_team, start_year_raw FROM coaches ORDER BY role").fetchall()
    assert rows == [("HC", 2013, "2013 (with team since 2010)"), ("OC", None, "unknown")]
