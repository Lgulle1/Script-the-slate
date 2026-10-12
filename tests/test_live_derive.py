"""The as-of working copy of the raw database for a live run (live/derive.py): plan P5.2 / P5.7."""
from datetime import datetime, timezone

import duckdb
import polars as pl
import pytest

from live import derive as DV

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


def _raw(path):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE schedules (game_id VARCHAR, season INTEGER, week INTEGER, game_type VARCHAR, gameday VARCHAR, home_team VARCHAR, away_team VARCHAR, pulled_at TIMESTAMP)")
    con.execute("INSERT INTO schedules VALUES ('2026_05_AAA_BBB',2026,5,'REG','2026-10-11','BBB','AAA','2026-10-07 12:00:00'),"
                "('2026_06_AAA_CCC',2026,6,'REG','2026-10-18','CCC','AAA','2026-10-07 12:00:00'),"
                "('2026_06_AAA_CCC',2026,6,'REG','2026-10-18','CCC','AAA','2026-10-20 12:00:00')")           # a later pull: after the clock
    con.execute("CREATE TABLE depth_charts (season INTEGER, club_code VARCHAR, week INTEGER, game_type VARCHAR, depth_team VARCHAR, formation VARCHAR, gsis_id VARCHAR, position VARCHAR, "
                "depth_position VARCHAR, dt VARCHAR, team VARCHAR, pos_abb VARCHAR, pos_rank INTEGER, pos_grp VARCHAR, pulled_at TIMESTAMP)")
    rows = []
    for dt, qb in (("2026-10-03T06:00:00Z", "qbA"), ("2026-10-10T06:00:00Z", "qbB"), ("2026-10-13T06:00:00Z", "qbC")):
        for team in ("AAA", "BBB", "CCC"):
            rows.append((None, None, None, None, None, None, f"{qb}_{team}", None, None, dt, team, "QB", 1, "3WR 1TE", "2026-10-07 12:00:00"))
            rows.append((None, None, None, None, None, None, f"lt_{team}", None, None, dt, team, "LT", 1, "3WR 1TE", "2026-10-07 12:00:00"))
            rows.append((None, None, None, None, None, None, f"de_{team}", None, None, dt, team, "LDE", 1, "Base 4-3 D", "2026-10-07 12:00:00"))
    rows.append((2024, "AAA", 5, "REG", "1", "Offense", "old_qb", "QB", "QB", None, None, None, None, None, "2026-10-07 12:00:00"))         # a weekly row of the old layout stays as it is
    rows.append((None, None, None, None, None, None, "late", None, None, "2026-10-13T06:00:00Z", "AAA", "RB", 1, "3WR 1TE", "2026-10-30 00:00:00"))   # pulled after the clock
    for r in rows:
        con.execute("INSERT INTO depth_charts VALUES (" + ",".join("?" * 15) + ")", list(r))
    con.close()


def test_rows_pulled_after_the_clock_are_not_in_the_copy_and_the_raw_file_is_untouched(tmp_path):
    raw, out = tmp_path / "raw.duckdb", tmp_path / "live.duckdb"
    _raw(raw)
    before = raw.read_bytes()
    DV.build_live_db(raw, out, NOW)
    assert raw.read_bytes() == before
    con = duckdb.connect(str(out), read_only=True)
    assert con.execute("SELECT count(*) FROM schedules").fetchone()[0] == 2                          # the 20 Oct pull is gone
    assert con.execute("SELECT count(*) FROM depth_charts WHERE gsis_id = 'late'").fetchone()[0] == 0


def test_weekly_charts_are_made_from_the_newest_snapshot_before_each_game_day_and_as_of_now_for_the_target_week(tmp_path):
    raw, out = tmp_path / "raw.duckdb", tmp_path / "live.duckdb"
    _raw(raw)
    DV.build_live_db(raw, out, NOW, target_season=2026, target_week=6)
    con = duckdb.connect(str(out), read_only=True)
    d = con.execute("SELECT season, week, club_code, gsis_id, position, depth_team, formation, game_type FROM depth_charts WHERE season = 2026 AND position = 'QB'").pl().sort("week", "club_code")
    got = {(r["week"], r["club_code"]): r["gsis_id"] for r in d.iter_rows(named=True)}
    assert got[(5, "AAA")] == "qbB_AAA"                       # week 5 game day 11 Oct: the newest chart strictly before that day is the one of 10 Oct 06:00
    assert got[(6, "AAA")] == "qbC_AAA"                       # the target week takes the chart as of the clock (13 Oct 06:00), not "before the game day" only
    assert set(d["formation"]) == {"Offense"} and set(d["game_type"]) == {"REG"}
    t = con.execute("SELECT position FROM depth_charts WHERE season = 2026 AND gsis_id = 'lt_AAA'").fetchall()
    assert {x[0] for x in t} == {"T"}                         # LT -> T, as the old layout names it
    assert con.execute("SELECT count(*) FROM depth_charts WHERE gsis_id LIKE 'de_%' AND season IS NOT NULL").fetchone()[0] == 0     # no defense in the offensive weekly chart
    assert con.execute("SELECT count(*) FROM depth_charts WHERE gsis_id = 'old_qb'").fetchone()[0] == 1


def test_a_game_with_no_earlier_chart_gets_no_weekly_rows(tmp_path):
    raw, out = tmp_path / "raw.duckdb", tmp_path / "live.duckdb"
    _raw(raw)
    DV.build_live_db(raw, out, NOW)
    con = duckdb.connect(str(out), read_only=True)
    assert con.execute("SELECT count(*) FROM depth_charts WHERE season = 2026 AND week = 1").fetchone()[0] == 0
