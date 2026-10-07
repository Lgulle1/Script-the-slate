from datetime import datetime, timedelta, timezone

import duckdb
import polars as pl
import pytest

import config
from ingest import pull_weather as pw

KICK = datetime(2024, 10, 6, 17, 0, tzinfo=timezone.utc)  # 1 pm ET


def obs(rows):
    """rows: (station, distance_m, hour_offset_from_kickoff, temp_c, wspd_kmh, prcp_mm)."""
    k = KICK.replace(tzinfo=None)
    return pl.DataFrame({
        "station": [r[0] for r in rows], "distance_m": [float(r[1]) for r in rows],
        "time": [k + timedelta(hours=r[2]) for r in rows],
        "temp_c": [r[3] for r in rows], "wspd_kmh": [r[4] for r in rows], "prcp_mm": [r[5] for r in rows],
    }, schema_overrides={"temp_c": pl.Float64, "wspd_kmh": pl.Float64, "prcp_mm": pl.Float64, "time": pl.Datetime})


def patch(monkeypatch, frame):
    calls = []

    def fake(lat, lon, start, end):
        calls.append((lat, lon, start, end))
        if isinstance(frame, Exception):
            raise frame
        return frame

    monkeypatch.setattr(pw, "_load_station_hours", fake)
    return calls


def test_units_and_source(monkeypatch):
    patch(monkeypatch, obs([("A", 5000, 0, 10.0, 16.0, 2.54)]))
    r = pw.fetch_weather("g", 40, -74, KICK)
    assert r["source"] == "meteostat_observed"
    assert r["temp_f"] == pytest.approx(50.0)                 # 10 C
    assert r["wind_mph"] == pytest.approx(16 * 0.621371)      # km/h -> mph
    assert r["precip_in"] == pytest.approx(0.1)               # mm -> in
    assert r["precip_prob_pct"] is None
    assert (r["station_id"], r["station_km"]) == ("A", 5.0)


def test_prefers_exact_hour_then_nearest_station(monkeypatch):
    patch(monkeypatch, obs([("far_exact", 9000, 0, 1.0, 10.0, 0.0), ("near_exact", 3000, 0, 2.0, 10.0, 0.0),
                            ("near_offhour", 1000, 1, 3.0, 10.0, 0.0)]))
    assert pw.fetch_weather("g", 40, -74, KICK)["station_id"] == "near_exact"


def test_skips_station_missing_wind_or_temp_and_falls_to_next(monkeypatch):
    patch(monkeypatch, obs([("nowind", 1000, 0, 5.0, None, 0.0), ("ok", 8000, 0, 6.0, 12.0, None)]))
    r = pw.fetch_weather("g", 40, -74, KICK)
    assert r["station_id"] == "ok" and r["precip_in"] is None  # unreported precip stays null, not 0


def test_gap_beyond_one_hour_is_missing(monkeypatch):
    patch(monkeypatch, obs([("A", 1000, 3, 5.0, 12.0, 0.0)]))
    assert pw.fetch_weather("g", 40, -74, KICK)["source"] == "missing"
    patch(monkeypatch, obs([("A", 1000, -1, 5.0, 12.0, 0.0)]))
    assert pw.fetch_weather("g", 40, -74, KICK)["source"] == "meteostat_observed"


def test_failure_never_raises_and_yields_missing(monkeypatch, tmp_path):
    patch(monkeypatch, RuntimeError("no network"))
    df = pw.pull_weather([{"game_id": "g", "lat": 1, "lon": 2, "kickoff": KICK},
                          {"game_id": "h", "lat": None, "lon": None, "kickoff": KICK}], tmp_path / "w.duckdb")
    assert df["source"].to_list() == ["missing", "missing"]
    assert duckdb.connect(str(tmp_path / "w.duckdb")).execute("SELECT count(*) FROM weather").fetchone()[0] == 2


def test_one_load_per_venue_per_year_and_input_order_kept(monkeypatch):
    calls = patch(monkeypatch, obs([("A", 1000, 0, 5.0, 12.0, 0.0)]))
    rows = [{"game_id": f"g{i}", "lat": 40.0, "lon": -74.0, "kickoff": KICK} for i in range(3)]
    rows.insert(1, {"game_id": "other", "lat": 41.0, "lon": -75.0, "kickoff": KICK})
    out = pw.pull_weather_rows(rows)
    assert [r["game_id"] for r in out] == ["g0", "other", "g1", "g2"]
    assert len(calls) == 2
    # a venue's games in different years are loaded separately (Meteostat caps requests at 3 years)
    calls.clear()
    pw.pull_weather_rows([{"game_id": "a", "lat": 40.0, "lon": -74.0, "kickoff": KICK},
                          {"game_id": "b", "lat": 40.0, "lon": -74.0, "kickoff": KICK.replace(year=2021)}])
    assert len(calls) == 2


def test_resolve_location():
    assert pw.resolve_location("NYG", "NYC01", "Home") == config.stadium_coordinates["New York Giants"][:2]
    assert pw.resolve_location("NE", "LON00", "Neutral") == config.venue_coordinates["LON00"]
    assert pw.resolve_location("NE", "UNKNOWN", "Neutral") is None  # never the home stadium
    assert pw.resolve_location("KC", None, "Home") == config.stadium_coordinates["Kansas City Chiefs"][:2]


def test_kickoff_utc_converts_eastern():
    assert pw.kickoff_utc("2024-09-08", "13:00") == datetime(2024, 9, 8, 17, 0, tzinfo=timezone.utc)
    assert pw.kickoff_utc("2024-09-08", None) is None


def test_dotenv_loaded_by_config(tmp_path):
    import os, shutil, subprocess, sys
    repo = tmp_path / "r"
    repo.mkdir()
    shutil.copy(config.ROOT / "config.py", repo / "config.py")
    (repo / ".env").write_text("SOME_SETTING=from-dotenv\n")
    env = {k: v for k, v in os.environ.items() if k != "SOME_SETTING"}
    out = subprocess.run([sys.executable, "-c", "import config, os; print(os.environ.get('SOME_SETTING'))"],
                         cwd=repo, capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "from-dotenv", out.stderr


@pytest.mark.network
def test_live_meteostat_smoke():
    r = pw.fetch_weather("live", 40.8128, -74.0742, datetime(2022, 10, 9, 17, 0, tzinfo=timezone.utc))
    if r["source"] == "missing":
        pytest.skip("Meteostat unreachable")
    assert r["source"] == "meteostat_observed" and 30 < r["temp_f"] < 80 and r["wind_mph"] >= 0


# ---- observed vs forecast, indoor, observed-only loading, migration (3.5.2 item 5)
def test_kickoff_before_the_pull_is_observed_and_after_it_is_a_forecast(monkeypatch):
    patch(monkeypatch, obs([("A", 1000, 0, 5.0, 12.0, 0.0)]))
    past = pw.fetch_weather("g", 40, -74, KICK, now=KICK + timedelta(hours=5))
    assert past["source"] == "meteostat_observed" and past["hours_before_kickoff"] is None
    ahead = pw.fetch_weather("g", 40, -74, KICK, now=KICK - timedelta(hours=30))
    assert ahead["source"] == "meteostat_forecast" and ahead["hours_before_kickoff"] == pytest.approx(30.0)
    # a missing lookup for a future game still records how far ahead it was asked
    patch(monkeypatch, RuntimeError("down"))
    gone = pw.fetch_weather("g", 40, -74, KICK, now=KICK - timedelta(hours=2))
    assert gone["source"] == "missing" and gone["hours_before_kickoff"] == pytest.approx(2.0)


def test_forecast_rows_store_pulled_at_and_are_kept_alongside_the_later_observation(monkeypatch, tmp_path):
    db = tmp_path / "w.duckdb"
    patch(monkeypatch, obs([("A", 1000, 0, 5.0, 12.0, 0.0)]))
    wed, mon = KICK - timedelta(days=4), KICK + timedelta(days=1)
    row = [{"game_id": "g", "lat": 40.0, "lon": -74.0, "kickoff": KICK, "indoor": False}]
    pw.pull_weather(row, db, now=wed)
    pw.pull_weather(row, db, now=mon)
    con = duckdb.connect(str(db))
    got = con.execute("SELECT source, pulled_at, hours_before_kickoff, indoor FROM weather ORDER BY pulled_at").fetchall()
    assert [g[0] for g in got] == ["meteostat_forecast", "meteostat_observed"]       # nothing overwritten
    assert got[0][1] == wed.replace(tzinfo=None) and got[0][2] == pytest.approx(96.0) and got[1][2] is None
    assert con.execute("SELECT count(*) FROM weather_observed").fetchone()[0] == 1
    con.close()
    only_obs = pw.load_weather(db)
    assert only_obs["source"].to_list() == ["meteostat_observed"]                      # the default excludes forecasts
    assert pw.load_weather(db, observed_only=False)["source"].to_list() == ["meteostat_observed"]  # latest of any source


def test_observed_only_loader_excludes_forecast_rows(monkeypatch, tmp_path):
    db = tmp_path / "w.duckdb"
    patch(monkeypatch, obs([("A", 1000, 0, 5.0, 12.0, 0.0)]))
    pw.pull_weather([{"game_id": "g", "lat": 40.0, "lon": -74.0, "kickoff": KICK}], db, now=KICK - timedelta(days=2))
    assert pw.load_weather(db).height == 0                                             # forecast only -> no observed rows
    assert pw.load_weather(db, observed_only=False)["source"].to_list() == ["meteostat_forecast"]


def test_indoor_flag_rules():
    f = pw.indoor_flag
    assert f("DET", "DET00", "dome") is True                  # fixed dome
    assert f("DET", None, "outdoors") is True                 # fixed dome wins over a stray schedules value
    assert f("DAL", None, "closed") is True                   # retractable, closed
    assert f("DAL", None, "open") is False                    # retractable, open
    assert f("DAL", None, None) is None                       # retractable, state unknown
    assert f("BUF", None, "outdoors") is False and f("BUF", None, None) is False
    assert f("NE", "LON00", "outdoors") is False              # international, roof from schedules
    assert f("NE", "MEL00", "dome") is True and f("NE", "MAD01", None) is None
    assert f("NYG", "LAX01", "dome") is True                  # relocated game: the venue's roof, not the home team's


def test_build_requests_carries_indoor(monkeypatch):
    rows = [dict(game_id="a", gameday="2024-09-08", gametime="13:00", home_team="DET", stadium_id="DET00", location="Home", roof="dome"),
            dict(game_id="b", gameday="2024-09-08", gametime="13:00", home_team="BUF", stadium_id="BUF00", location="Home", roof="outdoors")]
    assert [r["indoor"] for r in pw.build_requests(rows)] == [True, False]


def test_migration_retags_legacy_rows_and_is_idempotent(tmp_path):
    db = tmp_path / "raw.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE schedules AS SELECT * FROM (VALUES ('past','2024-09-08','13:00','DET','DET00','dome', TIMESTAMP '2024-09-01'),"
                " ('ahead','2024-09-15','13:00','DAL','DAL00','closed', TIMESTAMP '2024-09-01')) t(game_id, gameday, gametime, home_team, stadium_id, roof, pulled_at)")
    con.execute("CREATE TABLE weather AS SELECT * FROM (VALUES ('past', 50.0, 'meteostat', TIMESTAMP '2024-09-10'),"
                " ('ahead', 60.0, 'meteostat', TIMESTAMP '2024-09-10'), ('gone', NULL, 'missing', TIMESTAMP '2024-09-10')) t(game_id, temp_f, source, pulled_at)")
    con.close()
    counts = pw.migrate_weather(db)
    assert counts == {"meteostat_forecast": 1, "meteostat_observed": 1, "missing": 1}
    con = duckdb.connect(str(db))
    got = {r[0]: r[1:] for r in con.execute("SELECT game_id, source, indoor, hours_before_kickoff, temp_f FROM weather").fetchall()}
    assert got["past"][0] == "meteostat_observed" and got["past"][1] is True and got["past"][2] is None
    assert got["ahead"][0] == "meteostat_forecast" and got["ahead"][1] is True and got["ahead"][2] == pytest.approx(137.0)
    assert got["past"][3] == 50.0 and got["ahead"][3] == 60.0                         # no weather value changed
    con.close()
    assert pw.migrate_weather(db) == {}                                                # idempotent
