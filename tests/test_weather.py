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
    assert r["source"] == "meteostat"
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
    assert pw.fetch_weather("g", 40, -74, KICK)["source"] == "meteostat"


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
    assert 30 < r["temp_f"] < 80 and r["wind_mph"] >= 0
