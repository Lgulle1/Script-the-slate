from datetime import datetime, timedelta, timezone

import duckdb
import pytest
import requests

import config
from ingest import pull_weather as pw

KICK = datetime.now(timezone.utc) + timedelta(days=2)


class FakeResp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload


def om_payload(k):
    t = k.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
    return {"hourly": {"time": [t], "temperature_2m": [41.0], "wind_speed_10m": [12.0],
                       "precipitation": [0.1], "precipitation_probability": [30]}}


def test_open_meteo_primary(monkeypatch):
    monkeypatch.setattr(pw.requests, "get", lambda url, **kw: FakeResp(om_payload(KICK)))
    r = pw.fetch_weather("g", 40, -74, KICK)
    assert (r["source"], r["temp_f"], r["wind_mph"], r["precip_in"]) == ("open-meteo", 41.0, 12.0, 0.1)


def test_nws_fallback_when_open_meteo_fails(monkeypatch):
    monkeypatch.setenv(config.NWS_CONTACT_ENV, "me@example.com")
    start, end = KICK - timedelta(minutes=30), KICK + timedelta(minutes=30)

    def get(url, **kw):
        if "open-meteo" in url:
            return FakeResp({}, 500)
        assert "me@example.com" in kw["headers"]["User-Agent"]
        if "points" in url:
            return FakeResp({"properties": {"forecastHourly": "https://api.weather.gov/h"}})
        return FakeResp({"properties": {"periods": [{
            "startTime": start.isoformat(), "endTime": end.isoformat(), "temperature": 55,
            "temperatureUnit": "F", "windSpeed": "5 to 15 mph",
            "probabilityOfPrecipitation": {"value": 20}}]}})

    monkeypatch.setattr(pw.requests, "get", get)
    r = pw.fetch_weather("g", 40, -74, KICK)
    assert (r["source"], r["temp_f"], r["wind_mph"], r["precip_prob_pct"]) == ("nws", 55.0, 15.0, 20)


def test_both_fail_yields_missing_not_error(monkeypatch, tmp_path):
    monkeypatch.delenv(config.NWS_CONTACT_ENV, raising=False)
    monkeypatch.setattr(pw.requests, "get", lambda url, **kw: (_ for _ in ()).throw(requests.ConnectionError("x")))
    monkeypatch.setattr(pw, "REQUEST_PAUSE_S", 0)
    df = pw.pull_weather([{"game_id": "g", "lat": 1, "lon": 2, "kickoff": KICK},
                          {"game_id": "h", "lat": None, "lon": None, "kickoff": KICK}], tmp_path / "w.duckdb")
    assert df["source"].to_list() == ["missing", "missing"]
    assert duckdb.connect(str(tmp_path / "w.duckdb")).execute("SELECT count(*) FROM weather").fetchone()[0] == 2


def test_retry_on_429(monkeypatch):
    calls = []

    def get(url, **kw):
        calls.append(1)
        return FakeResp({}, 429) if len(calls) < 3 else FakeResp(om_payload(KICK))

    monkeypatch.setattr(pw.requests, "get", get)
    monkeypatch.setattr(pw.time, "sleep", lambda s: None)
    assert pw.fetch_weather("g", 1, 2, KICK)["source"] == "open-meteo"


def test_resolve_location():
    assert pw.resolve_location("NYG", "NYC01", "Home") == config.stadium_coordinates["New York Giants"][:2]
    assert pw.resolve_location("NE", "LON00", "Neutral") == config.venue_coordinates["LON00"]
    assert pw.resolve_location("NE", "UNKNOWN", "Neutral") is None  # never the home stadium
    assert pw.resolve_location("KC", None, "Home") == config.stadium_coordinates["Kansas City Chiefs"][:2]


def test_kickoff_utc_converts_eastern():
    assert pw.kickoff_utc("2024-09-08", "13:00") == datetime(2024, 9, 8, 17, 0, tzinfo=timezone.utc)
    assert pw.kickoff_utc("2024-09-08", None) is None


def test_open_meteo_retries_once_on_timeout(monkeypatch):
    calls = []

    def get(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            raise requests.Timeout("slow")
        return FakeResp(om_payload(KICK))

    monkeypatch.setattr(pw.requests, "get", get)
    monkeypatch.setattr(pw.time, "sleep", lambda s: None)
    assert pw.fetch_weather("g", 1, 2, KICK)["source"] == "open-meteo"
    assert len(calls) == 2


def test_two_timeouts_fall_through_to_missing(monkeypatch):
    monkeypatch.delenv(config.NWS_CONTACT_ENV, raising=False)
    monkeypatch.setattr(pw.requests, "get", lambda url, **kw: (_ for _ in ()).throw(requests.Timeout("x")))
    monkeypatch.setattr(pw.time, "sleep", lambda s: None)
    assert pw.fetch_weather("g", 1, 2, KICK)["source"] == "missing"


def test_dotenv_loaded_by_config(tmp_path):
    import subprocess, sys, shutil
    repo = tmp_path / "r"
    repo.mkdir()
    shutil.copy(config.ROOT / "config.py", repo / "config.py")
    (repo / ".env").write_text("NWS_CONTACT=from-dotenv@example.com\n")
    out = subprocess.run([sys.executable, "-c", "import config, os; print(os.environ.get('NWS_CONTACT'))"],
                         cwd=repo, capture_output=True, text=True, env={k: v for k, v in __import__('os').environ.items() if k != "NWS_CONTACT"})
    assert out.stdout.strip() == "from-dotenv@example.com", out.stderr
