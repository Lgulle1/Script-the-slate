"""Pull game-time weather: Open-Meteo first, NWS fallback, else 'missing'.

Input rows are dicts: {game_id, lat, lon, kickoff} with kickoff a datetime
(naive = UTC). Output rows carry temp_f, wind_mph, precip_in, precip_prob_pct
and a source tag in {"open-meteo", "nws", "missing"}. A failed lookup never
raises: it yields a row with source "missing" and null values.

Roof state is NOT handled here -- weather is pulled for every game location
and the schedules table's own roof column decides whether it matters.
"""
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import requests

import config
from ingest.pull_nflverse import _append_raw

log = logging.getLogger(__name__)

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NWS_POINTS = "https://api.weather.gov/points/{lat},{lon}"
TIMEOUT_S = 15
# The forecast endpoint only serves a limited window of past days; older
# kickoffs go to Open-Meteo's archive endpoint (same provider, same tag).
FORECAST_PAST_DAYS_LIMIT = 90

SOURCE_OPEN_METEO, SOURCE_NWS, SOURCE_MISSING = "open-meteo", "nws", "missing"
WEATHER_TABLE = "weather"
_HOURLY = "temperature_2m,wind_speed_10m,precipitation,precipitation_probability"


RETRIES_ON_429 = 3
REQUEST_PAUSE_S = 0.25  # polite spacing between games in a batch


def _get(url, **kw):
    """requests.get with exponential backoff on HTTP 429."""
    for attempt in range(RETRIES_ON_429 + 1):
        resp = requests.get(url, timeout=TIMEOUT_S, **kw)
        if resp.status_code != 429 or attempt == RETRIES_ON_429:
            return resp
        time.sleep(2 ** (attempt + 1))
    return resp


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _missing(game_id) -> dict:
    return dict(game_id=game_id, temp_f=None, wind_mph=None, precip_in=None,
                precip_prob_pct=None, source=SOURCE_MISSING, endpoint=None)


def _open_meteo(lat, lon, kickoff: datetime) -> dict:
    kickoff = _utc(kickoff)
    age = datetime.now(timezone.utc) - kickoff
    archive = age > timedelta(days=FORECAST_PAST_DAYS_LIMIT)
    day = kickoff.date().isoformat()
    params = dict(latitude=lat, longitude=lon, hourly=_HOURLY if not archive else "temperature_2m,wind_speed_10m,precipitation",
                  temperature_unit="fahrenheit", wind_speed_unit="mph", precipitation_unit="inch",
                  timezone="UTC", start_date=day, end_date=day)
    url = OPEN_METEO_ARCHIVE if archive else OPEN_METEO_FORECAST
    resp = _get(url, params=params)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    i = hourly["time"].index(kickoff.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00"))
    temp, wind = hourly["temperature_2m"][i], hourly["wind_speed_10m"][i]
    if temp is None or wind is None:
        raise ValueError("open-meteo returned null values for kickoff hour")
    return dict(temp_f=temp, wind_mph=wind, precip_in=hourly["precipitation"][i],
                precip_prob_pct=(hourly.get("precipitation_probability") or [None] * (i + 1))[i],
                source=SOURCE_OPEN_METEO, endpoint="archive" if archive else "forecast")


def _nws(lat, lon, kickoff: datetime) -> dict:
    contact = os.environ.get(config.NWS_CONTACT_ENV)
    if not contact:
        raise RuntimeError(f"{config.NWS_CONTACT_ENV} not set; NWS fallback skipped")
    headers = {"User-Agent": f"script-the-slate ({contact})", "Accept": "application/geo+json"}
    pts = _get(NWS_POINTS.format(lat=round(lat, 4), lon=round(lon, 4)), headers=headers)
    pts.raise_for_status()
    hourly_url = pts.json()["properties"]["forecastHourly"]
    resp = _get(hourly_url, headers=headers)
    resp.raise_for_status()
    kickoff = _utc(kickoff)
    for p in resp.json()["properties"]["periods"]:
        start = datetime.fromisoformat(p["startTime"]).astimezone(timezone.utc)
        end = datetime.fromisoformat(p["endTime"]).astimezone(timezone.utc)
        if start <= kickoff < end:
            temp = p["temperature"]
            if p.get("temperatureUnit", "F") == "C":
                temp = temp * 9 / 5 + 32
            speeds = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", p["windSpeed"])]
            if not speeds:
                raise ValueError(f"unparseable NWS wind: {p['windSpeed']!r}")
            prob = (p.get("probabilityOfPrecipitation") or {}).get("value")
            return dict(temp_f=float(temp), wind_mph=max(speeds), precip_in=None,
                        precip_prob_pct=prob, source=SOURCE_NWS, endpoint="forecastHourly")
    raise ValueError("kickoff outside NWS forecast window")


def fetch_weather(game_id, lat, lon, kickoff: datetime) -> dict:
    """Open-Meteo, then NWS, then a 'missing' row. Never raises."""
    if lat is None or lon is None or kickoff is None:
        return _missing(game_id)
    for fetch in (_open_meteo, _nws):
        try:
            return {"game_id": game_id, **fetch(lat, lon, kickoff)}
        except Exception as e:
            log.warning("weather %s: %s failed (%s)", game_id, fetch.__name__.strip("_"), e)
    return _missing(game_id)


def pull_weather(rows, db_path=config.RAW_DUCKDB_PATH) -> pl.DataFrame:
    """rows: iterable of dicts with game_id, lat, lon, kickoff. Appends to the weather table."""
    out = []
    for r in rows:
        out.append(fetch_weather(r["game_id"], r.get("lat"), r.get("lon"), r.get("kickoff")))
        time.sleep(REQUEST_PAUSE_S)
    df = pl.DataFrame(
        out,
        schema={"game_id": pl.String, "temp_f": pl.Float64, "wind_mph": pl.Float64,
                "precip_in": pl.Float64, "precip_prob_pct": pl.Float64,
                "source": pl.String, "endpoint": pl.String},
    )
    if df.height:
        _append_raw(WEATHER_TABLE, df, db_path)
    return df


def resolve_location(home_team_abbr: str, stadium_id: str | None, location: str | None):
    """(lat, lon) for a game, or None if it can't be located.

    Venue coordinates (by stadium_id) win; a home-team lookup is used only for
    non-neutral games. A neutral game at an unknown venue returns None -- it is
    never silently placed at the home team's stadium.
    """
    if stadium_id in config.venue_coordinates:
        return config.venue_coordinates[stadium_id]
    if location == "Neutral":
        return None
    team = config.TEAM_ABBR_TO_NAME.get(home_team_abbr)
    entry = config.stadium_coordinates.get(team)
    return entry[:2] if entry else None


def kickoff_utc(gameday: str, gametime: str | None) -> datetime | None:
    """nflverse gameday/gametime are US Eastern local; returns aware UTC."""
    if not gameday or not gametime:
        return None
    local = datetime.fromisoformat(f"{gameday}T{gametime}").replace(tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def build_requests(schedule_rows) -> list[dict]:
    """schedule_rows: dicts with game_id, gameday, gametime, home_team, stadium_id, location."""
    out = []
    for r in schedule_rows:
        loc = resolve_location(r["home_team"], r.get("stadium_id"), r.get("location"))
        if loc is None:
            log.warning("weather: no coordinates for %s (stadium_id=%s)", r["game_id"], r.get("stadium_id"))
        out.append(dict(game_id=r["game_id"], lat=loc[0] if loc else None, lon=loc[1] if loc else None,
                        kickoff=kickoff_utc(r["gameday"], r.get("gametime"))))
    return out
