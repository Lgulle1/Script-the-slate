"""Pull game-time weather from Meteostat (hourly station observations).

Input rows are dicts: {game_id, lat, lon, kickoff} with kickoff a datetime
(naive = UTC). Output rows carry temp_f, wind_mph, precip_in, a null
precip_prob_pct (Meteostat has no forecast probability) and source in
{"meteostat", "missing"}. A failed lookup never raises: it yields a row with
source "missing" and null values.

These are OBSERVED conditions at the nearest reporting stations, not a pregame
forecast -- fine for training/backtests, but not what a bettor would have seen.
Weather is also pulled for domed venues, where it is irrelevant to game
conditions; roof state comes from the schedules table's own roof column, and the
features layer must account for that so the model doesn't learn from indoor "weather".
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import meteostat as ms
import polars as pl

import config
from ingest.pull_nflverse import _append_raw

log = logging.getLogger(__name__)

SOURCE_METEOSTAT, SOURCE_MISSING = "meteostat", "missing"
WEATHER_TABLE = "weather"
N_STATIONS = 4               # nearest stations considered per venue
MAX_HOUR_GAP = 1             # accept an observation up to this many hours from kickoff
KMH_TO_MPH = 0.621371
MM_TO_IN = 1 / 25.4


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _naive_utc(dt: datetime) -> datetime:
    return _utc(dt).replace(tzinfo=None)


def _missing(game_id) -> dict:
    return dict(game_id=game_id, temp_f=None, wind_mph=None, precip_in=None, precip_prob_pct=None,
                source=SOURCE_MISSING, station_id=None, station_km=None)


def _load_station_hours(lat, lon, start: datetime, end: datetime) -> pl.DataFrame:
    """Hourly observations (UTC) from the nearest stations over [start, end].

    Columns: station, distance_m, time, temp_c, wspd_kmh, prcp_mm. This is the only
    function that touches Meteostat (data comes from its bulk files, not a metered API).
    """
    point = ms.Point(lat, lon)
    stations = ms.stations.nearby(point, limit=N_STATIONS)
    df = ms.hourly(stations, start, end).fetch()
    if df is None or len(df) == 0:
        return pl.DataFrame(schema={"station": pl.String, "distance_m": pl.Float64, "time": pl.Datetime,
                                    "temp_c": pl.Float64, "wspd_kmh": pl.Float64, "prcp_mm": pl.Float64})
    dist = stations["distance"].astype(float).to_dict()
    df = df.reset_index()
    return pl.DataFrame({
        "station": df["station"].astype(str).tolist(),
        "distance_m": [dist.get(sid) for sid in df["station"]],
        "time": df["time"].tolist(),
        "temp_c": df["temp"].astype(float).tolist(),
        "wspd_kmh": df["wspd"].astype(float).tolist(),
        "prcp_mm": df["prcp"].astype(float).tolist(),
    }, schema_overrides={"time": pl.Datetime}).with_columns(pl.col(pl.Float64).fill_nan(None))


def _pick(obs: pl.DataFrame, kickoff: datetime):
    """Closest usable observation: smallest time gap (<= MAX_HOUR_GAP h), then nearest station.

    Usable = temperature and wind both present. Returns a row dict or None.
    """
    k = _naive_utc(kickoff)
    cand = (obs.filter(pl.col("temp_c").is_not_null() & pl.col("wspd_kmh").is_not_null())
            .with_columns(gap=(pl.col("time") - pl.lit(k)).abs().dt.total_seconds() / 3600)
            .filter(pl.col("gap") <= MAX_HOUR_GAP)
            .sort("gap", "distance_m"))
    return cand.row(0, named=True) if cand.height else None


def _row(game_id, obs_row) -> dict:
    if obs_row is None:
        return _missing(game_id)
    prcp = obs_row["prcp_mm"]
    return dict(game_id=game_id, temp_f=obs_row["temp_c"] * 9 / 5 + 32, wind_mph=obs_row["wspd_kmh"] * KMH_TO_MPH,
                precip_in=None if prcp is None else prcp * MM_TO_IN, precip_prob_pct=None,
                source=SOURCE_METEOSTAT, station_id=obs_row["station"],
                station_km=None if obs_row["distance_m"] is None else obs_row["distance_m"] / 1000)


def fetch_weather(game_id, lat, lon, kickoff: datetime) -> dict:
    """One game's weather from Meteostat, or a 'missing' row. Never raises."""
    return pull_weather_rows([{"game_id": game_id, "lat": lat, "lon": lon, "kickoff": kickoff}])[0]


def pull_weather_rows(rows) -> list[dict]:
    """Weather rows for request dicts. One Meteostat load per venue per calendar year."""
    rows = list(rows)
    out: dict[int, dict] = {}
    by_venue: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("lat") is None or r.get("lon") is None or r.get("kickoff") is None:
            out[i] = _missing(r["game_id"])
        else:
            # Meteostat blocks hourly requests longer than 3 years, so one load per venue per year.
            key = (round(r["lat"], 4), round(r["lon"], 4), _naive_utc(r["kickoff"]).year)
            by_venue.setdefault(key, []).append(i)
    for (lat, lon, _year), idx in by_venue.items():
        kicks = [_naive_utc(rows[i]["kickoff"]) for i in idx]
        pad = timedelta(hours=MAX_HOUR_GAP + 1)
        try:
            obs = _load_station_hours(lat, lon, min(kicks) - pad, max(kicks) + pad)
        except Exception as e:
            log.warning("weather: venue (%s, %s) failed (%s); %d games -> missing", lat, lon, e, len(idx))
            obs = None
        for i in idx:
            try:
                out[i] = _row(rows[i]["game_id"], None if obs is None else _pick(obs, rows[i]["kickoff"]))
            except Exception as e:  # never raise for a single game
                log.warning("weather %s: %s", rows[i]["game_id"], e)
                out[i] = _missing(rows[i]["game_id"])
    return [out[i] for i in range(len(rows))]


def pull_weather(rows, db_path=config.RAW_DUCKDB_PATH) -> pl.DataFrame:
    """rows: iterable of dicts with game_id, lat, lon, kickoff. Appends to the weather table."""
    df = pl.DataFrame(
        pull_weather_rows(rows),
        schema={"game_id": pl.String, "temp_f": pl.Float64, "wind_mph": pl.Float64, "precip_in": pl.Float64,
                "precip_prob_pct": pl.Float64, "source": pl.String, "station_id": pl.String,
                "station_km": pl.Float64},
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


def pull_all_weather(db_path=config.RAW_DUCKDB_PATH, raw_db=config.RAW_DUCKDB_PATH, batch=500) -> dict:
    """Pull weather for every distinct game in the schedules table.

    Appends in batches (so an interrupted run keeps its progress) and returns a
    summary: rows inserted, missing, and the source breakdown.
    """
    import duckdb

    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        games = con.execute(
            "SELECT game_id, gameday, gametime, home_team, stadium_id, location FROM "
            "(SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC) "
            "ORDER BY gameday, game_id"
        ).pl().to_dicts()
    finally:
        con.close()
    reqs = sorted(build_requests(games), key=lambda r: (r['lat'] is None, r['lat'], r['lon'], str(r['kickoff'])))
    frames = []
    for i in range(0, len(reqs), batch):
        frames.append(pull_weather(reqs[i:i + batch], db_path))
        log.info("weather: %d/%d games", min(i + batch, len(reqs)), len(reqs))
    df = pl.concat(frames) if frames else pl.DataFrame()
    by_source = df["source"].value_counts().sort("source").to_dicts() if df.height else []
    return {"inserted": df.height, "missing": int((df["source"] == SOURCE_MISSING).sum()) if df.height else 0,
            "by_source": {r["source"]: r["count"] for r in by_source}}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = pull_all_weather()
    print(f"weather pull: {r['inserted']} rows inserted, {r['missing']} missing, sources: {r['by_source']}")
