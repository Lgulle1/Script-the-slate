"""Pull game-time weather from Meteostat (hourly station data).

Input rows are dicts: {game_id, lat, lon, kickoff} with kickoff a datetime
(naive = UTC). Output rows carry temp_f, wind_mph, precip_in, a null
precip_prob_pct (Meteostat has no forecast probability), an `indoor` flag and a source tag:

  meteostat_observed  the kickoff was already past when the row was pulled (station observations)
  meteostat_forecast  the kickoff was still ahead (Meteostat returns model forecasts for the coming
                      week); the row also stores hours_before_kickoff, and pulled_at says when
  missing             nothing usable (no coordinates, no station reading, or a failed lookup)

Baselines and backtests use OBSERVED rows only (`load_weather` / the `weather_observed` view); a forecast
row is never a training label. A failed lookup never raises: it yields a "missing" row.

`indoor` = a fixed dome, or a retractable roof the schedules roof column says was closed (None when the
roof state is unknown). Weather is still pulled for indoor games; consumers must not read it as game
conditions when indoor is true.
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

SOURCE_OBSERVED, SOURCE_FORECAST, SOURCE_MISSING = "meteostat_observed", "meteostat_forecast", "missing"
OBSERVED_VIEW = "weather_observed"
WEATHER_TABLE = "weather"
N_STATIONS = 4               # nearest stations considered per venue
MAX_HOUR_GAP = 1             # accept an observation up to this many hours from kickoff
KMH_TO_MPH = 0.621371
MM_TO_IN = 1 / 25.4


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _naive_utc(dt: datetime) -> datetime:
    return _utc(dt).replace(tzinfo=None)


def _missing(game_id, indoor=None, hours_before=None) -> dict:
    return dict(game_id=game_id, temp_f=None, wind_mph=None, precip_in=None, precip_prob_pct=None,
                source=SOURCE_MISSING, station_id=None, station_km=None, indoor=indoor, hours_before_kickoff=hours_before)


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


def _hours_before(kickoff, now):
    """Hours from the pull to a kickoff still ahead of it (None once the kickoff has passed)."""
    h = (_naive_utc(kickoff) - now).total_seconds() / 3600
    return h if h > 0 else None


def _row(game_id, obs_row, kickoff, now, indoor=None) -> dict:
    hours_before = _hours_before(kickoff, now)
    if obs_row is None:
        return _missing(game_id, indoor, hours_before)
    prcp = obs_row["prcp_mm"]
    return dict(game_id=game_id, temp_f=obs_row["temp_c"] * 9 / 5 + 32, wind_mph=obs_row["wspd_kmh"] * KMH_TO_MPH,
                precip_in=None if prcp is None else prcp * MM_TO_IN, precip_prob_pct=None,
                source=SOURCE_FORECAST if hours_before is not None else SOURCE_OBSERVED, station_id=obs_row["station"],
                station_km=None if obs_row["distance_m"] is None else obs_row["distance_m"] / 1000,
                indoor=indoor, hours_before_kickoff=hours_before)


def fetch_weather(game_id, lat, lon, kickoff: datetime, indoor=None, now: datetime | None = None) -> dict:
    """One game's weather from Meteostat, or a 'missing' row. Never raises."""
    return pull_weather_rows([{"game_id": game_id, "lat": lat, "lon": lon, "kickoff": kickoff, "indoor": indoor}], now)[0]


def pull_weather_rows(rows, now: datetime | None = None) -> list[dict]:
    """Weather rows for request dicts. One Meteostat load per venue per calendar year.

    `now` is the pull time: a kickoff at or before it is OBSERVED, one after it is a FORECAST.
    """
    now = _naive_utc(now or datetime.now(timezone.utc))
    rows = list(rows)
    out: dict[int, dict] = {}
    by_venue: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("lat") is None or r.get("lon") is None or r.get("kickoff") is None:
            out[i] = _missing(r["game_id"], r.get("indoor"))
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
                out[i] = _row(rows[i]["game_id"], None if obs is None else _pick(obs, rows[i]["kickoff"]),
                              rows[i]["kickoff"], now, rows[i].get("indoor"))
            except Exception as e:  # never raise for a single game
                log.warning("weather %s: %s", rows[i]["game_id"], e)
                out[i] = _missing(rows[i]["game_id"], rows[i].get("indoor"))
    return [out[i] for i in range(len(rows))]


WEATHER_SCHEMA = {"game_id": pl.String, "temp_f": pl.Float64, "wind_mph": pl.Float64, "precip_in": pl.Float64,
                  "precip_prob_pct": pl.Float64, "source": pl.String, "station_id": pl.String, "station_km": pl.Float64,
                  "indoor": pl.Boolean, "hours_before_kickoff": pl.Float64}


def pull_weather(rows, db_path=config.RAW_DUCKDB_PATH, now: datetime | None = None) -> pl.DataFrame:
    """rows: iterable of dicts with game_id, lat, lon, kickoff[, indoor]. Appends to the weather table.

    Append-only: every pull adds rows stamped with pulled_at (= `now`, the same instant that decides
    observed vs forecast), so a game forecast on Wednesday and observed on Monday keeps both rows.
    """
    now = _naive_utc(now or datetime.now(timezone.utc))
    df = pl.DataFrame(pull_weather_rows(rows, now), schema=WEATHER_SCHEMA)
    if df.height:
        _append_raw(WEATHER_TABLE, df, db_path, pulled_at=now)
        ensure_observed_view(db_path)
    return df


def ensure_observed_view(db_path=config.RAW_DUCKDB_PATH):
    """weather_observed: the latest OBSERVED row per game. Anything that trains or backtests reads this."""
    import duckdb
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE OR REPLACE VIEW {OBSERVED_VIEW} AS SELECT * FROM {WEATHER_TABLE} "
                    f"WHERE source = '{SOURCE_OBSERVED}' QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY pulled_at DESC) = 1")
    finally:
        con.close()


def load_weather(db_path=config.RAW_DUCKDB_PATH, observed_only: bool = True) -> pl.DataFrame:
    """Latest weather row per game. observed_only (the default, used by baselines and backtests) returns only
    meteostat_observed rows; observed_only=False returns the latest row of any source, forecasts included."""
    import duckdb
    ensure_observed_view(db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if observed_only:
            return con.execute(f"SELECT * FROM {OBSERVED_VIEW} ORDER BY game_id").pl()
        return con.execute(f"SELECT * FROM {WEATHER_TABLE} QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY pulled_at DESC) = 1 "
                           "ORDER BY game_id").pl()
    finally:
        con.close()


def venue_roof(home_team_abbr: str, stadium_id: str | None) -> str | None:
    """The venue's fixed roof type from config (outdoors / dome / retractable), or None when the venue is not
    in the stadium table (international sites)."""
    team = config._DOMESTIC_VENUE_TEAMS.get(stadium_id)
    if team is None and stadium_id not in config.venue_coordinates:
        team = config.TEAM_ABBR_TO_NAME.get(home_team_abbr)
    entry = config.stadium_coordinates.get(team)
    return entry[2] if entry else None


def indoor_flag(home_team_abbr: str, stadium_id: str | None, schedule_roof: str | None):
    """True for a fixed dome or a roof the schedules table says was closed; False for an outdoor stadium or an
    open roof; None when the roof state is unknown (retractable or international venue with no roof entry)."""
    fixed = venue_roof(home_team_abbr, stadium_id)
    if fixed == config.DOME or schedule_roof in ("dome", "closed"):
        return True
    if schedule_roof in ("open", "outdoors") or fixed == config.OUTDOORS:
        return False
    return None


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
    """schedule_rows: dicts with game_id, gameday, gametime, home_team, stadium_id, location[, roof]."""
    out = []
    for r in schedule_rows:
        loc = resolve_location(r["home_team"], r.get("stadium_id"), r.get("location"))
        if loc is None:
            log.warning("weather: no coordinates for %s (stadium_id=%s)", r["game_id"], r.get("stadium_id"))
        out.append(dict(game_id=r["game_id"], lat=loc[0] if loc else None, lon=loc[1] if loc else None,
                        kickoff=kickoff_utc(r["gameday"], r.get("gametime")),
                        indoor=indoor_flag(r["home_team"], r.get("stadium_id"), r.get("roof"))))
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
            "SELECT game_id, gameday, gametime, home_team, stadium_id, location, roof FROM "
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


def migrate_weather(db_path=config.RAW_DUCKDB_PATH) -> dict:
    """One-time, idempotent upgrade of a weather table written before the observed/forecast split.

    Adds `indoor` and `hours_before_kickoff`, re-tags legacy 'meteostat' rows as observed (kickoff at or
    before the row's pulled_at) or forecast (kickoff still ahead), and fills indoor from the venue and the
    schedules roof column. Nothing is deleted and no weather value changes; only the tag and the two new
    columns are written. Returns counts by new source tag.
    """
    import duckdb
    con = duckdb.connect(str(db_path))
    try:
        cols = {r[0] for r in con.execute(f"DESCRIBE {WEATHER_TABLE}").fetchall()}
        if {"indoor", "hours_before_kickoff"} <= cols:
            return {}
        for name, typ in (("indoor", "BOOLEAN"), ("hours_before_kickoff", "DOUBLE")):
            if name not in cols:
                con.execute(f"ALTER TABLE {WEATHER_TABLE} ADD COLUMN {name} {typ}")
        legacy = con.execute(
            f"SELECT w.game_id, w.pulled_at, w.source, s.gameday, s.gametime, s.home_team, s.stadium_id, s.roof FROM {WEATHER_TABLE} w "
            "LEFT JOIN (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC) s USING (game_id)").pl()
        rows = []
        for r in legacy.iter_rows(named=True):
            kick = kickoff_utc(r["gameday"], r["gametime"]) if r["gameday"] else None
            pulled = r["pulled_at"]
            hours = _hours_before(kick, pulled) if kick is not None else None
            src = r["source"]
            if src == "meteostat":
                src = SOURCE_FORECAST if hours is not None else SOURCE_OBSERVED
            rows.append(dict(game_id=r["game_id"], pulled_at=pulled, source=src, hours=hours,
                             indoor=indoor_flag(r["home_team"], r["stadium_id"], r["roof"]) if r["home_team"] else None))
        stage = pl.DataFrame(rows, schema={"game_id": pl.String, "pulled_at": pl.Datetime, "source": pl.String, "hours": pl.Float64,
                                           "indoor": pl.Boolean})
        con.register("stage", stage.to_arrow())
        con.execute(f"UPDATE {WEATHER_TABLE} SET source = s.source, indoor = s.indoor, hours_before_kickoff = s.hours FROM stage s "
                    f"WHERE {WEATHER_TABLE}.game_id = s.game_id AND {WEATHER_TABLE}.pulled_at = s.pulled_at")
        counts = dict(con.execute(f"SELECT source, count(*) FROM {WEATHER_TABLE} GROUP BY 1 ORDER BY 1").fetchall())
    finally:
        con.close()
    ensure_observed_view(db_path)
    return counts
