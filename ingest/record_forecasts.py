"""Record weather forecasts as they stand before kickoff (option C of the 4e decision, 2026-10-12).

The backtest can only use observed weather (nobody knew it before kickoff), so the only route to a real forecast-time record is to keep every
forecast the pipeline sees. Each run snapshots the forecast of every live-season game kicking off in the next HORIZON_HOURS and APPENDS it to
records/weather_forecast_log.csv, a TRACKED file (data/ is gitignored and the container is ephemeral), with its pulled_at and hours_before_kickoff.
A snapshot is never rewritten. The "day-before" forecast of a game is the snapshot closest to 24 hours before kickoff.

  python -m ingest.record_forecasts          snapshot now (also appends to the raw weather table, append-only)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

import config
from ingest import pull_weather as pw

HORIZON_HOURS = 96
MIN_GAP_HOURS = 3.0
LOG_PATH = config.ROOT / "records" / "weather_forecast_log.csv"
COLUMNS = ["game_id", "kickoff_utc", "pulled_at", "hours_before_kickoff", "source", "temp_f", "wind_mph", "precip_in", "indoor", "station_id", "station_km"]
SCHEMA = {"game_id": pl.String, "kickoff_utc": pl.String, "pulled_at": pl.String, "hours_before_kickoff": pl.Float64, "source": pl.String,
          "temp_f": pl.Float64, "wind_mph": pl.Float64, "precip_in": pl.Float64, "indoor": pl.Boolean, "station_id": pl.String, "station_km": pl.Float64}


def select_requests(reqs: list, now: datetime, horizon_hours: float = HORIZON_HOURS) -> list:
    """Requests of live-season games (after the holdout season) kicking off after `now` and within the horizon."""
    now = pw._naive_utc(now)
    out = []
    for r in reqs:
        k = r.get("kickoff")
        if k is None or int(r["game_id"][:4]) <= config.HOLDOUT_SEASON:
            continue
        ahead = (pw._naive_utc(k) - now).total_seconds() / 3600
        if 0 < ahead <= horizon_hours:
            out.append(r)
    return out


def record(reqs: list, now: datetime, log_path: Path = LOG_PATH, fetch=pw.pull_weather_rows, min_gap_hours: float = MIN_GAP_HOURS,
           horizon_hours: float = HORIZON_HOURS, select: bool = True) -> int:
    """Append a forecast snapshot for each selected request; returns the rows added. A game already recorded within `min_gap_hours` is skipped."""
    now = pw._naive_utc(now)
    if select:
        reqs = select_requests(reqs, now, horizon_hours)
    elif any(int(r["game_id"][:4]) <= config.HOLDOUT_SEASON for r in reqs):
        raise config.HoldoutError("the forecast recorder is for the live season only")
    old = pl.read_csv(log_path, schema_overrides=SCHEMA) if log_path.exists() else pl.DataFrame(schema=SCHEMA)
    last = {}
    for g, p in zip(old["game_id"].to_list(), old["pulled_at"].to_list()):
        t = datetime.fromisoformat(p)
        last[g] = max(last.get(g, t), t)
    todo = [r for r in reqs if r["game_id"] not in last or (now - last[r["game_id"]]).total_seconds() / 3600 >= min_gap_hours]
    if not todo:
        return 0
    rows = fetch(todo, now)
    kick = {r["game_id"]: pw._naive_utc(r["kickoff"]) for r in todo}
    new = pl.DataFrame([dict(r, kickoff_utc=kick[r["game_id"]].isoformat(), pulled_at=now.isoformat(),
                             hours_before_kickoff=(kick[r["game_id"]] - now).total_seconds() / 3600) for r in rows],
                       schema_overrides={k: v for k, v in SCHEMA.items()}, infer_schema_length=None).select(COLUMNS)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        with open(log_path, "a") as f:                                     # append only: earlier lines are never touched
            f.write(new.write_csv(include_header=False))
    else:
        log_path.write_text(new.write_csv())
    return new.height


def live_schedule(raw_db=config.RAW_DUCKDB_PATH, loader=None) -> list:
    """The live season's games (game_id, gameday, gametime, home_team, stadium_id, location, roof): from the raw database when it holds them, else
    straight from nflverse (a fresh container has no raw database)."""
    import duckdb
    cols = "game_id, gameday, gametime, home_team, stadium_id, location, roof"
    if Path(raw_db).exists():
        con = duckdb.connect(str(raw_db), read_only=True)
        try:
            games = con.execute(f"SELECT {cols} FROM (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC) "
                                f"WHERE season > {config.HOLDOUT_SEASON} ORDER BY gameday, game_id").pl().to_dicts()
        finally:
            con.close()
        if games:
            return games
    if loader is None:
        import nflreadpy
        loader = nflreadpy.load_schedules
    d = loader([config.HOLDOUT_SEASON + 1])
    d = d.to_polars() if hasattr(d, "to_polars") else d
    return d.select(cols.split(", ")).sort("gameday", "game_id").to_dicts()


def main(now: datetime | None = None) -> int:
    now = pw._naive_utc(now or datetime.now(timezone.utc))
    reqs = select_requests(pw.build_requests(live_schedule()), now)
    fetched = []

    def fetch(todo, at):                                                   # one Meteostat lookup feeds both the log and the raw table
        rows = pw.pull_weather_rows(todo, at)
        fetched.extend(rows)
        return rows

    n = record(reqs, now, select=False, fetch=fetch)
    if fetched and Path(config.RAW_DUCKDB_PATH).exists():                  # the raw table keeps its own append-only copy when there is one
        pw._append_raw(pw.WEATHER_TABLE, pl.DataFrame(fetched, schema=pw.WEATHER_SCHEMA), config.RAW_DUCKDB_PATH, pulled_at=now)
        pw.ensure_observed_view(config.RAW_DUCKDB_PATH)
    return n


if __name__ == "__main__":
    print(f"recorded {main()} forecast snapshots to {LOG_PATH.relative_to(config.ROOT)}")
