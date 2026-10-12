"""4e weather features (the gate is fixed in docs/overnight_plan.md "4e gate" before this was built).

The backtest only has OBSERVED weather (Meteostat station readings at kickoff), which nobody knew before kickoff. A layer that works only on exact
observed weather will not work live, so the gate runs the model on weather degraded to forecast quality, three ways, and the layer must pass under all:
  A1    bins          wind <10 / 10-15 / 15-20 / 20+ mph, temperature <=32 / 33-50 / 51-70 / >70 F, precipitation yes / no (the model sees the bin index)
  A2    noise         wind +/- N(0, 4 mph), temperature +/- N(0, 4 F), precipitation flipped with probability 0.2; one draw per game, seeded by the game id
  A2x2  double noise  the same draws at twice the size: N(0, 8 mph), N(0, 8 F), flip probability 0.4
  exact observed values, for reference only (never gates)
Domes and closed roofs (indoor True) get the neutral values wind 0, temperature 70 F, precipitation 0 and an explicit indoor flag, in every mode; they
are never binned or noised, and their observed row is ignored. An unknown roof (indoor null), a missing weather row and a missing component give
null features (never neutral, never zero). Columns (float): wx_indoor, wx_wind, wx_temp, wx_precip, wx_windy (wind >= 15 mph; the bin >= 2 under A1),
wx_cold (temperature <= 32 F; bin 0 under A1). Weather is a property of the game: both teams get the same values.
"""
from __future__ import annotations

import hashlib

import duckdb
import numpy as np
import polars as pl

import config

MODES = ("exact", "A1", "A2", "A2x2")
A2_WIND_SD, A2_TEMP_SD, A2_PRECIP_FLIP, A2_SEED = 4.0, 4.0, 0.2, 20261012
SCALE = {"A2": 1.0, "A2x2": 2.0}
INDOOR_NEUTRAL = {"wind": 0.0, "temp": 70.0, "precip": 0.0}
WINDY_MPH, COLD_F = 15.0, 32.0
WX_COLUMNS = ["wx_indoor", "wx_wind", "wx_temp", "wx_precip", "wx_windy", "wx_cold"]


def bin_wind(x: float) -> int:
    return 0 if x < 10.0 else 1 if x < 15.0 else 2 if x < 20.0 else 3


def bin_temp(x: float) -> int:
    return 0 if x <= 32.0 else 1 if x <= 50.0 else 2 if x <= 70.0 else 3


def _draws(game_id: str) -> tuple:
    """(z_wind, z_temp, u_precip) of a game: a generator seeded by the game id alone, so a draw never depends on the other games or the row order."""
    seed = int.from_bytes(hashlib.sha256(f"{A2_SEED}|{game_id}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return float(rng.standard_normal()), float(rng.standard_normal()), float(rng.random())


def _flags(wind, temp, mode):
    if mode == "A1":
        return (None if wind is None else float(wind >= 2), None if temp is None else float(temp == 0))
    return (None if wind is None else float(wind >= WINDY_MPH), None if temp is None else float(temp <= COLD_F))


def _one(r: dict, mode: str) -> dict:
    out = dict.fromkeys(WX_COLUMNS)
    indoor = r.get("indoor")
    if indoor is None:                                                      # roof state unknown: nothing is pretended
        return out
    out["wx_indoor"] = 1.0 if indoor else 0.0
    if indoor:
        wind, temp, precip = INDOOR_NEUTRAL["wind"], INDOOR_NEUTRAL["temp"], INDOOR_NEUTRAL["precip"]
        if mode == "A1":
            wind, temp = float(bin_wind(wind)), float(bin_temp(temp))
        out.update(wx_wind=wind, wx_temp=temp, wx_precip=precip, wx_windy=0.0, wx_cold=0.0)        # a dome is never windy or cold
        return out
    wind, temp = r.get("wind_mph"), r.get("temp_f")
    precip = None if r.get("precip_in") is None else float(r["precip_in"] > 0)
    if mode in SCALE:
        s = SCALE[mode]
        zw, zt, u = _draws(r["game_id"])
        wind = None if wind is None else max(0.0, wind + A2_WIND_SD * s * zw)
        temp = None if temp is None else temp + A2_TEMP_SD * s * zt
        precip = None if precip is None else (1.0 - precip if u < A2_PRECIP_FLIP * s else precip)
    elif mode == "A1":
        wind = None if wind is None else float(bin_wind(wind))
        temp = None if temp is None else float(bin_temp(temp))
    out.update(wx_wind=wind, wx_temp=temp, wx_precip=precip)
    out["wx_windy"], out["wx_cold"] = _flags(wind, temp, mode)
    return out


def features(wx: pl.DataFrame, mode: str, games: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per game: the six weather columns under `mode`. wx: game_id, temp_f, wind_mph, precip_in, indoor (one row per game). `games` (game_id) adds
    null rows for games without a weather row."""
    if mode not in MODES:
        raise ValueError(f"unknown weather mode {mode!r}")
    if wx["game_id"].is_duplicated().any():
        raise ValueError("more than one weather row for a game")
    rows = [dict(game_id=r["game_id"], **_one(r, mode)) for r in wx.iter_rows(named=True)]
    schema = {"game_id": pl.String, **{c: pl.Float64 for c in WX_COLUMNS}}
    f = pl.DataFrame(rows, schema=schema)
    if games is not None:
        f = games.select("game_id").unique().join(f, on="game_id", how="left")
    return f.sort("game_id")


def severe_games(wx: pl.DataFrame) -> list:
    """Outdoor games with real wind (>= 15 mph) or cold (<= 32 F) by the OBSERVED values: the rows the gate is judged on."""
    return sorted(wx.filter((pl.col("indoor") == False) & ((pl.col("wind_mph") >= WINDY_MPH) | (pl.col("temp_f") <= COLD_F)))["game_id"].to_list())  # noqa: E712


def load_observed(raw_db=config.RAW_DUCKDB_PATH, first: int = 2020, last: int = max(config.BACKTEST_SEASONS)) -> pl.DataFrame:
    """Observed weather rows (one per game; the latest pull) for games first..last; never the holdout."""
    if last >= config.HOLDOUT_SEASON:
        raise config.HoldoutError(f"weather through {last} would read the holdout season {config.HOLDOUT_SEASON}")
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        return con.execute("SELECT game_id, temp_f, wind_mph, precip_in, indoor FROM (SELECT DISTINCT ON (game_id) * FROM weather_observed "
                           "ORDER BY game_id, pulled_at DESC) WHERE CAST(substr(game_id, 1, 4) AS INTEGER) BETWEEN ? AND ? ORDER BY game_id",
                           [first, last]).pl()
    finally:
        con.close()


def attach(tables, feats: pl.DataFrame | None):
    """FeatureTables with the weather columns joined on game_id (players and teams; both teams of a game share them). None = Phase 3 unchanged."""
    from features.volume_features import FeatureTables
    if feats is None:
        return tables
    if feats["game_id"].is_duplicated().any():
        raise ValueError("weather features have more than one row for a game")
    missing = [c for c in WX_COLUMNS if c not in feats.columns]
    if missing:
        raise KeyError(f"weather features are missing columns {missing}")
    f = feats.select("game_id", *[pl.col(c).cast(pl.Float64) for c in WX_COLUMNS])
    players = {q: df.join(f, on="game_id", how="left", maintain_order="left") for q, df in tables.players.items()}
    return FeatureTables(players, tables.teams.join(f, on="game_id", how="left", maintain_order="left"))
