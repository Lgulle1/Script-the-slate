"""Layer 4b in a live run: the simulated spread (plan P5.7).

The game simulation (sim/simulate.py) draws the final margin as Normal(margin_mean, margin_sd) from a stream seeded by the game id, so the simulated spread of the
validated layer (4b.5, backtest mode, 2,000 draws) is the mean of that draw: the 4b.2 expected margin with its fixed-seed Monte Carlo wobble. margin_mean and margin_sd
come from models/game_state.py with the scaling terms fit on games before `floor_key` (2020-2024 rows only, plan P5.4) and the SD the RMS of the walk-forward errors of the
games before it.
"""
from __future__ import annotations

import polars as pl

from models import game_state as gs
from sim import simulate as sm


def spread_sim_mean(game_id: str, margin_mean: float, margin_sd: float, mode: sm.SimMode = sm.BACKTEST) -> float:
    """Mean of the first stream of simulate_game for this game: exactly what the 4b.5 backtest called sim_mean for the spread."""
    return float(sm.game_rng(game_id).normal(margin_mean, margin_sd, mode.n_sim).mean())


def game_expectations_live(features: pl.DataFrame, games: pl.DataFrame, game_ids: list, season: int, week: int, floor_key: int) -> pl.DataFrame:
    """game_id, margin_mean, margin_sd for the target games."""
    hist, _ = gs.walk_forward(features.filter((pl.col("season") * 100 + pl.col("week")) < floor_key))
    target, coef = gs.predict_week(features, season, week, floor_key)
    if coef is None:
        return pl.DataFrame(schema={"game_id": pl.String, "margin_mean": pl.Float64, "margin_sd": pl.Float64})
    ge = gs.game_expectations(pl.concat([hist, target], how="diagonal_relaxed"), games)
    return ge.filter(pl.col("game_id").is_in(game_ids)).select("game_id", "margin_mean", "margin_sd")


def spread_rows(ge: pl.DataFrame) -> pl.DataFrame:
    """(game_id, mean) for the spread market; a game without an SD yet (fewer than MIN_SD_GAMES earlier games) has no 4b prediction."""
    rows = [(r["game_id"], spread_sim_mean(r["game_id"], r["margin_mean"], r["margin_sd"])) for r in ge.iter_rows(named=True)
            if r["margin_mean"] is not None and r["margin_sd"] is not None]
    return pl.DataFrame(rows, schema={"game_id": pl.String, "mean": pl.Float64}, orient="row")
