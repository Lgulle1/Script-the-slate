"""Weekly grading of stored predictions (plan P5.5): idempotent, pure track only, did-not-play rows kept apart.

`grade(gradable, actuals, path, graded_at)` grades the newest pre-kickoff version of every prediction whose result is in `actuals` (game_id, player_id, market,
actual; a player market with no actuals row for an eligible player whose game HAS results is a did-not-play: actual 0, flagged), appends to the graded table at
`path` (parquet) and never regrades a prediction_id. Quantile / ladder scores stay null until those fields exist.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

GAME_MARKETS = ("spread", "moneyline", "total")


def grade(gradable: pl.DataFrame, actuals: pl.DataFrame, path, graded_at) -> pl.DataFrame:
    path = Path(path)
    done = pl.read_parquet(path) if path.exists() else None
    g = gradable.filter(~pl.col("prediction_id").is_in(done["prediction_id"].to_list())) if done is not None else gradable
    played_games = set(actuals["game_id"].unique().to_list())
    a = actuals.select("game_id", "player_id", "market", "actual")
    j = g.join(a, on=["game_id", "player_id", "market"], how="left", nulls_equal=True)
    is_game = pl.col("market").is_in(GAME_MARKETS)
    j = j.filter(pl.col("game_id").is_in(played_games))                                    # the game has a result
    j = j.filter(pl.col("actual").is_not_null() | ~is_game)                                  # a missing game result waits; a missing player result is a did-not-play
    j = j.with_columns(did_not_play=pl.col("actual").is_null() & ~is_game, actual=pl.col("actual").fill_null(0.0))
    new = j.select(
        "prediction_id", "game_id", "player_id", "market", "version", "model_version", "mean", "actual", "did_not_play", track=pl.lit("pure"),
        error=pl.col("mean") - pl.col("actual"), abs_error=(pl.col("mean") - pl.col("actual")).abs(),
        brier=pl.when(pl.col("market") == "moneyline").then((pl.col("mean") - pl.col("actual")) ** 2).otherwise(None),
        quantile_band=pl.lit(None, dtype=pl.Int32), graded_at=pl.lit(graded_at).cast(pl.Datetime("us", "UTC"))).sort("game_id", "market", "player_id")
    out = pl.concat([done, new]) if done is not None else new
    if new.height:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path, compression="zstd")
    return out


def summary(graded: pl.DataFrame) -> pl.DataFrame:
    """Per market: played rows and did-not-play rows reported separately, never pooled into one average."""
    played, dnp = pl.col("did_not_play").not_(), pl.col("did_not_play")
    return (graded.group_by("market").agg(
        n_played=played.sum(), n_did_not_play=dnp.sum(),
        mae_played=pl.col("abs_error").filter(played).mean(), mae_did_not_play=pl.col("abs_error").filter(dnp).mean(),
        brier=pl.col("brier").mean()).sort("market"))
