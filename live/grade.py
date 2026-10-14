"""Weekly grading of stored predictions (plan P5.5): idempotent, pure track only, did-not-play rows kept apart.

`grade(gradable, actuals, path, graded_at)` grades the newest pre-kickoff version of every prediction whose result is in `actuals` (game_id, player_id, market,
actual; a player market with no actuals row for an eligible player whose game HAS results is a did-not-play: actual 0, flagged), appends to the graded table at
`path` (parquet) and never regrades a prediction_id. Quantile / ladder scores stay null until those fields exist.
"""
from __future__ import annotations

from pathlib import Path

import json

import polars as pl

GAME_MARKETS = ("spread", "moneyline", "total")
QCOLS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")


def _ladder_hits(ladder, actual):
    """{rung: 1 if the actual reached it} for a stored JSON ladder, None without one."""
    if ladder is None:
        return None
    return json.dumps({t: int(actual >= float(t)) for t in json.loads(ladder)}, sort_keys=True)


def _ladder_brier(ladder, actual):
    if ladder is None:
        return None
    lad = json.loads(ladder)
    return sum((p - float(actual >= float(t))) ** 2 for t, p in lad.items()) / len(lad) if lad else None


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
        "prediction_id", "game_id", "player_id", "market", "version", "model_version", "stack", "mean", "actual", "did_not_play", track=pl.lit("pure"),
        error=pl.col("mean") - pl.col("actual"), abs_error=(pl.col("mean") - pl.col("actual")).abs(),
        brier=pl.when(pl.col("market") == "moneyline").then((pl.col("mean") - pl.col("actual")) ** 2).otherwise(None),
        quantile_band=pl.when(pl.col("q05").is_not_null()).then(pl.sum_horizontal([(pl.col("actual") > pl.col(q)).cast(pl.Int32) for q in QCOLS])).otherwise(None).cast(pl.Int32),
        in_50=pl.when(pl.col("q25").is_not_null()).then((pl.col("actual") >= pl.col("q25")) & (pl.col("actual") <= pl.col("q75"))),
        in_80=pl.when(pl.col("q10").is_not_null()).then((pl.col("actual") >= pl.col("q10")) & (pl.col("actual") <= pl.col("q90"))),
        in_90=pl.when(pl.col("q05").is_not_null()).then((pl.col("actual") >= pl.col("q05")) & (pl.col("actual") <= pl.col("q95"))),
        ladder_hits=pl.struct("ladder", "actual").map_elements(lambda r: _ladder_hits(r["ladder"], r["actual"]), return_dtype=pl.String),
        ladder_brier=pl.struct("ladder", "actual").map_elements(lambda r: _ladder_brier(r["ladder"], r["actual"]), return_dtype=pl.Float64),
        graded_at=pl.lit(graded_at).cast(pl.Datetime("us", "UTC"))).sort("game_id", "market", "player_id")
    out = pl.concat([done, new]) if done is not None else new
    if new.height:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path, compression="zstd")
    return out


def summary(graded: pl.DataFrame) -> pl.DataFrame:
    """Per stack label and market: played rows and did-not-play rows reported separately; stacks are never blended into one average."""
    played, dnp = pl.col("did_not_play").not_(), pl.col("did_not_play")
    return (graded.group_by("stack", "market").agg(
        n_played=played.sum(), n_did_not_play=dnp.sum(),
        mae_played=pl.col("abs_error").filter(played).mean(), mae_did_not_play=pl.col("abs_error").filter(dnp).mean(),
        brier=pl.col("brier").mean(), ladder_brier=pl.col("ladder_brier").mean(), n_with_distribution=pl.col("in_80").is_not_null().sum(),
        cov_50=pl.col("in_50").filter(played).mean(), cov_80=pl.col("in_80").filter(played).mean(), cov_90=pl.col("in_90").filter(played).mean()).sort("stack", "market"))
