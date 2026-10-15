"""Rebuild a stored 2025/2026 prediction from its stored feature contributions, with each layer (group of features) scaled by its own weight.

The explain tables of a run hold, for every player and quantity (targets, catch rate, carries, yards per carry, ...), the model's baseline and one additive contribution per
feature; they sum exactly to the quantity's prediction. Market means are products of quantities (rush yards = carries x yards per carry; receptions = targets x catch rate;
receiving yards = receptions x yards per catch; pass completions = attempts x completion rate; pass yards = completions x yards per completion). Scaling a group's
contributions by w is "this layer counts w times as much". w = 1 for every group gives back the stored mean.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

GROUPS = ["injuries", "opponent", "recent_form", "team_offense", "role"]


def group_of(f: str) -> str:
    if f == "(baseline)":
        return "baseline"
    if f.startswith(("inj_", "opp_inj_")):
        return "injuries"
    if f.startswith(("opp_def_", "opp_role_")):
        return "opponent"
    if f.startswith(("rec_", "rc_")) or f in ("last_val", "last3_val", "cont_mean_last3", "w_mass", "n_hist"):
        return "recent_form"
    if f.startswith("team_off_"):
        return "team_offense"
    return "role"


QTY = {"pass_att": ["pass_att"], "pass_cmp": ["pass_att", "comp_pct"], "pass_yds": ["pass_att", "comp_pct", "yds_per_cmp"], "rush_att": ["rush_att"], "rush_yds": ["rush_att", "ypc"],
       "rec": ["targets", "catch_pct"], "rec_yds": ["targets", "catch_pct", "yds_per_rec"]}


def parts() -> pl.DataFrame:
    """One row per (game, player, stack, quantity): the baseline and the summed contribution of each group."""
    fs = [f for f in sorted(glob.glob(str(config.ROOT / "records" / "*" / "predictions" / "explain" / "*.parquet")))
          if "margin" not in f and ("holdout_2025" in f or "retro_2026" in f)]
    d = pl.concat([pl.read_parquet(f).select("game_id", "player_id", "stack", "quantity", "feature", "contribution", "run_id") for f in fs])
    d = d.filter(pl.col("player_id").is_not_null()).with_columns(group=pl.col("feature").map_elements(group_of, return_dtype=pl.String))
    # the latest run per game (one per week in these replays)
    last = d.group_by("game_id").agg(run_id=pl.col("run_id").max())
    d = d.join(last, on=["game_id", "run_id"], how="semi")
    w = d.group_by("game_id", "player_id", "stack", "quantity", "group").agg(c=pl.col("contribution").sum())
    return w.pivot(on="group", index=["game_id", "player_id", "stack", "quantity"], values="c").fill_null(0.0)


def market_means(p: pl.DataFrame, preds: pl.DataFrame, weights: dict | None = None) -> pl.DataFrame:
    """Mean per (game, player, market) with group weights; `preds` carries each prediction's stack (game_id, player_id, market, stack)."""
    weights = weights or {}
    q = p.with_columns(val=pl.col("baseline") + pl.sum_horizontal([pl.col(g) * float(weights.get(g, 1.0)) for g in GROUPS if g in p.columns]))
    out = []
    for m, qs in QTY.items():
        t = preds.filter(pl.col("market") == m).select("game_id", "player_id", "stack")
        for qq in qs:
            t = t.join(q.filter(pl.col("quantity") == qq).select("game_id", "player_id", "stack", pl.col("val").alias(qq)), on=["game_id", "player_id", "stack"], how="left")
        t = t.with_columns(mean_w=pl.reduce(lambda a, b: a * b, [pl.col(qq) for qq in qs]), market=pl.lit(m))
        out.append(t.select("game_id", "player_id", "market", "mean_w"))
    return pl.concat(out)


def stored_preds() -> pl.DataFrame:
    from app import data as D
    from live import store as ST
    ps = [D.latest(ST.PredictionStore(config.ROOT / "records" / s / "predictions").load()) for s in ("holdout_2025", "retro_2026")]
    return pl.concat([p.select("game_id", "player_id", "market", "stack", "mean") for p in ps]).filter(pl.col("player_id").is_not_null() & pl.col("market").is_in(list(QTY)))


if __name__ == "__main__":
    p, sp = parts(), stored_preds()
    mm = market_means(p, sp).join(sp, on=["game_id", "player_id", "market"])
    mm = mm.with_columns(err=(pl.col("mean_w") - pl.col("mean")).abs())
    print(mm.group_by("market").agg(n=pl.len(), missing=pl.col("mean_w").is_null().sum(), max_err=pl.col("err").max(), mean_err=pl.col("err").mean()).sort("market"))
