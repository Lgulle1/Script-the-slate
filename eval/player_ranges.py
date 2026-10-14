"""Player-specific ranges (docs/overnight_plan.md, "Improvement plan from weeks 1-5", workstream B). Development seasons 2020-2024 only; no line is read.

The current range of a prediction is the quantiles of the K nearest past predictions in size (eval/calibrate.py), the same for every player. Here the range is rescaled by the player's own history:
  D(p)  mean absolute error of the pool's nearest predictions        d_j  mean absolute error of the player's earlier walk-forward predictions in the market (strictly before the target week)
  d~    = (n d_j + k D) / (n + k), n the player's earlier rows (r = 1 below 3 rows);   r = clip(d~ / D, 0.5, 2);   range = pool median + r * (pool offsets)
Then the existing width scale c (fitted on earlier folds only) is applied to each version.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from eval import calibrate as C

QUANTS = dict(zip(C.QCOLS, C.QS))
PLAYER_MARKETS = ("pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds", "targets", "rec", "rec_yds")
K_SHRINK = 8
MIN_PRIOR = 3
R_LO, R_HI = 0.5, 2.0
DRAWS, SEED = 2000, 7


def with_history(d: pl.DataFrame) -> pl.DataFrame:
    """n_prior and d_j (the mean absolute error of the entity's strictly earlier rows). One row per entity per week, so a shifted running sum is exactly the earlier rows."""
    d = d.sort("entity", "season", "week").with_columns(abs_err=(pl.col("actual") - pl.col("prediction")).abs())
    d = d.with_columns(n_prior=pl.int_range(pl.len()).over("entity"), sum_prior=(pl.col("abs_err").cum_sum() - pl.col("abs_err")).over("entity"))
    return d.with_columns(d_j=pl.when(pl.col("n_prior") > 0).then(pl.col("sum_prior") / pl.col("n_prior")))


def walk_forward(rows: pl.DataFrame, market: str, cfg: C.Config, k_shrink: int = K_SHRINK) -> pl.DataFrame:
    """Per (season, week, entity): the pooled quantiles (as calibrate.walk_forward), the pool's D, the player's d_j and n, and the ratio r."""
    C._check_seasons(rows)
    d = with_history(rows.filter(pl.col("market") == market).drop_nulls(["prediction", "actual"])).sort("season", "week", "entity")
    keys = d.select("season", "week").unique().sort("season", "week").rows()
    pred, act = d["prediction"].to_numpy(), d["actual"].to_numpy()
    sw = (d["season"] * 100 + d["week"]).to_numpy()
    out = []
    for s, w in keys:
        key = s * 100 + w
        pool = sw < key
        if pool.sum() < cfg.min_pool[market]:
            continue
        order = np.argsort(pred[pool], kind="stable")
        sp, sa = pred[pool][order], act[pool][order]
        tgt = np.flatnonzero(sw == key)
        idx = C._window(sp, pred[tgt], cfg.k[market])
        desc = C._describe(sa[idx], ())
        D = np.abs(sa[idx] - sp[idx]).mean(axis=1)
        t = d[tgt.tolist()]
        n = t["n_prior"].to_numpy().astype(float)
        dj = t["d_j"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            dt = (n * np.nan_to_num(dj) + k_shrink * D) / (n + k_shrink)
            r = np.where((n >= MIN_PRIOR) & (D > 0), np.clip(dt / D, R_LO, R_HI), 1.0)
        out.append(t.with_columns([pl.Series(q, v) for q, v in desc.items()] + [pl.Series("D", D), pl.Series("r", r)]))
    return pl.concat(out) if out else d.head(0)


def scaled(df: pl.DataFrame, per_player: bool) -> pl.DataFrame:
    """The frame with its quantile columns replaced by the player-specific version (median fixed, offsets times r)."""
    if not per_player:
        return df
    med = df["q50"].to_numpy()
    r = df["r"].to_numpy()
    return df.with_columns([pl.Series(q, med + r * (df[q].to_numpy() - med)) for q in C.QCOLS])


def pinball(df: pl.DataFrame, c: float) -> np.ndarray:
    """Per-row mean pinball loss over the seven quantiles after width scale c."""
    q = C._scaled(df, c)
    a = df["actual"].to_numpy()
    loss = np.zeros(len(a))
    for name, tau in QUANTS.items():
        e = a - q[name]
        loss += np.maximum(tau * e, (tau - 1) * e)
    return loss / len(QUANTS)


def boot_gain(gain: np.ndarray, cluster: np.ndarray) -> tuple:
    """Mean of `gain` and its 95% interval, resampling the clusters (season-weeks) with replacement."""
    u, inv = np.unique(cluster, return_inverse=True)
    s = np.bincount(inv, weights=gain)
    n = np.bincount(inv).astype(float)
    idx = np.random.default_rng(SEED).integers(0, len(u), size=(DRAWS, len(u)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def evaluate(wf: pl.DataFrame) -> dict:
    """Folds 2022-2024: width scale fitted on earlier seasons only, for the pooled and the player-specific range; per-fold and pooled pinball gain, coverage."""
    res = dict(folds=[], fold_gain=[], gains=[], clusters=[], cov80_ps=[], cov80_pool=[], n=0, c_pool=[], c_ps=[])
    for f in C.FOLDS:
        tr, te = wf.filter(pl.col("season") < f), wf.filter(pl.col("season") == f)
        if tr.height < 300 or te.height < 100:
            continue
        c0, c1 = C.fit_width(tr), C.fit_width(scaled(tr, True))
        l0, l1 = pinball(te, c0), pinball(scaled(te, True), c1)
        res["folds"].append(f)
        res["c_pool"].append(c0)
        res["c_ps"].append(c1)
        res["fold_gain"].append(float((l0 - l1).mean()))
        res["gains"].append(l0 - l1)
        res["clusters"].append((te["season"] * 100 + te["week"]).to_numpy())
        res["cov80_pool"].append(C.coverage(te, c0)[0.8])
        res["cov80_ps"].append(C.coverage(scaled(te, True), c1)[0.8])
        res["n"] += te.height
    if res["folds"]:
        g, cl = np.concatenate(res["gains"]), np.concatenate(res["clusters"])
        res["gain"], res["lo"], res["hi"] = boot_gain(g, cl)
        wts = np.array([len(x) for x in res["gains"]], float)
        res["cov80_ps_all"] = float(np.average(res["cov80_ps"], weights=wts))
        res["cov80_pool_all"] = float(np.average(res["cov80_pool"], weights=wts))
        res["passes"] = bool(sum(x > 0 for x in res["fold_gain"]) >= 2 and res["lo"] > 0 and 0.77 <= res["cov80_ps_all"] <= 0.83)
    else:
        res["passes"] = False
    return res
