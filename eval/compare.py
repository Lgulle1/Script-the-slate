"""Model-vs-best-baseline comparison: paired gains with bootstrap confidence intervals.

Gain = baseline loss - model loss, so POSITIVE means the model is better. Loss is absolute error for
point markets and the Brier score for moneyline. Everything is paired row by row on the SAME rows.

Two intervals are reported:
  cluster  resamples whole (season, week) slates with replacement -- the primary interval, because
           rows in one week share weather, schedule and slate-wide effects and are not independent;
  iid      resamples individual rows -- narrower, shown for reference only.
Both are percentile intervals from config.BOOTSTRAP_RESAMPLES resamples with a fixed seed, so
re-running gives identical numbers.
"""
from __future__ import annotations

import numpy as np

import config


def _ci(samples: np.ndarray, level: float):
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def cluster_bootstrap(diff, cluster, n_boot=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED,
                      level=config.BOOTSTRAP_CI):
    """CI for mean(diff), resampling clusters (e.g. season-weeks) with replacement."""
    diff, cluster = np.asarray(diff, float), np.asarray(cluster)
    ids, inv = np.unique(cluster, return_inverse=True)
    s, n = np.bincount(inv, weights=diff), np.bincount(inv).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
    return _ci(s[pick].sum(1) / n[pick].sum(1), level)


def iid_bootstrap(diff, n_boot=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, level=config.BOOTSTRAP_CI,
                  chunk=500):
    """CI for mean(diff), resampling individual rows."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    means = np.concatenate([diff[rng.integers(0, len(diff), size=(min(chunk, n_boot - i), len(diff)))].mean(1)
                            for i in range(0, n_boot, chunk)])
    return _ci(means, level)


def verdict(gain: float, ci_lo: float, ci_hi: float, seasons_won: int, min_seasons=config.MIN_SEASONS_WON) -> str:
    """CLEARS: interval excludes zero on the good side AND the model wins in >= min_seasons seasons separately.
    EDGE: positive gain but the interval touches/crosses zero (or too few seasons) -- flag, do not ship quietly.
    NO: no gain, or the model is worse."""
    if gain > 0 and ci_lo > 0 and seasons_won >= min_seasons:
        return "CLEARS"
    if gain > 0 and ci_hi > 0:
        return "EDGE"
    return "NO"


# ---------------------------------------------------------------- whole-backtest comparison
import polars as pl  # noqa: E402

BASELINE_METHODS = ("last3", "season_avg", "recency", "blend_70_30", "role_avg")
KEY = ["season", "week", "entity"]


def _loss(market: str, pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return (pred - actual) ** 2 if market == "moneyline" else np.abs(pred - actual)


def compare_market(market: str, baseline_preds: pl.DataFrame, model_preds: pl.DataFrame) -> dict:
    """Compare one market's model predictions with its best baseline, on rows where the model and all
    five baselines have a prediction. Loss: absolute error (point markets) / Brier (moneyline).

    The best baseline is the one with the lowest pooled loss on those rows -- picking it with the test
    rows can only make the baseline look better, so the reported gain is conservative.
    Returns {"summary": {...}, "by_season": [{...}, ...]}.
    """
    base = (baseline_preds.filter(pl.col("market") == market)
            .pivot(on="method", index=[*KEY, "actual"], values="prediction"))
    model = model_preds.filter(pl.col("market") == market).select(*KEY, model=pl.col("prediction"))
    t = base.join(model, on=KEY, how="inner").drop_nulls([*BASELINE_METHODS, "model"]).sort(KEY)
    y, m = t["actual"].to_numpy(), t["model"].to_numpy()
    losses = {b: _loss(market, t[b].to_numpy(), y) for b in BASELINE_METHODS}
    best = min(losses, key=lambda b: losses[b].mean())
    lm = _loss(market, m, y)
    diff = losses[best] - lm
    cluster = (t["season"].to_numpy() * 100 + t["week"].to_numpy())
    seasons = t["season"].to_numpy()

    def block(mask):
        d, c = diff[mask], cluster[mask]
        gain = float(d.mean())
        lo, hi = cluster_bootstrap(d, c)
        return dict(n=int(mask.sum()), model_loss=float(lm[mask].mean()), best_loss=float(losses[best][mask].mean()),
                    gain=gain, gain_pct=100 * gain / float(losses[best][mask].mean()), ci_lo=lo, ci_hi=hi)

    by_season = []
    for s in sorted(set(seasons.tolist())):
        by_season.append({"season": int(s), **block(seasons == s)})
    full = block(np.ones(len(t), bool))
    full["ci_lo_iid"], full["ci_hi_iid"] = iid_bootstrap(diff)
    full["best_baseline"] = best
    full["metric"] = "brier" if market == "moneyline" else "mae"
    full["seasons_won"] = sum(1 for b in by_season if b["gain"] > 0)
    full["seasons_total"] = len(by_season)
    full["model_bias"] = float((m - y).mean()) if market != "moneyline" else None
    full["best_bias"] = float((t[best].to_numpy() - y).mean()) if market != "moneyline" else None
    full["baseline_losses"] = {b: float(v.mean()) for b, v in losses.items()}
    full["verdict"] = verdict(full["gain"], full["ci_lo"], full["ci_hi"], full["seasons_won"])
    return {"summary": full, "by_season": by_season}
