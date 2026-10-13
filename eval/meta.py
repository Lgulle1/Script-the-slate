"""Phase 6.1 stage-1 shrinkage blend and 6.2 stage-2 stacking (rules: docs/overnight_plan.md, "Phase 7 app and Phase 6.1 / 6.2").

Input rows (one per player-game, walk-forward): market, season, week, entity, actual, model (the Phase 3 prediction), baseline (the market's best simple baseline) and, for stage 2, extra feature columns.
Development seasons 2020-2024 only; every weight and model is fit on strictly earlier weeks (stage 1) or earlier seasons (stage 2).
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config

GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
K_SHRINK = 500
LAST = config.HOLDOUT_SEASON - 1


def blend(model: np.ndarray, baseline: np.ndarray, a: float) -> np.ndarray:
    return a * model + (1.0 - a) * baseline


def shrink(a_fit: float, n: int, k: int = K_SHRINK) -> float:
    return float(min(max(a_fit * n / (n + k), 0.0), 1.0))


def _fit_a(model: np.ndarray, baseline: np.ndarray, actual: np.ndarray) -> float:
    errs = [np.abs(blend(model, baseline, a) - actual).mean() for a in GRID]
    return float(GRID[int(np.argmin(errs))])


def stage1(rows: pl.DataFrame, market: str) -> pl.DataFrame:
    """Per row: a_fit (MAE-minimising weight on all earlier rows), a_used (shrunk) and the blended prediction. Weeks without earlier rows use a_used = 0 (the baseline)."""
    if rows["season"].max() > LAST:
        raise ValueError(f"meta reads seasons <= {LAST} only")
    d = rows.filter(pl.col("market") == market).drop_nulls(["model", "baseline", "actual"]).sort("season", "week", "entity")
    m, b, y = d["model"].to_numpy(), d["baseline"].to_numpy(), d["actual"].to_numpy()
    sw = (d["season"] * 100 + d["week"]).to_numpy()
    a_fit, a_used = np.zeros(len(d)), np.zeros(len(d))
    for key in np.unique(sw):
        tgt, prior = sw == key, sw < key
        n = int(prior.sum())
        if n == 0:
            continue
        af = _fit_a(m[prior], b[prior], y[prior])
        a_fit[tgt], a_used[tgt] = af, shrink(af, n)
    return d.with_columns(a_fit=pl.Series(a_fit), a_used=pl.Series(a_used), final=pl.Series(a_used * m + (1 - a_used) * b))


def stage2_active(rows: pl.DataFrame, market: str, min_games: int = config.STAGE2_MIN_GAMES) -> bool:
    return rows.filter(pl.col("market") == market).drop_nulls(["model", "baseline", "actual"]).height >= min_games


def stage2(rows: pl.DataFrame, market: str, min_games: int = config.STAGE2_MIN_GAMES, features: list | None = None):
    """Walk-forward by season: for each season with earlier training rows, a LightGBM (L1) fit on all earlier seasons predicts that season. None below the activation threshold."""
    if not stage2_active(rows, market, min_games):
        return None
    import lightgbm as lgb
    d = rows.filter(pl.col("market") == market).drop_nulls(["model", "baseline", "actual"]).sort("season", "week", "entity")
    feats = features or ["model", "baseline"]
    parts = []
    for s in sorted(d["season"].unique().to_list()):
        tr, te = d.filter(pl.col("season") < s), d.filter(pl.col("season") == s)
        if tr.height < 300 or te.height == 0:
            continue
        mdl = lgb.LGBMRegressor(objective="l1", n_estimators=150, learning_rate=0.05, num_leaves=8, min_child_samples=40, random_state=0, n_jobs=1, verbose=-1)
        mdl.fit(tr.select(feats).to_pandas(), tr["actual"].to_numpy())
        parts.append(te.with_columns(pred=pl.Series(mdl.predict(te.select(feats).to_pandas()))))
    return pl.concat(parts) if parts else None
