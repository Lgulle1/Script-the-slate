"""Grading functions: MAE, Brier at each ladder rung, log loss, calibration, whole-curve (PIT)
check, and skill versus a baseline. All are pure functions of arrays; `predictive_scores`
builds a walk-forward predictive distribution for point-prediction baselines.

Predictive distribution (point baselines only): within each (method, market), every row of
week W gets the empirical distribution of that method's own residuals (actual - prediction)
from weeks BEFORE W, shifted to the row's prediction. Residuals are pooled within terciles of
earlier predicted values (small predictions have smaller spread), falling back to all earlier
residuals when a tercile is thin. Nothing from week W or later is used, and a method gets no
distribution until it has config.MIN_RESIDUALS earlier residuals.
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config

EPS = 1e-6  # clip for log loss


# ---------------------------------------------------------------- scalar metrics
def mean_absolute_error(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def brier_score(p, outcome) -> float:
    p, o = np.asarray(p, float), np.asarray(outcome, float)
    return float(np.mean((p - o) ** 2))


def log_loss(p, outcome, eps: float = EPS) -> float:
    p, o = np.clip(np.asarray(p, float), eps, 1 - eps), np.asarray(outcome, float)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def skill_vs_baseline(model_brier: float, baseline_brier: float) -> float:
    """1 - model Brier / baseline Brier: 0 = no better than the baseline, 1 = perfect."""
    return 1.0 - model_brier / baseline_brier


def calibration_table(p, outcome, width: float = config.CALIBRATION_BAND_WIDTH) -> pl.DataFrame:
    """Bucket stated probabilities into bands of `width` (20% by default) and compare to the
    actual frequency. Columns: band, lo, hi, n, mean_predicted, actual_frequency."""
    p, o = np.asarray(p, float), np.asarray(outcome, float)
    k = int(round(1 / width))
    idx = np.minimum((p / width).astype(int), k - 1)  # p == 1.0 falls in the top band
    rows = []
    for b in range(k):
        m = idx == b
        lo, hi = b * width, (b + 1) * width
        rows.append(dict(band=f"{lo:.1f}-{hi:.1f}", lo=lo, hi=hi, n=int(m.sum()),
                         mean_predicted=float(p[m].mean()) if m.any() else None,
                         actual_frequency=float(o[m].mean()) if m.any() else None))
    return pl.DataFrame(rows, schema={"band": pl.String, "lo": pl.Float64, "hi": pl.Float64, "n": pl.Int64,
                                      "mean_predicted": pl.Float64, "actual_frequency": pl.Float64})


def pit_histogram(pit, bins: int = 10) -> np.ndarray:
    """Share of PIT values in each of `bins` equal bins. Flat (all ~1/bins) = well calibrated."""
    counts, _ = np.histogram(np.asarray(pit, float), bins=bins, range=(0.0, 1.0))
    return counts / counts.sum()


def pit_flatness(pit, bins: int = 10) -> dict:
    """max_dev: largest |bin share - 1/bins|. ks: Kolmogorov-Smirnov distance to Uniform(0,1)."""
    pit = np.sort(np.asarray(pit, float))
    n = len(pit)
    cdf = np.arange(1, n + 1) / n
    ks = max(np.max(cdf - pit), np.max(pit - (cdf - 1 / n)))
    return {"max_dev": float(np.max(np.abs(pit_histogram(pit, bins) - 1 / bins))), "ks": float(ks)}


# ---------------------------------------------------------------- predictive distribution
def predictive_scores(preds: pl.DataFrame, ladder, seed: int = config.PIT_JITTER_SEED) -> pl.DataFrame:
    """Walk-forward predictive probabilities for ONE (method, market) of point predictions.

    `preds` needs columns season, week, entity, prediction, actual. Returns the rows that have a
    distribution, with: pit (outcome jittered by U(-.5,.5) because outcomes are integers, then
    located in the predictive distribution), and one p_over_<rung> column per ladder rung
    (P(actual > rung)).
    """
    d = preds.filter(pl.col("prediction").is_not_null()).sort("season", "week", "entity")
    rng = np.random.default_rng(seed)
    hist_p, hist_r = np.array([]), np.array([])
    out = []
    for (season, week), g in d.group_by(["season", "week"], maintain_order=True):
        p, y = g["prediction"].to_numpy(), g["actual"].to_numpy()
        if len(hist_p) >= config.MIN_RESIDUALS:
            edges = np.quantile(hist_p, np.linspace(0, 1, config.N_STRATA + 1)[1:-1])
            h_strat, g_strat = np.digitize(hist_p, edges), np.digitize(p, edges)
            pools = {}
            for s in range(config.N_STRATA):
                r = hist_r[h_strat == s]
                pools[s] = np.sort(r if len(r) >= config.MIN_STRATUM_RESIDUALS else hist_r)
            jitter = rng.uniform(-0.5, 0.5, size=len(p))
            cols = {"pit": np.empty(len(p))} | {f"p_over_{t}": np.empty(len(p)) for t in ladder}
            for s in range(config.N_STRATA):
                m = g_strat == s
                if not m.any():
                    continue
                pool, n = pools[s], len(pools[s])
                cols["pit"][m] = (np.searchsorted(pool, y[m] + jitter[m] - p[m], side="right") + 0.5) / (n + 1)
                for t in ladder:
                    greater = n - np.searchsorted(pool, t - p[m], side="right")
                    cols[f"p_over_{t}"][m] = (greater + 0.5) / (n + 1)
            out.append(g.with_columns([pl.Series(k, v) for k, v in cols.items()]))
        hist_p, hist_r = np.concatenate([hist_p, p]), np.concatenate([hist_r, y - p])  # week W joins history only now
    if not out:
        return d.head(0).with_columns([pl.lit(None, dtype=pl.Float64).alias(c)
                                       for c in ["pit"] + [f"p_over_{t}" for t in ladder]])
    return pl.concat(out)


def moneyline_scores(preds: pl.DataFrame) -> pl.DataFrame:
    """Moneyline baselines state P(home win) directly; no residual distribution needed."""
    return (preds.filter(pl.col("prediction").is_not_null())
            .with_columns(p_over_win=pl.col("prediction").clip(EPS, 1 - EPS)))


# ---------------------------------------------------------------- whole-backtest grading
def grade_backtest(preds: pl.DataFrame, reference: str = config.SKILL_REFERENCE_METHOD) -> pl.DataFrame:
    """Grade every (method, market) in a walk-forward prediction table.

    All methods are graded on the SAME rows (the ones every method could predict and give a
    distribution for), so the comparison is like-for-like; `coverage` reports how many of each
    market's target rows a method could predict at all. Returns a long table:
    method, market, metric, key, value, n  with metrics
      mae                      (point markets only)
      brier / logloss          key = rung (moneyline: "win") and "mean" across rungs
      calib_pred / calib_actual key = 20% band, n = rows in band (all rungs pooled)
      pit                      key = decile 0..9 (share of PIT values), plus pit_max_dev, pit_ks
      skill_brier              key = rung / "mean", 1 - Brier / reference-method Brier
      coverage                 share of target rows with a point prediction
    """
    out = []
    methods = sorted(preds["method"].unique().to_list())
    for market in sorted(preds["market"].unique().to_list()):
        mp = preds.filter(pl.col("market") == market)
        is_ml = market == "moneyline"
        rungs = ["win"] if is_ml else list(config.LADDERS[market])
        scores = {}
        for m in methods:
            sub = mp.filter(pl.col("method") == m)
            n_all = sub.height
            out.append(dict(method=m, market=market, metric="coverage", key="",
                            value=sub["prediction"].is_not_null().sum() / n_all if n_all else None, n=n_all))
            scores[m] = moneyline_scores(sub) if is_ml else predictive_scores(sub, rungs)
        keys = None
        for s in scores.values():
            k = set(zip(s["season"].to_list(), s["week"].to_list(), s["entity"].to_list()))
            keys = k if keys is None else keys & k
        brier_by = {}
        for m in methods:
            s = scores[m].filter(pl.struct("season", "week", "entity").map_elements(
                lambda r: (r["season"], r["week"], r["entity"]) in keys, return_dtype=pl.Boolean))
            s = s.sort("season", "week", "entity")
            n = s.height
            y = s["actual"].to_numpy()
            if not is_ml:
                out.append(dict(method=m, market=market, metric="mae", key="", value=mean_absolute_error(y, s["prediction"]), n=n))
                f = pit_flatness(s["pit"].to_numpy())
                out += [dict(method=m, market=market, metric="pit_max_dev", key="", value=f["max_dev"], n=n),
                        dict(method=m, market=market, metric="pit_ks", key="", value=f["ks"], n=n)]
                for i, share in enumerate(pit_histogram(s["pit"].to_numpy())):
                    out.append(dict(method=m, market=market, metric="pit", key=str(i), value=float(share), n=n))
            all_p, all_o, bs, ls = [], [], [], []
            for r in rungs:
                p = s["p_over_win" if is_ml else f"p_over_{r}"].to_numpy()
                o = y if is_ml else (y > r).astype(float)
                bs.append(brier_score(p, o)); ls.append(log_loss(p, o))
                out += [dict(method=m, market=market, metric="brier", key=str(r), value=bs[-1], n=n),
                        dict(method=m, market=market, metric="logloss", key=str(r), value=ls[-1], n=n)]
                all_p.append(p); all_o.append(o)
            out += [dict(method=m, market=market, metric="brier", key="mean", value=float(np.mean(bs)), n=n),
                    dict(method=m, market=market, metric="logloss", key="mean", value=float(np.mean(ls)), n=n)]
            cal = calibration_table(np.concatenate(all_p), np.concatenate(all_o))
            for r in cal.iter_rows(named=True):
                out += [dict(method=m, market=market, metric="calib_pred", key=r["band"], value=r["mean_predicted"], n=r["n"]),
                        dict(method=m, market=market, metric="calib_actual", key=r["band"], value=r["actual_frequency"], n=r["n"])]
            brier_by[m] = {**{str(r): b for r, b in zip(rungs, bs)}, "mean": float(np.mean(bs))}
        if reference in brier_by:
            for m in methods:
                for key, b in brier_by[m].items():
                    out.append(dict(method=m, market=market, metric="skill_brier", key=key,
                                    value=skill_vs_baseline(b, brier_by[reference][key]), n=len(keys)))
    schema = {"method": pl.String, "market": pl.String, "metric": pl.String, "key": pl.String, "value": pl.Float64, "n": pl.Int64}
    df = pl.DataFrame(out, schema=schema).with_columns(pl.col("value").round(10))
    return df.sort("market", "method", "metric", "key")
