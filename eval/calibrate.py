"""Predictive distribution, width fix and isotonic correction (Phase 6.3 and the distribution fields of Phase 5.1).

Rules: docs/overnight_plan.md, "Prediction schedule, calibration and grading". Development seasons 2020-2024 only (any other season raises); no sportsbook line is read;
the mean of a prediction is never changed here, only its distribution is described.

For a prediction p of market m at (season, week): the pool is every walk-forward row of that market from a STRICTLY EARLIER (season, week); the K rows nearest to p in
prediction rank are a sample of what the real number did next to a prediction like p. Its quantiles are the stored quantiles, its tail shares are the ladder.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

import config

QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QCOLS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
LEVELS = {0.5: ("q25", "q75"), 0.8: ("q10", "q90"), 0.9: ("q05", "q95")}
GRID = np.round(np.arange(0.70, 1.5001, 0.01), 2)
LAST = config.HOLDOUT_SEASON - 1
FOLDS = (2022, 2023, 2024)
MIN_FOLDS = 2

RUNGS = {"pass_att": (20, 25, 30, 35, 40), "pass_cmp": (15, 20, 25, 30), "pass_yds": (150, 200, 250, 300, 350), "qb_rush_att": (2, 4, 6, 8), "qb_rush_yds": (10, 20, 30, 50),
         "rush_att": (5, 10, 15, 20, 25), "rush_yds": (25, 50, 75, 100, 125), "targets": (2, 4, 6, 8, 10), "rec": (2, 4, 6, 8), "rec_yds": (25, 50, 75, 100, 125),
         "spread": (), "total": (35, 40, 45, 50, 55)}
K = {m: 600 for m in RUNGS} | {"qb_rush_att": 250, "qb_rush_yds": 250, "spread": 200, "total": 200}
MIN_POOL = {m: 1500 for m in RUNGS} | {"spread": 400, "total": 400}


@dataclass
class Config:
    k: dict = field(default_factory=lambda: dict(K))
    min_pool: dict = field(default_factory=lambda: dict(MIN_POOL))
    rungs: dict = field(default_factory=lambda: {m: tuple(v) for m, v in RUNGS.items()})


def _check_seasons(rows: pl.DataFrame) -> None:
    if rows.height and rows["season"].max() > LAST:
        raise ValueError(f"calibration reads seasons <= {LAST} only")


def _window(sorted_pred: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    """(len(p), k) row positions of the k pool rows nearest to each p in prediction rank."""
    n = len(sorted_pred)
    i = np.searchsorted(sorted_pred, p)
    lo = np.clip(i - k // 2, 0, max(n - k, 0))
    return lo[:, None] + np.arange(min(k, n))[None, :]


def _describe(sample: np.ndarray, rungs: tuple) -> dict:
    out = dict(zip(QCOLS, np.quantile(sample, QS, axis=1, method="linear")))
    for t in rungs:
        out[f"p_ge_{t}"] = (sample >= t).mean(axis=1)
    return out


def walk_forward(rows: pl.DataFrame, market: str, cfg: Config) -> pl.DataFrame:
    """One row per (season, week, entity) with a distribution: quantiles and tail shares from the strictly earlier pool. Rows in a week whose pool is below min_pool get none."""
    _check_seasons(rows)
    d = rows.filter(pl.col("market") == market).drop_nulls(["prediction", "actual"]).sort("season", "week", "entity")
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
        tgt = sw == key
        idx = _window(sp, pred[tgt], cfg.k[market])
        desc = _describe(sa[idx], cfg.rungs[market])
        out.append(d.filter(pl.col("season") == s).filter(pl.col("week") == w).with_columns([pl.Series(n, v) for n, v in desc.items()]))
    return pl.concat(out) if out else d.head(0)


def _scaled(df: pl.DataFrame, c: float) -> dict:
    med = df["q50"].to_numpy()
    return {q: med + c * (df[q].to_numpy() - med) for q in QCOLS}


def coverage(df: pl.DataFrame, c: float = 1.0) -> dict:
    """Share of actuals inside each central interval (50, 80, 90 %) after scaling the spread by c."""
    q = _scaled(df, c)
    a = df["actual"].to_numpy()
    return {lvl: float(((a >= q[lo]) & (a <= q[hi])).mean()) for lvl, (lo, hi) in LEVELS.items()}


def coverage_error(df: pl.DataFrame, c: float = 1.0) -> float:
    cov = coverage(df, c)
    return float(sum(abs(cov[l] - l) for l in LEVELS))


def fit_width(df: pl.DataFrame) -> float:
    errs = [coverage_error(df, c) for c in GRID]
    return float(GRID[int(np.argmin(errs))])


def use_fix(with_fix, without_fix, min_folds: int = MIN_FOLDS) -> bool:
    """Apply a correction only if it lowers the error in at least `min_folds` folds."""
    return int(sum(w < wo for w, wo in zip(with_fix, without_fix))) >= min_folds


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def fit_isotonic(p: np.ndarray, y: np.ndarray):
    """A monotone map from a ladder probability to the observed share; returns a callable."""
    from sklearn.isotonic import IsotonicRegression
    m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(p, float), np.asarray(y, float))
    return lambda x: m.predict(np.asarray(x, float))


def _pairs(df: pl.DataFrame, rungs: tuple) -> tuple:
    p = np.concatenate([df[f"p_ge_{t}"].to_numpy() for t in rungs]) if rungs else np.array([])
    y = np.concatenate([(df["actual"].to_numpy() >= t).astype(float) for t in rungs]) if rungs else np.array([])
    return p, y


def evaluate(wf: pl.DataFrame, rungs: tuple) -> dict:
    """The plan's fold check for one market: width fix and isotonic, fitted before each fold and scored on it."""
    res = dict(cov_before=[], cov_after=[], c=[], brier_before=[], brier_after=[], folds=[])
    for f in FOLDS:
        tr, te = wf.filter(pl.col("season") < f), wf.filter(pl.col("season") == f)
        if tr.height < 300 or te.height < 100:
            continue
        c = fit_width(tr)
        res["folds"].append(f)
        res["c"].append(c)
        res["cov_before"].append(coverage_error(te, 1.0))
        res["cov_after"].append(coverage_error(te, c))
        if rungs:
            ptr, ytr = _pairs(tr, rungs)
            pte, yte = _pairs(te, rungs)
            iso = fit_isotonic(ptr, ytr)
            res["brier_before"].append(brier(pte, yte))
            res["brier_after"].append(brier(iso(pte), yte))
    res["use_width"] = use_fix(res["cov_after"], res["cov_before"]) if res["folds"] else False
    res["use_isotonic"] = bool(rungs) and use_fix(res["brier_after"], res["brier_before"]) if res["folds"] else False
    return res


class Calibrator:
    """Per market: the pool (prediction, actual rows), the width scale c, the isotonic map (breakpoints) and flags. Describes a distribution; never changes a mean."""

    def __init__(self, markets: dict, cfg: Config):
        self.markets, self.cfg = markets, cfg

    @property
    def hash(self) -> str:
        blob = json.dumps({m: dict(n=len(v["pred"]), sp=float(np.sum(v["pred"])), sa=float(np.sum(v["act"])), c=v["c"], iso=v["iso"], w=v["use_width"], i=v["use_isotonic"])
                           for m, v in sorted(self.markets.items())}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def distribution(self, market: str, p: float):
        v = self.markets.get(market)
        if v is None or len(v["pred"]) < self.cfg.min_pool[market]:
            return None
        sp, sa = v["pred"], v["act"]
        idx = _window(sp, np.array([p]), self.cfg.k[market])
        sample = sa[idx]
        desc = _describe(sample, self.cfg.rungs[market])
        c = v["c"] if v["use_width"] else 1.0
        med = float(desc["q50"][0])
        out = {"median": med, **{q: med + c * (float(desc[q][0]) - med) for q in QCOLS}}
        lad = {}
        iso = None
        if v["use_isotonic"] and v["iso"]:
            xs, ys = np.array(v["iso"][0]), np.array(v["iso"][1])
            iso = lambda x: float(np.interp(x, xs, ys))
        for t in self.cfg.rungs[market]:
            pr = float(desc[f"p_ge_{t}"][0])
            lad[str(t)] = iso(pr) if iso else pr
        out["ladder"] = lad
        return out

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {m: dict(c=v["c"], iso=v["iso"], use_width=v["use_width"], use_isotonic=v["use_isotonic"], eval=v["eval"]) for m, v in self.markets.items()}
        (path.with_suffix(".json")).write_text(json.dumps(dict(meta=meta, hash=self.hash, k=self.cfg.k, min_pool=self.cfg.min_pool, rungs={m: list(r) for m, r in self.cfg.rungs.items()}), sort_keys=True, indent=1))
        pl.concat([pl.DataFrame(dict(market=m, prediction=v["pred"], actual=v["act"])) for m, v in sorted(self.markets.items())]).write_parquet(path.with_suffix(".parquet"))

    @classmethod
    def load(cls, path) -> "Calibrator":
        path = Path(path)
        j = json.loads(path.with_suffix(".json").read_text())
        pool = pl.read_parquet(path.with_suffix(".parquet"))
        cfg = Config(k=j["k"], min_pool=j["min_pool"], rungs={m: tuple(r) for m, r in j["rungs"].items()})
        markets = {}
        for m, meta in j["meta"].items():
            d = pool.filter(pl.col("market") == m)
            markets[m] = dict(pred=d["prediction"].to_numpy(), act=d["actual"].to_numpy(), c=meta["c"], iso=meta["iso"], use_width=meta["use_width"], use_isotonic=meta["use_isotonic"], eval=meta["eval"])
        return cls(markets, cfg)


def build(rows: pl.DataFrame, cfg: Config | None = None, markets=None) -> Calibrator:
    """Fit the live calibrator on ALL given rows (2020-2024): pool, width scale and isotonic map, each kept only if the fold check says it helps."""
    cfg = cfg or Config()
    _check_seasons(rows)
    out = {}
    for m in (markets or list(cfg.rungs)):
        d = rows.filter(pl.col("market") == m).drop_nulls(["prediction", "actual"]).sort("season", "week", "entity")
        if d.height < cfg.min_pool[m]:
            out[m] = dict(pred=np.array([]), act=np.array([]), c=1.0, iso=None, use_width=False, use_isotonic=False, eval={})
            continue
        order = np.argsort(d["prediction"].to_numpy(), kind="stable")
        pred, act = d["prediction"].to_numpy()[order], d["actual"].to_numpy()[order]
        wf = walk_forward(rows, m, cfg)
        ev = evaluate(wf, cfg.rungs[m]) if wf.height else dict(use_width=False, use_isotonic=False)
        c = fit_width(wf) if (wf.height and ev["use_width"]) else 1.0
        iso = None
        if wf.height and ev["use_isotonic"]:
            from sklearn.isotonic import IsotonicRegression
            p, y = _pairs(wf, cfg.rungs[m])
            r = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
            iso = [r.X_thresholds_.tolist(), r.y_thresholds_.tolist()]
        out[m] = dict(pred=pred, act=act, c=c, iso=iso, use_width=bool(ev["use_width"]), use_isotonic=bool(ev["use_isotonic"]),
                      eval={k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in ev.items()})
    return Calibrator(out, cfg)
