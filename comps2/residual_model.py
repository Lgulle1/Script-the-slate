"""The matchup model on residuals: where does the projection stack go wrong, given everything the comps know?

Target: relc (actual vs the comp-free expectation, a fraction, clipped). Inputs per player-game row:
  - tonight's OFFENSE fingerprint (all unit stats, z) and the opponent's DEFENSE fingerprint (z), and their differences (offense minus defense on the
    same stat: the "mismatch" columns of the interaction idea),
  - the player's archetype (z, within his family),
  - the situation (vacated carry / target share, own shares, injury status, played hurt last game),
  - the comp components (own, dfn, int, sit: shift and effective sample size), the expectation itself, market and family.
LightGBM, walk-forward by season: the prediction for season s is made by a model trained on seasons < s only (2021 has no earlier comps -> none).
Output: comps2_residual_preds.parquet (rid, season, pred) and the practice ladder in records/tuning/comps2_ladder.csv (comp = "model").
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

import config
from comps2 import archetypes as AR
from comps2 import evaluate as V
from comps2 import fingerprints as FP
from comps2 import searches as S
from comps2 import situations as SI

OUT = config.PROCESSED_DIR / "comps2_residual_preds.parquet"
PARAMS = dict(objective="huber", alpha=0.6, learning_rate=0.03, n_estimators=400, num_leaves=15, min_child_samples=300, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7)
MARKETS = list(S.MARKET_UNITS)
FAMILIES = ["RB", "WR", "TE", "QB"]


def frame() -> pl.DataFrame:
    e = S.load()
    c = pl.read_parquet(config.PROCESSED_DIR / "comps2_components.parquet").select(
        "rid", *[f"{s}_{k}" for s in ("own", "dfn", "int") for k in ("shift", "neff")], "sit_shift", "sit_neff", *SI.SCALE)
    d = e.join(c, on="rid", how="left")
    mism = [(pl.col(f"o_{s}") - pl.col(f"d_{s}")).alias(f"m_{s}") for s in FP.STATS]
    return d.with_columns(mism).with_columns(mk=pl.col("market").replace_strict({m: i for i, m in enumerate(MARKETS)}, default=-1),
                                             fm=pl.col("family").replace_strict({f: i for i, f in enumerate(FAMILIES)}, default=-1),
                                             lexp=pl.col("expected").clip(0.1, None).log())


def locked_columns() -> list:
    """The structure fixed on the practice seasons 2022-2023 (2026-10-19, records/tuning/comps2_ladder.csv) BEFORE the exam: the search outputs and the
    situation only. Adding the raw fingerprints, archetypes and mismatch columns made practice worse in every group (rush 0.054 vs 0.089)."""
    return [f"{s}_{k}" for s in ("own", "dfn", "int", "sit") for k in ("shift", "neff")] + list(SI.SCALE) + ["mk", "fm", "lexp"]


def columns(d: pl.DataFrame) -> list:
    return ([f"o_{s}" for s in FP.STATS] + [f"d_{s}" for s in FP.STATS] + [f"m_{s}" for s in FP.STATS] + [f"a_{s}" for s in AR.ALL]
            + list(SI.SCALE) + [f"{s}_{k}" for s in ("own", "dfn", "int", "sit") for k in ("shift", "neff")] + ["mk", "fm", "lexp"])


def walk_forward(d: pl.DataFrame, seasons, params=PARAMS, cols=None) -> pl.DataFrame:
    cols = cols or locked_columns()
    out = []
    for s in seasons:
        tr = d.filter((pl.col("season") < s) & (pl.col("season") >= 2021))
        te = d.filter(pl.col("season") == s)
        if tr.height < 1000:
            continue
        m = lgb.LGBMRegressor(**params)
        m.fit(tr.select(cols).to_numpy(), tr["relc"].to_numpy(), categorical_feature=[cols.index("mk"), cols.index("fm")])
        out.append(te.select("rid", "season").with_columns(pred=m.predict(te.select(cols).to_numpy())))
    return pl.concat(out)


def practice(params=PARAMS, label="model", cols=None):
    d = frame()
    p = walk_forward(d, (2022, 2023), params, cols)
    c = d.select("rid", "season", "market", "expected", "actual", "relc").join(p.rename({"pred": "model_shift"}), on="rid")
    lad = V.ladder(c, ("model",), (2022, 2023)).with_columns(label=pl.lit(label))
    return c, lad


if __name__ == "__main__":
    c, lad = practice()
    print(lad)
