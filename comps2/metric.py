"""Learn what "similar" means, per search and market (yours: "the model learns the similarity weights itself").

For pairs (i, j) a search would compare (j earlier than i, in i's candidate pool), y = r_i * r_j with r the standardized residual. If two games are
similar in the features that matter, their residuals move together, so y falls as the difference in those features grows:
    y = a - sum_k beta_k * (z_ik - z_jk)^2,   beta_k >= 0   (non-negative least squares on centered columns)
beta is the feature weight in the kernel's distance. Fit on target rows of FIT_SEASONS only; the weights are then used unchanged later.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import nnls

from comps2 import archetypes as AR
from comps2 import searches as S

MAX_PAIRS_PER_TARGET = 25
# the families a market's weights are learned on (their archetype columns are identical); other families of the market reuse the unit weights
FAMS = dict(rush_att=("RB",), rush_yds=("RB",), targets=("WR", "TE"), rec=("WR", "TE"), rec_yds=("WR", "TE"), pass_att=("QB",), pass_cmp=("QB",),
            pass_yds=("QB",), qb_rush_att=("QB",), qb_rush_yds=("QB",))
RNG = np.random.default_rng(7)


def _std_resid(e):
    return e.with_columns(r=((pl.col("relc") - pl.col("relc").mean().over("market", "season")) / pl.col("relc").std().over("market", "season")))


def pairs(e: pl.DataFrame, search: str, market: str, fit_seasons, window_days=730):
    e = _std_resid(e)
    ostats, dstats = S.MARKET_UNITS[market]
    x = e.filter(pl.col("market") == market)
    out_X, out_y, cols = [], [], None
    for fam in FAMS[market]:
        xf = x.filter(pl.col("family") == fam)
        arch = [f"a_{s}" for s in AR.FEATS.get(fam, ())]
        if search == "own":
            feats, key = [f"d_{s}" for s in dstats], "player_id"
        elif search == "dfn":
            feats, key = [f"o_{s}" for s in ostats] + arch, "opponent"
        else:
            feats, key = [f"o_{s}" for s in ostats] + [f"d_{s}" for s in dstats] + arch, None
        cols = cols or feats
        Z = xf.select(feats).to_numpy().astype(float)
        r, gd, season = xf["r"].to_numpy(), xf["gd"].to_numpy(), xf["season"].to_numpy()
        groups = S._groups(xf[key].to_numpy()) if key else {None: np.arange(xf.height)}
        for _, idx in groups.items():
            for a in idx:
                if season[a] not in fit_seasons:
                    continue
                past = idx[(gd[idx] < gd[a]) & (gd[idx] >= gd[a] - window_days)]
                if not len(past):
                    continue
                if len(past) > MAX_PAIRS_PER_TARGET:
                    past = RNG.choice(past, MAX_PAIRS_PER_TARGET, replace=False)
                D = (Z[past] - Z[a]) ** 2
                out_X.append(D)
                out_y.append(r[past] * r[a])
    X, y = np.vstack(out_X), np.concatenate(out_y)
    with np.errstate(all="ignore"):
        X = np.where(np.isnan(X), np.nan_to_num(np.nanmean(X, 0)), X)
    return X, y, cols


def learn(e, search, market, fit_seasons):
    X, y, cols = pairs(e, search, market, fit_seasons)
    Xc, yc = X - X.mean(0), y - y.mean()
    beta, _ = nnls(-Xc, yc)
    # strength of the learned similarity: y explained, vs pairs drawn at random from the same pool (shuffle y)
    pred = -Xc @ beta
    r2 = 1 - ((yc - pred) ** 2).sum() / (yc ** 2).sum()
    yb = RNG.permutation(yc)
    bb, _ = nnls(-Xc, yb)
    r2_null = 1 - ((yb + Xc @ bb) ** 2).sum() / (yb ** 2).sum()
    return dict(zip(cols, beta)), dict(n_pairs=len(y), r2=float(r2), r2_shuffled=float(r2_null))
