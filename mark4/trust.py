"""Mark 4j: learn, line by line, how much to trust us vs FanDuel (2026-10-20).

Mark 4i trusts the book the same amount on every line of a season. The Moneyball question is WHERE the book is weaker than us. Mark 4j is a stacking
model on the chance scale: it starts from FanDuel's no-vig chance and learns how far to move toward Mark 3 and toward Mark 4h, and how that trust
changes with the market, the situation (promoted back / receiver, QB change: game 1-3 of the change or later), how much history the player has, whether
the comps agree, how far the price moved, and how big the disagreement is. Also a per-market correction of the book itself (e.g. overs priced too high).

  logit p = a * logit(book) + sum_g [ b3_g * (logit Mark 3 - logit book) + b4_g * (logit Mark 4h - logit book) ] * group_g + c_market

Forward like everything else: each 6-week block is fitted only on lines graded before it (recency-weighted, half-life one season); the inputs are the
forward predictions of Mark 3 and Mark 4h, which are themselves out of sample. Ridge penalty keeps it from chasing small groups.

  python -m mark4.trust       # writes records/lab/forward_p_Mark_4j_0.npy and prints the comparison with Mark 4i
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from mark4 import forward as F

BLOCKS = (1, 7, 13)
SEASONS = (2022, 2023, 2024, 2025, 2026)
HALF_LIFE = 1.0
C_RIDGE = 0.3
MIN_TRAIN = 3000


def logit(q):
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def design(d, A, X3, S, C, E, p3, p4):
    lf = logit(A.fair_o.astype(float))
    d3, d4 = logit(p3) - lf, logit(p4) - lf
    mk = d["market"].to_numpy()
    markets = sorted(set(mk))
    cols, names = [lf], ["book"]
    # groups the trust can depend on
    sit = {n: np.nan_to_num(S[:, i]) for i, n in enumerate(F.SIT_COLS)}
    hist = d["history"].to_numpy().astype(float)
    ev_early = ((E > 0) & (E <= 3)).any(1).astype(float)
    ev_late = (E >= 4).any(1).astype(float)
    agree = np.nan_to_num(C[:, 10])          # comps vs the line: agreement of the chains (rline)
    nch = np.nan_to_num(C[:, 11])
    groups = {"all": np.ones(len(mk)), "role change, games 1-3": ev_early, "role change, game 4+": ev_late,
              "thin history (<8 games)": (hist < 8).astype(float), "comps agree": (agree > 0.5).astype(float) * (nch > 0),
              "price moved toward over": (sit["move_over"] > 0).astype(float), "price moved toward under": (sit["move_over"] < 0).astype(float),
              "big disagreement": (np.abs(0.5 * (d3 + d4)) > 0.4).astype(float)}
    for m in markets:
        groups[f"market {m}"] = (mk == m).astype(float)
    for g, ind in groups.items():
        cols += [d3 * ind, d4 * ind]
        names += [f"Mark 3 x {g}", f"Mark 4h x {g}"]
    for m in markets:
        cols.append((mk == m).astype(float))
        names.append(f"book bias {m}")
    X = np.column_stack(cols)
    return np.nan_to_num(X), names


def fit_forward(d, A, X, verbose=False):
    y = A.over_won.astype(float)
    season, week = d["season"].to_numpy(), d["week"].to_numpy()
    ok = ~np.isnan(y) & ~np.isnan(A.fair_o.astype(float))
    p = np.full(len(y), np.nan)
    coefs = {}
    for Sx in SEASONS:
        for i, b0 in enumerate(BLOCKS):
            b1 = BLOCKS[i + 1] if i + 1 < len(BLOCKS) else 99
            blk = (season == Sx) & (week >= b0) & (week < b1)
            if not blk.any():
                continue
            tr = ok & ((season < Sx) | ((season == Sx) & (week < b0)))
            if tr.sum() < MIN_TRAIN:
                continue
            w = 0.5 ** ((Sx - season[tr]) / HALF_LIFE)
            m = LogisticRegression(C=C_RIDGE, max_iter=2000)
            m.fit(X[tr], y[tr], sample_weight=w / w.mean())
            p[blk] = m.predict_proba(X[blk])[:, 1]
            coefs[(Sx, b0)] = m.coef_[0]
    return p, coefs


def main():
    d, A, X3, S, C, E, P = F.frame()
    p3, p4 = np.load(F.OUT / "forward_p_Mark_3_det_0.npy"), np.load(F.OUT / "forward_p_Mark_4h_clean_0.npy")
    X, names = design(d, A, X3, S, C, E, p3, p4)
    p, coefs = fit_forward(d, A, X)
    p4i = np.load(F.OUT / "forward_p_Mark_4i_0.npy")
    # lines a block could not fit (2022 weeks 1-6) fall back to Mark 4i so arrays align
    p = np.where(np.isnan(p), p4i, p)
    np.save(F.OUT / "forward_p_Mark_4j_0.npy", p)
    y, fair, season = A.over_won.astype(float), A.fair_o.astype(float), d["season"].to_numpy()
    ok = ~np.isnan(y) & ~np.isnan(fair)
    ll = lambda q: -(y * np.log(np.clip(q, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(q, 1e-4, 1 - 1e-4)))
    for name, q in (("Mark 3", p3), ("Mark 4h", p4), ("Mark 4i", p4i), ("Mark 4j", p)):
        g = {int(s): round(1000 * float((ll(fair) - ll(q))[ok & (season == s)].mean()), 2) for s in SEASONS}
        print(f"{name:8s} log-loss gain vs the book per 1,000 lines, by season: {g}")
    last = coefs[max(coefs)]
    print("\nlatest block's trust weights (positive = move toward that model in that group):")
    for n, c in sorted(zip(names, last), key=lambda x: -abs(x[1]))[:25]:
        print(f"   {c:+.3f}  {n}")
    for v, q in (("Mark 4i", p4i), ("Mark 4j", p)):
        eo, eu = q - A.be_o, (1 - q) - A.be_u
        ed = np.maximum(eo, eu)
        legs = pl.DataFrame(dict(season=season, week=d["week"].to_numpy(), pos=(ed >= 0) & ok)).group_by("season", "week").agg(pl.col("pos").sum())
        print(v, "lines with a positive edge per week, by season:", legs.group_by("season").agg(pl.col("pos").mean().round(1)).sort("season").rows())
    return p


if __name__ == "__main__":
    main()
