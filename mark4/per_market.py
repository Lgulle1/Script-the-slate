"""Per-market settings (2026-10-23): Mark 4i used one setting for every market. Each variant here is chosen forward (each season from earlier seasons only,
best recency-weighted log loss, the same rule Mark 4i uses) and played on the live card and leg by leg.

  devig   how FanDuel's vig is removed for the book's fair chance: "mult" (proportional, what Mark 4i uses), "power", "add" (equal split);
          chosen forward on the book's own log loss
  scope   "global" (Mark 4i: one model weight for every market) or "market" (its own weights per market; a market with too few past lines keeps the global)
  shift   with "market": also a logit bias correction per market, chosen with the weights

Results: money on the real 5/3/2/2 card (mark4/live.py PLAN), 2022 - 2026 week 6, and leg by leg (every leg in the week's top-12 pool, its realized
return at its price) against Mark 4i, weeks resampled.

  python -m mark4.per_market      # writes records/lab/per_market.csv
"""
from __future__ import annotations

import itertools

import numpy as np
import polars as pl

from mark4 import cards as C
from mark4 import compare as CP
from mark4 import forward as F
from mark4 import live as LV
from mark4 import tournament as T

SHIFTS = (-0.15, -0.075, 0.0, 0.075, 0.15)
MIN_PAST = 1500                  # past lines a market needs to get its own weights
PLAN = [(k, c, True, s) for k, c, s in LV.PLAN]
cl = lambda q: np.clip(q, 1e-4, 1 - 1e-4)
logit = lambda q: np.log(cl(q) / (1 - cl(q)))
sig = lambda z: 1 / (1 + np.exp(-z))


def devig(be_o, be_u, how):
    if how == "mult":
        return be_o / (be_o + be_u)
    if how == "add":
        return be_o - (be_o + be_u - 1) / 2
    lo, hi = np.ones_like(be_o), np.full_like(be_o, 3.0)              # power: k with be_o^k + be_u^k = 1
    for _ in range(60):
        k = (lo + hi) / 2
        big = be_o ** k + be_u ** k > 1
        lo, hi = np.where(big, k, lo), np.where(big, hi, k)
    return be_o ** ((lo + hi) / 2)


def build(d, A, p3, p4, fair, scope, shift, verbose=False):
    y, season, market = A.over_won.astype(float), d["season"].to_numpy(), d["market"].to_numpy()
    ok = ~np.isnan(p3) & ~np.isnan(p4) & ~np.isnan(fair) & ~np.isnan(y)
    ll = lambda q, m: -(y[m] * np.log(cl(q)) + (1 - y[m]) * np.log(1 - cl(q)))
    out = (p3 + p4) / 2
    grid = list(itertools.product(T.MIX_WM, T.MIX_B, SHIFTS if shift else (0.0,)))
    picks = {}

    def best(m, w):
        def score(c):
            q = sig(logit(c[1] * (c[0] * p3[m] + (1 - c[0]) * p4[m]) + (1 - c[1]) * fair[m]) + c[2])
            return (w * (ll(fair[m], m) - ll(q, m))).sum()
        return max(grid, key=score)

    for S in F.SEASONS[1:]:
        past = ok & (season < S)
        g = best(past, 0.5 ** ((S - season[past]) / F.HALF_LIFE))
        groups = {"all": (g, season == S)}
        if scope == "market":
            groups = {}
            for mk in np.unique(market):
                pm = past & (market == mk)
                c = best(pm, 0.5 ** ((S - season[pm]) / F.HALF_LIFE)) if pm.sum() >= MIN_PAST else g
                groups[mk] = (c, (season == S) & (market == mk))
        for mk, (c, cur) in groups.items():
            q = sig(logit(c[1] * (c[0] * p3 + (1 - c[0]) * p4) + (1 - c[1]) * fair) + c[2])
            out[cur] = np.where(np.isnan(fair[cur]), (c[0] * p3 + (1 - c[0]) * p4)[cur], q[cur])
            picks[(S, mk)] = c
    if verbose:
        for (S, mk), c in sorted(picks.items()):
            if S == 2026:
                print(f"     2026 {mk:12s} Mark 3 weight {c[0]:.2f}  model weight {c[1]:.2f}  shift {c[2]:+.3f}")
    return out


def legs(W):
    """{week: (summed realized return of the week's top-12 legs at their price, legs)}"""
    out = {}
    for k, rows in W.items():
        top = T.pool(rows, 12)
        out[k] = (float(sum((r[2] - 1) if r[3] == 1 else -1.0 for r in top)), float(len(top)))
    return out


def main():
    d, A, *_ = F.frame()
    season = d["season"].to_numpy()
    y = A.over_won.astype(float)
    p3, p4 = np.load(F.OUT / "forward_p_Mark_3_det_0.npy"), np.load(F.OUT / "forward_p_Mark_4h_clean_0.npy")
    be_o, be_u = A.be_o.astype(float), A.be_u.astype(float)
    # de-vig method, chosen forward on the book's own log loss
    FAIR = {h: devig(be_o, be_u, h) for h in ("mult", "power", "add")}
    ok = ~np.isnan(y) & ~np.isnan(be_o) & ~np.isnan(be_u)
    llb = {h: -(y * np.log(cl(f)) + (1 - y) * np.log(1 - cl(f))) for h, f in FAIR.items()}
    print("book log loss per 1,000 lines by de-vig method (lower is better):")
    for S in F.SEASONS:
        m = ok & (season == S)
        print(f"   {S}: " + "  ".join(f"{h} {1000 * llb[h][m].mean():.2f}" for h in FAIR))
    fwd = {}
    for S in F.SEASONS[1:]:
        m = ok & (season < S); w = 0.5 ** ((S - season[m]) / F.HALF_LIFE)
        fwd[S] = min(FAIR, key=lambda h: (w * llb[h][m]).sum())
    print("de-vig chosen forward:", fwd)
    fair_fwd = FAIR["mult"].copy()
    for S, h in fwd.items():
        fair_fwd[season == S] = FAIR[h][season == S]

    V = {"Mark 4i (now)": ("mult", "global", False), "per-market weights": ("mult", "market", False), "per-market weights + bias": ("mult", "market", True),
         "de-vig chosen forward": ("fwd", "global", False), "de-vig + per-market weights + bias": ("fwd", "market", True)}
    R, LG, LL = {}, {}, {}
    for name, (dv, scope, sh) in V.items():
        print(f"\n{name}")
        fair = fair_fwd if dv == "fwd" else FAIR[dv]
        p = build(d, A, p3, p4, fair, scope, sh, verbose=True)
        p = p.copy(); p[season < 2022] = np.nan
        m = ok & ~np.isnan(p)
        LL[name] = {S: 1000 * float((-(y * np.log(cl(fair)) + (1 - y) * np.log(1 - cl(fair))) - (-(y * np.log(cl(p)) + (1 - y) * np.log(1 - cl(p)))))[m & (season == S)].mean())
                    for S in F.SEASONS}
        W = T.week_pools(d, A, p, "both", "all", 0.0, "ev")
        R[name] = {k: C.play_week(W[k], PLAN, 0) for k in sorted(W)}
        LG[name] = legs(W)
    base = "Mark 4i (now)"
    rows = []
    for name in V:
        r = R[name]; u = sum(v[0] for v in r.values()); b = sum(v[1] for v in r.values())
        u25 = sum(v[0] for k, v in r.items() if k[0] >= 2025); b25 = sum(v[1] for k, v in r.items() if k[0] >= 2025)
        lu = sum(v[0] for v in LG[name].values()); ln = sum(v[1] for v in LG[name].values())
        rows.append(dict(variant=name, card_units=u, card_staked=b, card_ret=u / b, card_2025on=u25, staked_2025on=b25,
                         **{f"u{s}": sum(v[0] for k, v in r.items() if k[0] == s) for s in range(2022, 2027)},
                         legs=ln, leg_ret=lu / ln, **{f"beats_book_{s}": LL[name][s] for s in range(2022, 2027)},
                         card_chance_better=None if name == base else CP.boot(R[base], r)["p_b_better"],
                         legs_chance_better=None if name == base else CP.boot(LG[base], LG[name])["p_b_better"]))
    t = pl.DataFrame(rows)
    t.write_csv(F.OUT / "per_market.csv")
    print()
    for r in t.iter_rows(named=True):
        print(f"{r['variant']:36s} card {r['card_units']:+7.1f} on {r['card_staked']:6.1f} ({r['card_ret']:+.0%}) | 2025 on {r['card_2025on']:+6.1f} on {r['staked_2025on']:5.1f} | "
              f"legs {r['leg_ret']:+.2%} on {r['legs']:.0f} | beats book {[round(r[f'beats_book_{s}'], 2) for s in range(2022, 2027)]}"
              + (f" | chance better than now: card {r['card_chance_better']:.0%}, legs {r['legs_chance_better']:.0%}" if r["card_chance_better"] is not None else ""))
    return t


if __name__ == "__main__":
    main()
