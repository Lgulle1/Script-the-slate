"""One fixed betting setup tuned on all graded lines at once (yours, 2026-10-15: "give them all one set of data and tweak the numbers until it works"), then the same tuning
done on one season and checked on the other, so the in-sample number can be compared with what it does on data it was not tuned on.

Knobs: w = weight on the model in the blended chance (logit scale; the rest on the book's vig-free chance), b = shift added to the over's chance (a formula change),
K / c = neighbour pool and width of the model's distribution, theta = minimum edge, side, and the market list (a market is kept when it is profitable under the other knobs).
Descriptive; sets nothing in the model.

  python scripts/tune_setup.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

SEL = config.PROCESSED_DIR / "line_study_sel.parquet"
MARKETS = ["pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds", "rec", "rec_yds"]
WS = np.round(np.arange(0, 1.01, 0.1), 2)
BS = (-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04)
KC = [(k, c) for k in (300, 600, 1200) for c in (85, 100, 114)]
THETAS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
SIDES = ("both", "over", "under")
MIN_BETS = 300


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def lines() -> pl.DataFrame:
    s = pl.read_parquet(SEL)
    k = ["season", "week", "game_id", "player_id", "market", "line"]
    o = s.filter(pl.col("side") == "over").select(*k, "tier", "fair", "be", "dec", "won", *[c for c in s.columns if c.startswith("p_over_k")])
    u = s.filter(pl.col("side") == "under").select(*k, be_u="be", dec_u="dec")
    return o.join(u, on=k).rename({"fair": "fair_o", "be": "be_o", "dec": "dec_o", "won": "over_won"})


def settle(L: dict, w, b, kc):
    """For one (w, b, K/c): chosen side, its edge, profit, won, price per line."""
    pm = L[f"p_over_k{kc[0]}_c{kc[1]}"]
    p = 1 / (1 + np.exp(-(w * logit(pm) + (1 - w) * logit(L["fair_o"])))) + b
    eo, eu = p - L["be_o"], (1 - p) - L["be_u"]
    over = eo >= eu
    won = np.where(over, L["over_won"], 1 - L["over_won"])
    dec = np.where(over, L["dec_o"], L["dec_u"])
    return over, np.where(over, eo, eu), np.where(won == 1, dec - 1, -1.0), won, dec


def tune(L: dict, mk_idx: np.ndarray):
    """Best setup by total return: every knob combination, markets kept when profitable under it, at least MIN_BETS bets."""
    best = None
    nm = len(MARKETS)
    for kc in KC:
        for w in WS:
            for b in BS:
                over, edge, prof, won, dec = settle(L, w, b, kc)
                for th in THETAS:
                    base = edge >= th
                    for sd in SIDES:
                        m = base & (over if sd == "over" else (~over if sd == "under" else True))
                        n = np.bincount(mk_idx[m], minlength=nm)
                        s = np.bincount(mk_idx[m], weights=prof[m], minlength=nm)
                        keep = s > 0
                        if n[keep].sum() < MIN_BETS:
                            continue
                        roi = s[keep].sum() / n[keep].sum()
                        if best is None or roi > best["roi"]:
                            best = dict(roi=float(roi), n=int(n[keep].sum()), kc=kc, w=float(w), b=b, theta=th, side=sd, markets=[MARKETS[i] for i in np.flatnonzero(keep)])
    return best


def apply(L: dict, mk_idx: np.ndarray, st: dict):
    over, edge, prof, won, dec = settle(L, st["w"], st["b"], st["kc"])
    m = (edge >= st["theta"]) & np.isin(mk_idx, [MARKETS.index(x) for x in st["markets"]])
    if st["side"] != "both":
        m &= over if st["side"] == "over" else ~over
    return m, prof, won, dec


def as_dict(d: pl.DataFrame) -> dict:
    return {c: d[c].to_numpy() for c in d.columns if d[c].dtype != pl.String} | {"game_id": d["game_id"].to_numpy()}


def describe(st):
    return (f"model weight {st['w']:.0%} (book {1 - st['w']:.0%}), over-chance shift {st['b']:+.0%}, pool {st['kc'][0]}, width x{st['kc'][1] / 100:.2f}, "
            f"edge >= {st['theta']:.0%}, side {st['side']}, markets {', '.join(st['markets'])}")


def summary(name, L, mk, st):
    m, prof, won, dec = apply(L, mk, st)
    if m.sum() == 0:
        return dict(test=name, bets=0)
    be = (1 / dec[m]).mean()
    return dict(test=name, bets=int(m.sum()), hit=float(won[m].mean()), break_even=float(be), roi=float(prof[m].mean()), units=float(prof[m].sum()))


def main():
    d = lines().filter(pl.col("tier") != "late")                  # clean rows; the late rows changed nothing in the line study
    parts = {"all": d, "2024": d.filter(pl.col("season") == 2024), "2025-26": d.filter(pl.col("season") >= 2025)}
    Ls = {k: as_dict(v) for k, v in parts.items()}
    mks = {k: np.array([MARKETS.index(x) for x in v["market"].to_list()]) for k, v in parts.items()}
    rows = []
    for tuned_on, tested_on in (("all", "all"), ("2024", "2024"), ("2024", "2025-26"), ("2025-26", "2025-26"), ("2025-26", "2024")):
        st = tune(Ls[tuned_on], mks[tuned_on])
        r = summary(f"tuned on {tuned_on}, scored on {tested_on}", Ls[tested_on], mks[tested_on], st)
        rows.append(r | dict(setup=describe(st)))
        print(r, "\n   ", describe(st))
    pl.DataFrame(rows).write_parquet(config.PROCESSED_DIR / "tune_setup.parquet")


if __name__ == "__main__":
    main()
