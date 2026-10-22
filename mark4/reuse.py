"""Reusing legs (2026-10-22, the PM's question): put the best legs in more than one bet (a short parlay AND the long one), or also bet them straight,
instead of using each leg once. Same Mark 4l legs, same 4 boosts a week; every variant played week by week 2022 - 2026 week 6.

Legs: each week's legs with an edge, ranked by expected value, keeping one leg per game (G[0] best, G[1] next best from another game, ...), so any
set of them is a valid FanDuel parlay. A variant is a list of bets: (leg ranks, stake in units, boosted or not); straights are 1-leg bets.

  python -m mark4.reuse       # writes records/lab/reuse.csv and prints the comparison
"""
from __future__ import annotations

import numpy as np
import polars as pl

from mark4 import compare as CP
from mark4 import forward as F
from mark4 import portfolio as PF
from mark4 import tournament as T

B = True
VARIANTS = {
    "now: 5 / 3 / 2 / 2, each leg once": [((0, 1, 2, 3, 4), .25, B), ((5, 6, 7), .5, B), ((8, 9), 1, B), ((10, 11), 1, B)],
    "top 2 also as a 2-leg (5 / 2 reuse / 3 / 2)": [((0, 1, 2, 3, 4), .25, B), ((0, 1), 1, B), ((5, 6, 7), .5, B), ((8, 9), 1, B)],
    "nested: 5-leg, its top 3, its top 2, legs 3+4": [((0, 1, 2, 3, 4), .25, B), ((0, 1, 2), .5, B), ((0, 1), 1, B), ((2, 3), 1, B)],
    "5-leg + its legs as two 2-legs + next 3-leg": [((0, 1, 2, 3, 4), .25, B), ((0, 1), 1, B), ((2, 3), 1, B), ((5, 6, 7), .5, B)],
    "four 2-legs from the top 4 (each leg twice)": [((0, 1), 1, B), ((2, 3), 1, B), ((0, 2), 1, B), ((1, 3), 1, B)],
    "four 2-legs, each leg once (top 8)": [((0, 1), 1, B), ((2, 3), 1, B), ((4, 5), 1, B), ((6, 7), 1, B)],
    "now + the 5-leg's legs straight @ 1u": [((0, 1, 2, 3, 4), .25, B), ((5, 6, 7), .5, B), ((8, 9), 1, B), ((10, 11), 1, B)] + [((i,), 1, False) for i in range(5)],
    "now + every card leg straight @ 0.5u": [((0, 1, 2, 3, 4), .25, B), ((5, 6, 7), .5, B), ((8, 9), 1, B), ((10, 11), 1, B)] + [((i,), .5, False) for i in range(12)],
    "top 5 straight only @ 1u": [((i,), 1, False) for i in range(5)],
    "5 / 3 / 2 / 2 nested, bigger 5-leg @ .5": [((0, 1, 2, 3, 4), .5, B), ((0, 1, 2), .5, B), ((0, 1), 1, B), ((2, 3), 1, B)],
}


def unique_games(rows):
    out, games, players = [], set(), set()
    for r in rows:
        if r[4] in games or r[5] in players:
            continue
        out.append(r)
        games.add(r[4]); players.add(r[5])
    return out


def play(G, bets):
    u = st = 0.0
    for ranks, stake, boosted in bets:
        if max(ranks) >= len(G):
            continue
        legs = [G[i] for i in ranks]
        win = all(r[3] == 1 for r in legs)
        dec = float(np.prod([r[2] for r in legs]))
        u += stake * (((dec - 1) * (1 + T.BOOST if boosted else 1.0)) if win else -1.0)
        st += stake
    return u, st


def live_card(rows):
    """The live card's own construction (mark4/live.py): 5-leg, 3-leg, 2-leg, 2-leg built long-first, each leg once."""
    legs = T.pool(rows, 40)
    used, pars = set(), []
    for k, c, stake in ((5, 1, .25), (3, 1, .5), (2, 2, 1.0)):
        free = [r for j, r in enumerate(legs) if j not in used]
        for par in T.build(free, k, c, "rank", "any"):
            for r in par:
                used.add(next(j for j, y in enumerate(legs) if y is r))
            pars.append((par, stake))
    return pars


def derived(pars, mode):
    """Bets [(legs, stake, boosted)] from the live card's parlays for each reuse idea."""
    P = {len(p): (p, s) for p, s in pars[:2]}
    p5 = P.get(5, (None, 0))[0]
    base = [(p, s, True) for p, s in pars]
    if mode == "now":
        return base
    if mode == "now + 5-leg's legs straight 1u":
        return base + ([([r], 1.0, False) for r in p5] if p5 else [])
    if mode == "now + every card leg straight 0.5u":
        return base + [([r], .5, False) for p, s in pars for r in p]
    if mode == "now + every card leg straight 1u":
        return base + [([r], 1.0, False) for p, s in pars for r in p]
    if p5 is None:
        return base
    others = [(p, s) for p, s in pars if len(p) != 5]
    if mode == "5-leg's top 2 as the 2nd 2-leg":
        keep = [(p, s, True) for p, s in others if len(p) == 3] + [(p, s, True) for p, s in others if len(p) == 2][:1]
        return [(p5, .25, True)] + keep + [(p5[:2], 1.0, True)]
    if mode == "nested: 5-leg, its top 3, top 2, legs 3+4":
        return [(p5, .25, True), (p5[:3], .5, True), (p5[:2], 1.0, True), (p5[2:4], 1.0, True)]
    if mode == "5-leg + its legs as two 2-legs + the 3-leg":
        keep = [(p, s, True) for p, s in others if len(p) == 3]
        return [(p5, .25, True), (p5[:2], 1.0, True), (p5[2:4], 1.0, True)] + keep
    if mode == "5-leg + its legs straight, no other parlays":
        return [(p5, .25, True)] + [([r], 1.0, False) for r in p5]
    raise ValueError(mode)


MODES = ["now", "5-leg's top 2 as the 2nd 2-leg", "nested: 5-leg, its top 3, top 2, legs 3+4", "5-leg + its legs as two 2-legs + the 3-leg",
         "now + 5-leg's legs straight 1u", "now + every card leg straight 0.5u", "now + every card leg straight 1u", "5-leg + its legs straight, no other parlays"]


def play_bets(bets):
    u = st = 0.0
    for legs, stake, boosted in bets:
        win = all(r[3] == 1 for r in legs)
        dec = float(np.prod([r[2] for r in legs]))
        u += stake * (((dec - 1) * (1 + T.BOOST if boosted else 1.0)) if win else -1.0)
        st += stake
    return u, st


def main_live():
    d, A, *_ = F.frame()
    p = np.load(F.OUT / "forward_p_Mark_4l_0.npy").copy()
    season = d["season"].to_numpy()
    p[season < 2022] = np.nan
    W = T.week_pools(d, A, p, "both", "all", 0.0, "ev")
    keys = sorted(W)
    cards = {k: live_card(W[k]) for k in keys}
    res, weeks = [], {}
    for mode in MODES:
        r = {k: play_bets(derived(cards[k], mode)) for k in keys}
        weeks[mode] = r
        u = np.array([r[k][0] for k in keys]); st = np.array([r[k][1] for k in keys]); ss = np.array([k[0] for k in keys])
        dd, _ = PF.worst_run(u); d25, _ = PF.worst_run(u[ss >= 2025])
        res.append(dict(variant=mode, units=u.sum(), staked=st.sum(), ret=u.sum() / st.sum(), **{str(y): float(u[ss == y].sum()) for y in range(2022, 2027)},
                        worst_run=dd, worst_run_2025=d25, winning_weeks=int((u > 0).sum()), weeks=len(keys)))
    for r in res:
        r["chance_better_than_now"] = CP.boot(weeks["now"], weeks[r["variant"]])["p_b_better"] if r["variant"] != "now" else None
    t = pl.DataFrame(res)
    t.write_csv(F.OUT / "reuse_live.csv")
    for r in t.iter_rows(named=True):
        print(f"{r['variant']:44s} won {r['units']:+7.1f} on {r['staked']:6.1f} ({r['ret']:+5.0%}) | seasons {[round(r[str(y)], 1) for y in range(2022, 2027)]} | worst run {r['worst_run']:5.1f}u"
              f" (since 2025 {r['worst_run_2025']:4.1f}) | winning weeks {r['winning_weeks']}/{r['weeks']}" + (f" | chance better than now {r['chance_better_than_now']:.0%}" if r["chance_better_than_now"] is not None else ""))
    return t


def main():
    d, A, *_ = F.frame()
    p = np.load(F.OUT / "forward_p_Mark_4l_0.npy").copy()
    season = d["season"].to_numpy()
    p[season < 2022] = np.nan
    W = T.week_pools(d, A, p, "both", "all", 0.0, "ev")
    keys = sorted(W)
    G = {k: unique_games(W[k]) for k in keys}
    res, weeks = [], {}
    for name, bets in VARIANTS.items():
        r = {k: play(G[k], bets) for k in keys}
        weeks[name] = r
        u = np.array([r[k][0] for k in keys]); st = np.array([r[k][1] for k in keys]); ss = np.array([k[0] for k in keys])
        dd, _ = PF.worst_run(u)
        d25, _ = PF.worst_run(u[ss >= 2025])
        res.append(dict(variant=name, units=u.sum(), staked=st.sum(), ret=u.sum() / st.sum(), **{str(y): float(u[ss == y].sum()) for y in range(2022, 2027)},
                        worst_run=dd, worst_run_2025=d25, winning_weeks=int((u > 0).sum()), weeks=len(keys)))
    base = weeks["now: 5 / 3 / 2 / 2, each leg once"]
    for r in res:
        b = CP.boot(base, weeks[r["variant"]])
        r["chance_better_than_now"] = b["p_b_better"] if r["variant"] != "now: 5 / 3 / 2 / 2, each leg once" else None
    t = pl.DataFrame(res)
    t.write_csv(F.OUT / "reuse.csv")
    for r in t.iter_rows(named=True):
        print(f"{r['variant']:46s} won {r['units']:+7.1f} on {r['staked']:6.1f} ({r['ret']:+5.0%}) | seasons {[round(r[str(y)], 1) for y in range(2022, 2027)]} | worst run {r['worst_run']:5.1f}u"
              f" (since 2025 {r['worst_run_2025']:4.1f}) | winning weeks {r['winning_weeks']}/{r['weeks']}" + (f" | chance better than now {r['chance_better_than_now']:.0%}" if r["chance_better_than_now"] is not None else ""))
    return t


if __name__ == "__main__":
    import sys
    main_live() if "--grid" not in sys.argv else main()
