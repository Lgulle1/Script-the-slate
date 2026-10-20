"""Real weekly cards, played week by week (2026-10-20): what a card would actually have made, hits and misses, not the all-combo average.

A card: up to 4 boosted parlays (one per boost window; the 30% profit boost), optionally more plain parlays and straights, built from the week's legs
ranked by the version's rule, never two legs of one game, a player at most once a week. 1 unit per bet. Per season: units, units bet, return, worst
drawdown (peak-to-trough of the running total, in units), weeks in profit.

  python -m mark4.cards      # writes records/lab/cards.csv and cards_weeks.csv
"""
from __future__ import annotations

import numpy as np
import polars as pl

from mark4 import forward as F
from mark4 import tournament as T

CARDS = {
    # name: (version, pool spec, [(legs, count, boosted)], straights)
    "Mark 3 rule (frozen)": ("Mark 3 det", dict(rank="chance", sides="under", markets="four", edge=0.04), [(2, 4, True)], 0),
    "Mark 4i: 4 boosted 2-leg": ("Mark 4i", dict(rank="ev", sides="both", markets="all", edge=0.0), [(2, 4, True)], 0),
    "Mark 4i: 4 boosted 3-leg": ("Mark 4i", dict(rank="ev", sides="both", markets="all", edge=0.0), [(3, 4, True)], 0),
    "Mark 4i: 4 boosted 4-leg": ("Mark 4i", dict(rank="ev", sides="both", markets="all", edge=0.0), [(4, 4, True)], 0),
    "Mark 4i: 2 boosted 2-leg + 2 boosted 4-leg": ("Mark 4i", dict(rank="ev", sides="both", markets="all", edge=0.0), [(2, 2, True), (4, 2, True)], 0),
    "Mark 4i: 4 boosted 2-leg + top 10 straights": ("Mark 4i", dict(rank="ev", sides="both", markets="all", edge=0.0), [(2, 4, True)], 10),
    "Mark 4i: 4 boosted 2-leg + 4 plain 2-leg": ("Mark 4i", dict(rank="ev", sides="both", markets="all", edge=0.0), [(2, 4, True), (2, 4, False)], 0),
    "Mark 4i: top 10 straights only": ("Mark 4i", dict(rank="ev", sides="both", markets="all", edge=0.0), [], 10),
    "Blend: 4 boosted 2-leg": ("Blend", dict(rank="ev", sides="both", markets="all", edge=0.0), [(2, 4, True)], 0),
    "Blend: 4 boosted 3-leg": ("Blend", dict(rank="ev", sides="both", markets="all", edge=0.0), [(3, 4, True)], 0),
}


def play_week(rows, plan, straights):
    """(units, bets) for one week. Parlays are built best-first from the ranked legs; each leg is used once."""
    legs = T.pool(rows, 40)
    used = set()
    u = n = 0.0
    for k, c, boosted in plan:
        free = [r for i, r in enumerate(legs) if i not in used]
        pars = T.build(free, k, c, "rank", "any")
        for par in pars:
            for r in par:
                used.add(next(i for i, x in enumerate(legs) if x is r))
            win = all(r[3] == 1 for r in par)
            gain = float(np.prod([r[2] for r in par])) - 1
            u += gain * (1 + T.BOOST if boosted else 1.0) if win else -1.0
            n += 1
    if straights:
        for r in [r for i, r in enumerate(legs) if i not in used][:straights]:
            u += (r[2] - 1) if r[3] == 1 else -1.0
            n += 1
    return u, n


def main():
    d, A, *_ = F.frame()
    P = {}
    rows, weeks = [], []
    for name, (v, spec, plan, st) in CARDS.items():
        if v not in P:
            P[v] = T.load_p(v)
        W = T.week_pools(d, A, P[v], spec["sides"], spec["markets"], spec["edge"], spec["rank"])
        for (s, w) in sorted(W):
            u, n = play_week(W[(s, w)], plan, st)
            weeks.append(dict(card=name, season=s, week=w, units=u, bets=n))
    wk = pl.DataFrame(weeks).sort("card", "season", "week")
    wk.write_csv(F.OUT / "cards_weeks.csv")
    for (name,), x in wk.group_by("card", maintain_order=True):
        for (s,), y in x.group_by("season", maintain_order=True):
            run = np.cumsum(y["units"].to_numpy())
            dd = float((np.maximum.accumulate(np.concatenate([[0], run])) - np.concatenate([[0], run])).max())
            rows.append(dict(card=name, season=s, units=float(y["units"].sum()), bets=float(y["bets"].sum()), ret=float(y["units"].sum() / y["bets"].sum()) if y["bets"].sum() else None,
                             worst_drawdown=dd, weeks_up=int((y["units"] > 0).sum()), weeks=y.height))
    t = pl.DataFrame(rows)
    t.write_csv(F.OUT / "cards.csv")
    w = t.with_columns(w=0.5 ** (2026 - pl.col("season")))
    summ = w.group_by("card", maintain_order=True).agg(
        ret_recency=(pl.col("w") * pl.col("units")).sum() / (pl.col("w") * pl.col("bets")).sum(), ret_all=pl.col("units").sum() / pl.col("bets").sum(),
        units=pl.col("units").sum(), bets=pl.col("bets").sum(), seasons_up=(pl.col("units") > 0).sum(), worst_dd=pl.col("worst_drawdown").max(),
        u2026=pl.col("units").filter(pl.col("season") == 2026).sum())
    pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(60)
    print(summ.with_columns(pl.col(pl.Float64).round(3)))
    print(t.pivot(on="season", index="card", values="units").with_columns(pl.col(pl.Float64).round(1)))
    return t


if __name__ == "__main__":
    main()
