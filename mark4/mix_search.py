"""Portfolio mix search (2026-10-20): which parlay lengths, how many, at what stake, to win the most for a losing run the project manager can live with.

The 4 boosted slots a week are split among 2-, 3-, 4- and 5-leg parlays (every split), built in either order (short parlays get the best legs first,
or long ones do); optionally plain (unboosted) parlays on top from the leftover legs. Each group gets its own stake (0.25 to 1 unit). Legs: Mark 4i,
ranked by expected value, either side, all 7 markets, never two legs of one game, each leg used once a week.

Rated on:
  units won 2022 to 2026 week 6, and separately 2022-24 and 2025-26 (a mix that only works in one era is luck)
  the worst losing run counted from 2025 week 1 on only (the PM's rule: the game changes, older droughts don't count)
  parlays hit per group (a group carried by two hits is not proven)

  python -m mark4.mix_search      # writes records/lab/mix_search.csv (every mix) and prints the frontier
"""
from __future__ import annotations

import itertools

import numpy as np
import polars as pl

from mark4 import forward as F
from mark4 import portfolio as PF
from mark4 import tournament as T

LEGS = (2, 3, 4, 5)
STAKES = (0.25, 0.5, 0.75, 1.0)
PLAIN = [None, (2, 2), (2, 4), (3, 2), (4, 2)]       # extra unboosted parlays: (legs, count)
RUN_FROM = 2025


def splits():
    for c in itertools.product(range(5), repeat=len(LEGS)):
        if sum(c) == 4:
            yield {k: n for k, n in zip(LEGS, c) if n}


def group_weeks(W, keys, plan):
    """Per group of the plan: weekly (units at 1u stake, bets, hits)."""
    out = np.zeros((len(plan), len(keys), 3))
    for wi, key in enumerate(keys):
        legs = T.pool(W[key], 40)
        used = set()
        for gi, (k, c, boosted) in enumerate(plan):
            free = [r for i, r in enumerate(legs) if i not in used]
            for par in T.build(free, k, c, "rank", "any"):
                for r in par:
                    used.add(next(i for i, x in enumerate(legs) if x is r))
                win = all(r[3] == 1 for r in par)
                gain = float(np.prod([r[2] for r in par])) - 1
                out[gi, wi, 0] += gain * (1 + T.BOOST if boosted else 1.0) if win else -1.0
                out[gi, wi, 1] += 1
                out[gi, wi, 2] += win
    return out


def worst_run(u):
    run = np.concatenate([[0.0], np.cumsum(u)])
    return float((np.maximum.accumulate(run) - run).max())


def main():
    d, A, *_ = F.frame()
    W = T.week_pools(d, A, T.load_p("Mark 4i"), "both", "all", 0.0, "ev")
    keys = sorted(W)
    season = np.array([s for s, _ in keys])
    recent, old = season >= RUN_FROM, season < RUN_FROM
    rows, weekly, wstaked = [], [], []
    for split in splits():
        for order in ("short first", "long first"):
            ks = sorted(split, reverse=(order == "long first"))
            for extra in PLAIN:
                plan = [(k, split[k], True) for k in ks] + ([(extra[0], extra[1], False)] if extra else [])
                G = group_weeks(W, keys, plan)
                for stakes in itertools.product(STAKES, repeat=len(plan)):
                    if max(stakes) != 1.0:          # a mix and the same mix scaled down are the same strategy; fix the biggest stake at 1u
                        continue
                    sv = np.array(stakes)[:, None]
                    u = (G[:, :, 0] * sv).sum(0)
                    st = (G[:, :, 1] * sv).sum(0)
                    weekly.append(u); wstaked.append(st)
                    name = " + ".join(f"{int(c)} x {k}-leg{'' if b else ' plain'} @ {s:g}u" for (k, c, b), s in zip(plan, stakes))
                    rows.append(dict(mix=name, order=order, units=u.sum(), staked=st.sum(), ret=u.sum() / st.sum() if st.sum() else None,
                                     units_2022_24=u[old].sum(), units_2025_26=u[recent].sum(), per_week=st.sum() / len(keys),
                                     worst_run_2025=worst_run(u[recent]), worst_run_all=worst_run(u),
                                     winning_weeks_2025=int((u[recent] > 0).sum()), weeks_2025=int(recent.sum()),
                                     hits=" / ".join(f"{k}-leg {int(G[i, :, 2].sum())} of {int(G[i, :, 1].sum())}" for i, (k, c, b) in enumerate(plan))))
    t = pl.DataFrame(rows).with_row_index("i").unique("mix", keep="first").sort("i")
    np.savez_compressed(F.OUT / "mix_search_weeks.npz", units=np.array(weekly)[t["i"].to_numpy()], staked=np.array(wstaked)[t["i"].to_numpy()],
                        season=season, week=np.array([w for _, w in keys]))
    t = t.drop("i")
    t.write_csv(F.OUT / "mix_search.csv")
    nested(t)
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(90)
    show = ["mix", "units", "ret", "units_2022_24", "units_2025_26", "worst_run_2025", "worst_run_all", "winning_weeks_2025", "hits"]
    print(len(t), "mixes")
    for cap in (15, 20, 25, 30, 40, 1e9):
        x = t.filter((pl.col("worst_run_2025") <= cap) & (pl.col("units_2022_24") > 0) & (pl.col("units_2025_26") > 0)).sort("units", descending=True)
        print(f"\n== worst run since 2025 <= {cap}u, up in both eras: {x.height} mixes; best by units")
        print(x.head(6).select(show).with_columns(pl.col(pl.Float64).round(2)))
    return t


def nested(t: pl.DataFrame, caps=(15.0, 25.0, 1e9), seasons=(2023, 2024, 2025, 2026)):
    """The honest test of the search: before each season, pick the mix with the most recency-weighted units over EARLIER seasons whose worst losing
    run in the latest earlier season stays within the cap; bet it that season. Compared with fixed simple mixes chosen in advance."""
    z = np.load(F.OUT / "mix_search_weeks.npz")
    U, S, season = z["units"], z["staked"], z["season"]
    names = t["mix"].to_list()
    fixed = ["2 x 2-leg @ 1u + 1 x 3-leg @ 0.5u + 1 x 4-leg @ 0.25u", "4 x 2-leg @ 1u", "4 x 4-leg @ 1u", "2 x 2-leg @ 1u + 2 x 4-leg @ 0.25u"]
    out = []
    for cap in caps:
        tot_u = tot_s = 0.0
        picks = []
        for Sx in seasons:
            past = season < Sx
            w = 0.5 ** (Sx - season[past])
            score = (U[:, past] * w).sum(1) / np.maximum((S[:, past] * w).sum(1), 1e-9) * np.minimum((S[:, past] * w).sum(1), 1e9)
            last = season == Sx - 1
            ok = np.array([worst_run(U[i, last]) <= cap for i in range(len(U))])
            i = int(np.argmax(np.where(ok, score, -1e18)))
            cur = season == Sx
            tot_u += U[i, cur].sum(); tot_s += S[i, cur].sum()
            picks.append(f"{Sx}: {names[i]} -> {U[i, cur].sum():+.1f}u on {S[i, cur].sum():.1f}")
        out.append(dict(chooser=f"search, cap {cap:g}u", units=tot_u, staked=tot_s, picks=" | ".join(picks)))
    for f in fixed:
        if f in names:
            i = names.index(f); cur = season >= seasons[0]
            out.append(dict(chooser=f"fixed: {f}", units=float(U[i, cur].sum()), staked=float(S[i, cur].sum()), picks=""))
    r = pl.DataFrame(out).with_columns(ret=pl.col("units") / pl.col("staked"))
    r.write_csv(F.OUT / "mix_search_nested.csv")
    for x in r.iter_rows(named=True):
        print(f"{x['chooser']:70s} 2023-26: {x['units']:+7.1f}u on {x['staked']:6.1f} ({x['ret']:+.0%})")
        if x["picks"]:
            print("     ", x["picks"].replace(" | ", "\n      "))
    return r


if __name__ == "__main__":
    main()
