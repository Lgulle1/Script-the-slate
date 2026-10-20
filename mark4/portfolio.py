"""Balancing return against losing streaks (2026-10-20): the same Mark 4i legs, different stakes per parlay length.

A long parlay wins rarely and big; a short one often and small. Staking them the same is a choice, not a rule: a 4-leg parlay at a quarter unit keeps
its share of the upside while a 60-parlay drought costs 15 units, not 60. Each portfolio below fills the 4 weekly boosts (and optionally plain parlays)
with parlays of given lengths and stakes, built from Mark 4i's legs ranked by expected value, either side, all 7 markets, never two legs of one game.

Judged on:
  history   2022 to 2026 week 6, played week by week: units won, units bet, return per unit, the worst losing run (peak to trough of the running total)
  seasons   10,000 simulated 18-week seasons drawn from the real weeks (recent weeks drawn more often): median season, 1-in-20 bad season, chance the
            season loses, chance of a losing run of 20+ units inside the season
  growth    with a bankroll of BANKROLL units, the average weekly growth of the bankroll (log growth): the measure that rewards return and punishes
            ruinous swings at the same time, the standard way to compare bets of different risk

  python -m mark4.portfolio      # writes records/lab/portfolio.csv
"""
from __future__ import annotations

import numpy as np
import polars as pl

from mark4 import forward as F
from mark4 import tournament as T

BANKROLL = 100.0
SEASON_WEEKS, SIMS = 18, 10_000
SPEC = dict(rank="ev", sides="both", markets="all", edge=0.0)
# name: [(legs, count, boosted, stake in units)]
PORTFOLIOS = {
    "A. 4 x 2-leg @ 1u (steady)": [(2, 4, True, 1.0)],
    "B. 4 x 4-leg @ 1u (max return)": [(4, 4, True, 1.0)],
    "C. 4 x 4-leg @ 0.25u": [(4, 4, True, 0.25)],
    "D. 2 x 2-leg @ 1u + 2 x 4-leg @ 0.5u": [(2, 2, True, 1.0), (4, 2, True, 0.5)],
    "E. 2 x 2-leg @ 1u + 2 x 4-leg @ 0.25u": [(2, 2, True, 1.0), (4, 2, True, 0.25)],
    "F. 3 x 2-leg @ 1u + 1 x 4-leg @ 0.5u": [(2, 3, True, 1.0), (4, 1, True, 0.5)],
    "G. 2 x 2-leg @ 1u + 1 x 3-leg @ 0.5u + 1 x 4-leg @ 0.25u": [(2, 2, True, 1.0), (3, 1, True, 0.5), (4, 1, True, 0.25)],
    "H. 2 x 2-leg @ 1u + 2 x 3-leg @ 0.5u": [(2, 2, True, 1.0), (3, 2, True, 0.5)],
    "I. 4 x 3-leg @ 0.5u": [(3, 4, True, 0.5)],
    "J. 4 x 2-leg @ 1u + 2 x 4-leg plain @ 0.25u": [(2, 4, True, 1.0), (4, 2, False, 0.25)],
}


def play_week(rows, plan):
    """(units won, units bet, parlays, parlays won) for one week."""
    legs = T.pool(rows, 40)
    used = set()
    u = st = n = hits = 0.0
    for k, c, boosted, stake in plan:
        free = [r for i, r in enumerate(legs) if i not in used]
        for par in T.build(free, k, c, "rank", "any"):
            for r in par:
                used.add(next(i for i, x in enumerate(legs) if x is r))
            win = all(r[3] == 1 for r in par)
            gain = float(np.prod([r[2] for r in par])) - 1
            u += stake * (gain * (1 + T.BOOST if boosted else 1.0) if win else -1.0)
            st += stake; n += 1; hits += win
    return u, st, n, hits


def worst_run(u):
    run = np.concatenate([[0.0], np.cumsum(u)])
    peak = np.maximum.accumulate(run)
    j = int((peak - run).argmax())
    i = int(np.flatnonzero(run[:j + 1] == peak[j])[-1])
    return float(peak[j] - run[j]), j - i


def main():
    d, A, *_ = F.frame()
    W = T.week_pools(d, A, T.load_p("Mark 4i"), SPEC["sides"], SPEC["markets"], SPEC["edge"], SPEC["rank"])
    keys = sorted(W)
    season = np.array([s for s, _ in keys])
    pw = 0.5 ** (2026 - season); pw = pw / pw.sum()          # recent weeks drawn more often
    rng = np.random.default_rng(7)
    draws = rng.choice(len(keys), size=(SIMS, SEASON_WEEKS), p=pw)
    rows, weeks = [], []
    for name, plan in PORTFOLIOS.items():
        r = np.array([play_week(W[k], plan) for k in keys])
        u, st = r[:, 0], r[:, 1]
        weeks += [dict(portfolio=name, season=s, week=w, units=a, staked=b) for (s, w), a, b in zip(keys, u, st)]
        dd, dd_weeks = worst_run(u)
        sims = u[draws]
        tot = sims.sum(1)
        sim_dd = np.array([worst_run(x)[0] for x in sims[:2000]])
        growth = np.mean(np.log1p(np.clip(u / BANKROLL, -0.99, None)))
        rows.append(dict(portfolio=name, units=u.sum(), staked=st.sum(), ret=u.sum() / st.sum(), per_week_staked=st.sum() / len(keys),
                         parlays=int(r[:, 2].sum()), hits=int(r[:, 3].sum()), winning_weeks=int((u > 0).sum()), weeks=len(keys),
                         worst_run=dd, worst_run_weeks=dd_weeks,
                         season_median=float(np.median(tot)), season_bad_1in20=float(np.quantile(tot, 0.05)), season_good_1in20=float(np.quantile(tot, 0.95)),
                         p_season_loses=float((tot < 0).mean()), p_run_20u=float((sim_dd >= 20).mean()),
                         growth_per_season_pct=float(np.expm1(growth * SEASON_WEEKS))))
    t = pl.DataFrame(rows)
    t.write_csv(F.OUT / "portfolio.csv")
    pl.DataFrame(weeks).write_csv(F.OUT / "portfolio_weeks.csv")
    pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(60)
    print(t.with_columns(pl.col(pl.Float64).round(3)))
    return t


if __name__ == "__main__":
    main()
