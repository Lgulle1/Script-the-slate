"""Overreaction strength (2026-10-22): Mark 4i with the promotion-overreaction signal blended in at strength k instead of overriding.

  p = p4i + k * (p4l - p4i) on flagged lines (k = 0 is Mark 4i, k = 1 is Mark 4l's override), every other line untouched.
  Each k is played on the live card (mark4/live.py PLAN: 5-leg 0.25u, 3-leg 0.5u, 2 x 2-leg 1u, boosted) week by week 2022 - 2026 week 6.
  Forward choice: each season's k is the one with the best recency-weighted return per unit staked on EARLIER seasons only (2022 warm-up).

  python -m mark4.nudge      # writes records/lab/nudge.csv and prints the comparison
"""
from __future__ import annotations

import numpy as np
import polars as pl

from mark4 import cards as C
from mark4 import compare as CP
from mark4 import forward as F
from mark4 import live as LV
from mark4 import rolechange as RC
from mark4 import tournament as T

KS = (0.0, 0.25, 0.5, 0.75, 1.0)
PLAN = [(k, c, True, s) for k, c, s in LV.PLAN]


def main():
    d, A, *_ = F.frame()
    season = d["season"].to_numpy()
    p4i = T.load_p("Mark 4i").copy()
    p4l = RC.mark4l(d, A, p4i)
    print("flagged lines per season:", {int(s): int(((p4l != p4i) & (season == s)).sum()) for s in np.unique(season)})
    R = {}
    for k in KS:
        p = p4i + k * (p4l - p4i)
        p[season < 2022] = np.nan
        W = T.week_pools(d, A, p, "both", "all", 0.0, "ev")
        R[k] = {key: C.play_week(W[key], PLAN, 0) for key in sorted(W)}
    keys = sorted(set().union(*[set(r) for r in R.values()]))
    seasons = sorted({s for s, _ in keys})

    def ret(k, ss):
        u = sum(R[k][key][0] for key in R[k] if key[0] in ss); b = sum(R[k][key][1] for key in R[k] if key[0] in ss)
        return u, b

    rows = []
    for k in KS:
        r = dict(k=k)
        for s in seasons:
            u, b = ret(k, [s]); r[f"u{s}"] = u; r[f"b{s}"] = b
        u, b = ret(k, seasons); r.update(units=u, staked=b, ret=u / b)
        u, b = ret(k, [s for s in seasons if s >= 2025]); r.update(units_2025on=u, staked_2025on=b, ret_2025on=u / b)
        rows.append(r)
    t = pl.DataFrame(rows)
    # forward choice of k: recency-weighted return per unit on earlier seasons only
    chosen, fwd = {}, {}
    for s in seasons[1:]:
        past = [x for x in seasons if x < s]
        score = {k: sum(0.5 ** (s - 1 - x) * ret(k, [x])[0] for x in past) / sum(0.5 ** (s - 1 - x) * ret(k, [x])[1] for x in past) for k in KS}
        chosen[s] = max(score, key=score.get)
        fwd.update({key: R[chosen[s]][key] for key in R[chosen[s]] if key[0] == s})
    fu = sum(v[0] for v in fwd.values()); fb = sum(v[1] for v in fwd.values())
    f25 = [v for key, v in fwd.items() if key[0] >= 2025]
    print("\nk (0 = Mark 4i, 1 = Mark 4l)  units / staked  return | 2025 on | by season")
    for r in t.iter_rows(named=True):
        print(f"  k={r['k']:.2f}: {r['units']:+7.1f} on {r['staked']:6.1f} ({r['ret']:+.0%}) | 2025 on {r['units_2025on']:+6.1f} on {r['staked_2025on']:5.1f} ({r['ret_2025on']:+.0%}) | "
              + " ".join(f"{s}: {r[f'u{s}']:+.1f}/{r[f'b{s}']:.1f}" for s in seasons))
    print(f"\nforward-chosen k by season: {chosen}")
    print(f"forward-chosen card {min(chosen)}-2026: {fu:+.1f} on {fb:.1f} ({fu / fb:+.0%}); 2025 on {sum(v[0] for v in f25):+.1f} on {sum(v[1] for v in f25):.1f}")
    for k in KS[1:]:
        print(f"  chance k={k} beats k=0 (Mark 4i), weeks resampled, recency-weighted: {CP.boot(R[0.0], R[k])['p_b_better']:.0%}")
    t.write_csv(F.OUT / "nudge.csv")
    return t, chosen


if __name__ == "__main__":
    main()
