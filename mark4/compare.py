"""Head-to-head of forward setups from their saved predictions (records/lab/forward_p_*.npy): same weeks, same betting rule, paired week bootstrap.

The chance B beats A is the share of bootstrap resamples (weeks drawn with replacement, each week weighted by recency: 0.5 per season back from 2026) in
which B's recency-weighted return is higher. Rules compared: Mark 3's rule (4 boosted 2-leg parlays from the top 8 unders, 4% edge, 4 markets) and the
strategy each setup chose forward.

  python -m mark4.compare "Mark 3 (forward)" "Mark 3 inputs + recency" --seeds 1-2-3
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from mark4 import forward as F


def weekly_units(d, A, p, spec):
    x = F.candidates(d, A, p, spec["sides"], F.IB.FOUR if spec["markets"] == "four" else None)
    return F.run_strategy(F.weekly(x, spec["edge"]), spec["kind"], spec["n"])


def chosen_weeks(d, A, p):
    res, chosen = F.evaluate(d, A, p, "")
    out = {}
    for S, c in chosen.items():
        for (s, w), v in res[c["strategy"]]["weeks"].items():
            if s == S:
                out[(s, w)] = v
    return out


def boot(a: dict, b: dict, n=4000, seed=7, target=2026):
    weeks = sorted(set(a) | set(b))
    wt = np.array([0.5 ** (target - s) for s, _ in weeks])
    A_ = np.array([[a.get(k, (0, 0, 0))[0], a.get(k, (0, 0, 0))[1]] for k in weeks])
    B_ = np.array([[b.get(k, (0, 0, 0))[0], b.get(k, (0, 0, 0))[1]] for k in weeks])
    roi = lambda M, w: (w * M[:, 0]).sum() / max((w * M[:, 1]).sum(), 1e-9)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        i = rng.integers(0, len(weeks), len(weeks))
        wins += roi(B_[i], wt[i]) > roi(A_[i], wt[i])
    return dict(a=roi(A_, wt), b=roi(B_, wt), p_b_better=wins / n, weeks=len(weeks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="+")
    ap.add_argument("--seeds", default="1-2-3")
    a = ap.parse_args()
    d, A, *_ = F.frame()
    load = lambda name: np.load(F.OUT / f"forward_p_{name.split(':')[0].replace(' ', '_')}_{a.seeds}.npy")
    m3 = dict(kind="mark3", n=8, edge=0.04, sides="under", markets="four")
    pa = load(a.a)
    wa_rule, wa_ch = weekly_units(d, A, pa, m3), chosen_weeks(d, A, pa)
    out = {}
    for nb in a.b:
        pb = load(nb)
        st20 = dict(kind="singles", n=20, edge=0.0, sides="both", markets="all")
        out[nb] = dict(mark3_rule=boot(wa_rule, weekly_units(d, A, pb, m3)), chosen_strategy=boot(wa_ch, chosen_weeks(d, A, pb)),
                       top20_straights=boot(weekly_units(d, A, pa, st20), weekly_units(d, A, pb, st20)))
        print(nb, "vs", a.a, json.dumps(out[nb], indent=1))
    (F.OUT / f"compare_{a.seeds}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
