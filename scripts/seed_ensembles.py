"""Seed-averaged pickers (yours, 2026-10-16). mark3_tests.py showed that one random seed moves the hidden-season parlay return from +20% to +50% for the same
picker, so single-seed comparisons cannot separate Mark 2 from Mark 3. Here each picker averages its six model versions over 5 seeds (30 models), and every
variant is run with two separate seed groups (1-5 and 11-15): a result counts only if both groups agree.

Variants: Mark 2; Mark 3; Mark 3 without primetime; Mark 3 plus travel; Mark 3 plus weather (game-day values: not available before kickoff, shown for reference).

  python scripts/seed_ensembles.py      # writes records/tuning/seed_ensembles.json
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


M3T = _load("mark3_tests")
IB, CM = M3T.IB, M3T.CM
TN, T2 = IB.TN, IB.T2
GROUPS = {"seeds 1-5": (1, 2, 3, 4, 5), "seeds 11-15": (11, 12, 13, 14, 15)}


def main():
    t0 = time.time()
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X = TN.feats(d, layered=True)
    sched, wx = IB.schedule_feats(d)
    trav = M3T.travel_feats(d)
    no_prime = [c for c in range(sched.shape[1]) if c not in M3T.SCHED_GROUPS["primetime"]]
    variants = {"Mark 2": X, "Mark 3": np.hstack([X, sched]), "Mark 3 without primetime": np.hstack([X, sched[:, no_prime]]),
                "Mark 3 plus travel": np.hstack([X, sched, trav]), "Mark 3 plus weather (not available live)": np.hstack([X, sched, wx])}
    out, weekly = {}, {}
    for g, seeds in GROUPS.items():
        mk = [m for s in seeds for m in M3T.makers(s)]
        for name, Xv in variants.items():
            W = IB.week_rows(d, A, IB.hidden_preds(d, A, Xv, mk))
            r = IB.summary(IB.run_rule(W))
            weekly[(g, name)] = CM.weekly(W)
            out.setdefault(name, {})[g] = dict(parlays=r["roi"], parlay_stake=r["stake"], seasons={s: v["roi"] for s, v in r["seasons"].items()},
                                               singles=IB.singles(W), three_leg=IB.summary(IB.run_rule(W, n=9, k=3))["roi"])
            x = out[name][g]
            print(f"[{time.time() - t0:5.0f}s] {g:<12} {name:<42} parlays {x['parlays']:+.1%} on {x['parlay_stake']:.0f}  3-leg {x['three_leg']:+.1%}  singles {x['singles']:+.1%}", flush=True)
    for name in variants:
        if name == "Mark 2":
            continue
        out[name]["p_better_than_mark2"] = {g: dict(parlays=CM.compare(weekly[(g, "Mark 2")], weekly[(g, name)], "parlays")["p_mark3_better"],
                                                   singles=CM.compare(weekly[(g, "Mark 2")], weekly[(g, name)], "singles")["p_mark3_better"]) for g in GROUPS}
    (config.ROOT / "records" / "tuning" / "seed_ensembles.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v.get("p_better_than_mark2") for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
