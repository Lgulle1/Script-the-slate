"""Export the Mark 4 forward results for the Lab page (records/lab/*.csv).

  python -m mark4.lab_export
"""
from __future__ import annotations

import json

import polars as pl

from mark4 import forward as F

# internal setup names -> what the page calls them
NAMES = {"Mark 3 (forward)": "Mark 3 (frozen)", "Mark 3 inputs + recency": "Mark 4.0: recency-weighted picker",
         "Mark 4a: + situations, recency": "Mark 4a: + blind-spot situations", "Mark 4b: + situations + comps, recency": "Mark 4b: + situations + comps",
         "Mark 4c: + situations + role-event games + comps, recency": "Mark 4c: + situations + role-change games + comps"}


def results(seeds="1-2-3"):
    """Every setup whose forward predictions are saved, scored now (no retraining)."""
    import numpy as np
    d, A, *_ = F.frame()
    m3 = F.strat_name(dict(kind="mark3", n=8, edge=0.04, sides="under", markets="four"))
    out = {}
    for k in F.SETUPS:
        pf = F.OUT / f"forward_p_{k.split(':')[0].replace(' ', '_')}_{seeds}.npy"
        if not pf.exists():
            continue
        p = np.load(pf)
        _, chosen = F.evaluate(d, A, p, k)
        _, rule = F.evaluate(d, A, p, k, fixed=m3)
        out[k] = dict(chosen=chosen, summary=F.summarize(chosen), mark3_rule=rule, mark3_rule_summary=F.summarize(rule))
    return out


def main(seeds="1-2-3"):
    res = results(seeds)
    rows, summ = [], []
    for k, v in res.items():
        name = NAMES.get(k, k)
        for s, c in v["chosen"].items():
            rows.append(dict(setup=name, rule="Strategy chosen forward", season=int(s), ret=c["roi"], staked=c["staked"], units=c["units"], strategy=c["strategy"]))
        for s, c in v["mark3_rule"].items():
            rows.append(dict(setup=name, rule="Mark 3's rule", season=int(s), ret=c["roi"], staked=c["staked"], units=c["units"], strategy=c["strategy"]))
        for rule, key in (("Strategy chosen forward", "summary"), ("Mark 3's rule", "mark3_rule_summary")):
            sm = v[key]
            summ.append(dict(key=f"{name} | {rule}", setup=name, rule=rule, recency_ret=sm["recency_roi"], units=sm["units"], staked=sm["staked"]))
    pl.DataFrame(rows).with_columns(key=pl.concat_str(["setup", "rule", pl.col("season").cast(pl.Utf8)], separator=" | ")).write_csv(F.OUT / "forward_seasons.csv")
    pl.DataFrame(summ).write_csv(F.OUT / "forward_summary.csv")
    cf = F.OUT / f"compare_{seeds}.json"
    if cf.exists():
        c = json.loads(cf.read_text())
        h = [dict(key=f"{NAMES.get(k, k)} | {rule}", challenger=NAMES.get(k, k), rule={"mark3_rule": "Mark 3's rule", "chosen_strategy": "Strategy chosen forward"}[rule],
                  mark3=r["a"], challenger_ret=r["b"], p_better=r["p_b_better"], weeks=r["weeks"]) for k, v in c.items() for rule, r in v.items()]
        pl.DataFrame(h).write_csv(F.OUT / "forward_h2h.csv")
    print(pl.read_csv(F.OUT / "forward_summary.csv"))


if __name__ == "__main__":
    main()
