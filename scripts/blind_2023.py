"""Blind test of the locked strategies on 2023 (yours, 2026-10-15). Nothing about 2023 was used to choose any rule: the picker (six versions averaged, the locked
setup) is trained once on every graded 2024-2026 line, then applied unchanged to every 2023 regular-season week; strategies A and B (scripts/strategy.py) are
applied exactly as locked. Main version: the four markets we bet at FanDuel now (receptions, receiving yards, rush yards, pass yards). Second version: the six
markets the picker was built on (adds rush attempts and pass attempts, which FanDuel did post in 2023).

  python scripts/blind_2023.py      # writes records/cards/blind_2023.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CR, ST, RP = _load("card_replay"), _load("strategy"), _load("make_record_page")
CF, TN = CR.CF, CR.TN
FOUR = ["rec", "rec_yds", "rush_yds", "pass_yds"]
SIX = ["rec", "rec_yds", "rush_yds", "pass_yds", "rush_att", "pass_att"]


def main():
    d = TN.build()
    A = TN.Arr(d)
    X = TN.feats(d, layered=True)
    d = d.with_columns(history=pl.Series(CR.history_counts(d)))
    season = d["season"].to_numpy()
    graded = ~np.isnan(A.over_won.astype(float))
    tr = np.flatnonzero((season >= 2024) & graded)
    te = np.flatnonzero(season == 2023)
    print(f"training lines (2024-2026, graded): {len(tr)}; 2023 lines: {len(te)}", flush=True)
    p_over = np.mean([m().fit(X[tr], A.over_won[tr]).predict_proba(X[te])[:, 1] for m in CF.ensemble_makers()], axis=0)
    p_under = 1 - p_over
    x = d[te].with_columns(picker_under_chance=pl.Series(p_under), edge=pl.Series(p_under - A.be_u[te]),
                           won=1 - pl.col("over_won"), profit=pl.when(pl.col("over_won") == 0).then(pl.col("dec_u") - 1).otherwise(-1.0))
    x = x.filter((pl.col("edge") >= 0.04) & (pl.col("history") >= CF.MIN_HISTORY))
    out = {}
    for name, mk in (("four", FOUR), ("six", SIX)):
        y = x.filter(pl.col("market").is_in(mk))
        weeks = []
        for (s, w), g in sorted(y.group_by("season", "week"), key=lambda t: t[0]):
            weeks.append(dict(season=int(s), week=int(w), **RP.ledger(g.to_dicts())))
        out[name] = weeks
        for k in ST.STRATEGIES:
            sg = [r for wk in weeks for r in wk[k]["singles"]]
            pr = [r for wk in weeks for r in wk[k]["parlays"]]
            su, pu = sum(r["profit"] for r in sg), sum(r["profit"] for r in pr)
            print(f"{name} markets, {k}: singles {sum(r['won'] for r in sg):.0f}-{len(sg) - sum(r['won'] for r in sg):.0f} ({su / len(sg):+.1%}), "
                  f"parlays {sum(r['profit'] > 0 for r in pr)}-{sum(r['profit'] <= 0 for r in pr)} ({pu / len(pr):+.0%}), all {su + pu:+.1f}u on {len(sg) + len(pr)} ({(su + pu) / (len(sg) + len(pr)):+.1%})")
    (config.ROOT / "records" / "cards" / "blind_2023.json").write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
