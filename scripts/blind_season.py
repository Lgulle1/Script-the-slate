"""Blind test of the locked strategies A and B on a season that was never used to choose anything (yours, 2026-10-15).
The picker (locked setup) is trained once on every graded line of the given training seasons and applied unchanged to every week of the blind season;
FanDuel's four live markets; strategies exactly as locked (scripts/strategy.py). Writes records/cards/blind_<season>.json.

  python scripts/blind_season.py --season 2022 --train 2023 2024 2025 2026
"""
import argparse
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


T2, ST, RP = _load("tournament2"), _load("strategy"), _load("make_record_page")
CF, TN = T2.CF, T2.TN


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--train", type=int, nargs="+", required=True)
    a = ap.parse_args(argv)
    d = TN.build()
    A = TN.Arr(d)
    X = TN.feats(d, layered=True)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    season = d["season"].to_numpy()
    graded = ~np.isnan(A.over_won.astype(float))
    tr = np.flatnonzero(np.isin(season, a.train) & graded)
    te = np.flatnonzero(season == a.season)
    p_under = 1 - np.mean([m().fit(X[tr], A.over_won[tr]).predict_proba(X[te])[:, 1] for m in CF.ensemble_makers()], axis=0)
    x = d[te].with_columns(picker_under_chance=pl.Series(p_under), edge=pl.Series(p_under - A.be_u[te]),
                           won=1 - pl.col("over_won"), profit=pl.when(pl.col("over_won") == 0).then(pl.col("dec_u") - 1).otherwise(-1.0))
    x = x.filter((pl.col("edge") >= 0.04) & (pl.col("history") >= CF.MIN_HISTORY) & pl.col("market").is_in(T2.FOUR))
    weeks = [dict(season=int(s), week=int(w), **RP.ledger(g.to_dicts())) for (s, w), g in sorted(x.group_by("season", "week"), key=lambda t: t[0])]
    for k in ST.STRATEGIES:
        sg = [r for wk in weeks for r in wk[k]["singles"]]
        pr = [r for wk in weeks for r in wk[k]["parlays"]]
        su, pu = sum(r["profit"] for r in sg), sum(r["profit"] for r in pr)
        print(f"{a.season} {k}: singles {sum(r['won'] for r in sg):.0f}-{len(sg) - sum(r['won'] for r in sg):.0f} ({su / len(sg):+.1%}), parlays {sum(r['profit'] > 0 for r in pr)}-"
              f"{sum(r['profit'] <= 0 for r in pr)} ({pu / len(pr):+.0%}), all {su + pu:+.1f}u on {len(sg) + len(pr)} ({(su + pu) / (len(sg) + len(pr)):+.1%})")
    (config.ROOT / "records" / "cards" / f"blind_{a.season}.json").write_text(json.dumps(dict(season=a.season, train=a.train, four=weeks), default=str))


if __name__ == "__main__":
    main()
