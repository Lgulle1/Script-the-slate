"""Final exam of comps v2 through Mark 3's picker (yours, 2026-10-19).

Mark 3 = the locked picker inputs + schedule inputs, six HGB versions x five seeds, hidden-season protocol, 4 boosted 2-leg parlays from the top 8.
Mark 3 + comps = the same plus the comps outputs as picker inputs:
  model_shift   the residual model's prediction (walk-forward: season s from a model trained on seasons < s)
  gap_adj       (projection x (1 + model_shift) - line) / line
  own / dfn / int / sit shifts and agree (how many of the four components point the same way as model_shift)
The structure was fixed on the practice seasons 2021-2023 BEFORE this runs. Exam = 2025 + 2026 (never used to choose anything); 2024 is reported too
(also not used to choose). Every season's return, and the paired week bootstrap chance that Mark 3 + comps beats Mark 3 on the exam weeks.

  python scripts/comps2_exam.py        # writes records/tuning/comps2_exam.json
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
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


M3T = _load("mark3_tests")
IB, CM, TN, T2 = M3T.IB, M3T.CM, M3T.TN, M3T.T2
GROUPS = {"seeds 1-5": (1, 2, 3, 4, 5), "seeds 11-15": (11, 12, 13, 14, 15)}
EXAM = (2025, 2026)
OTHER = (2022, 2023, 2024)
COMP_COLS = ("own_shift", "dfn_shift", "int_shift", "sit_shift")


def comps_feats(d: pl.DataFrame, preds: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from comps2 import residual_model as RM
    f = RM.frame().select("rid", "game_id", "player_id", "market", *COMP_COLS)
    f = f.join(preds.select("rid", model_shift="pred"), on="rid", how="left")
    x = d.select("game_id", "player_id", "market", "line", "mean").join(f.drop("rid"), on=["game_id", "player_id", "market"], how="left")
    sh = x["model_shift"].to_numpy().astype(float)
    comps = x.select(COMP_COLS).to_numpy().astype(float)
    agree = np.where(np.isnan(sh), np.nan, np.nansum(np.sign(comps) == np.sign(sh)[:, None], 1))
    gap_adj = (x["mean"].to_numpy() * (1 + np.nan_to_num(sh)) - x["line"].to_numpy()) / x["line"].to_numpy()
    gap_adj = np.where(np.isnan(sh), np.nan, gap_adj)
    full = np.column_stack([sh, gap_adj, comps, agree])
    small = np.column_stack([sh, gap_adj])
    return full, small


def main():
    from comps2 import residual_model as RM
    preds = RM.walk_forward(RM.frame(), (2022, 2023, 2024, 2025, 2026))
    preds.write_parquet(RM.OUT)
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X3 = np.hstack([TN.feats(d, layered=True), IB.schedule_feats(d)[0]])
    full, small = comps_feats(d, preds)
    print("rows with a comps prediction:", int((~np.isnan(full[:, 0])).sum()), "of", d.height, flush=True)
    setups = {"Mark 3": X3, "Mark 3 + comps (all)": np.hstack([X3, full]), "Mark 3 + comps (shift + adjusted gap)": np.hstack([X3, small])}
    res, weekly = {}, {}
    for name, X in setups.items():
        for g, seeds in GROUPS.items():
            W = IB.week_rows(d, A, IB.hidden_preds(d, A, X, [m for s in seeds for m in M3T.makers(s)]))
            weekly[(name, g)] = CM.weekly(W)
            per = IB.run_rule(W)
            ex = IB.summary({s: v for s, v in per.items() if s in EXAM})
            ot = IB.summary({s: v for s, v in per.items() if s in OTHER})
            res.setdefault(name, {})[g] = dict(exam_parlays=ex["roi"], exam_stake=ex["stake"], other_parlays=ot["roi"], seasons={k: v["roi"] for k, v in IB.summary(per)["seasons"].items()},
                                              exam_singles=IB.singles({sw: r for sw, r in W.items() if sw[0] in EXAM}))
            x = res[name][g]
            print(f"{name:<40} {g:<12} EXAM 2025-26 parlays {x['exam_parlays']:+.1%} on {x['exam_stake']:.0f} | 2022-24 {x['other_parlays']:+.1%} | exam singles {x['exam_singles']:+.1%} | {x['seasons']}", flush=True)
    for name in list(setups)[1:]:
        res[name]["p_better_than_mark3_exam"] = {}
        for g in GROUPS:
            a = {k: v for k, v in weekly[("Mark 3", g)].items() if k[0] in EXAM}
            b = {k: v for k, v in weekly[(name, g)].items() if k[0] in EXAM}
            res[name]["p_better_than_mark3_exam"][g] = dict(parlays=CM.compare(a, b, "parlays")["p_mark3_better"], singles=CM.compare(a, b, "singles")["p_mark3_better"])
        print(name, "chance better than Mark 3 on the exam weeks:", res[name]["p_better_than_mark3_exam"], flush=True)
    (config.ROOT / "records" / "tuning" / "comps2_exam.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
