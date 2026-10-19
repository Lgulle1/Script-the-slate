"""Exam of comps v3 chains through Mark 3's picker (2026-10-19). Setups fixed before it runs:
  A "Mark 3 + v3 chains vs line"        per row: dfn_eff, int_eff, sit_eff, chains_mean, agree, n_chains (chains scored against the book's line)
  B "Mark 3 + v3 chains vs projection"  the same columns, chains scored against our projection
Exam = 2025 + 2026 (never used to choose); 2022-2024 reported. Chains for every row use only earlier games.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comps2_exam as CE  # noqa: E402

M3T, IB, CM, TN, T2 = CE.M3T, CE.IB, CE.CM, CE.TN, CE.T2
GROUPS, EXAM, OTHER = CE.GROUPS, CE.EXAM, CE.OTHER
COLS = ("dfn_eff", "int_eff", "sit_eff", "chains_mean", "agree", "n_chains")


def chains(outcome):
    from comps3 import chains as CH
    path = CH.OUT.with_name(f"comps3_chains_all_{outcome}.parquet")
    if path.exists():
        return pl.read_parquet(path)
    x = CH.frame()
    tg = x.filter(pl.col("season") >= 2022)["rid"].to_numpy()
    c = CH.with_situation(CH.run(x, tg, outcome), x, tg, outcome)
    c.write_parquet(path)
    return c


def main():
    feats = {}
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X3 = np.hstack([TN.feats(d, layered=True), IB.schedule_feats(d)[0]])
    for outcome in ("rline", "relc"):
        c = chains(outcome)
        j = d.select("game_id", "player_id", "market").join(c.select("game_id", "player_id", "market", *COLS), on=["game_id", "player_id", "market"], how="left")
        feats[outcome] = j.select(COLS).to_numpy().astype(float)
        print(outcome, "rows with a chain value:", int((~np.isnan(feats[outcome][:, 3])).sum()), "of", d.height, flush=True)
    setups = {"Mark 3": X3, "Mark 3 + v3 chains vs line": np.hstack([X3, feats["rline"]]), "Mark 3 + v3 chains vs projection": np.hstack([X3, feats["relc"]])}
    res, weekly = {}, {}
    for name, X in setups.items():
        for g, seeds in GROUPS.items():
            W = IB.week_rows(d, A, IB.hidden_preds(d, A, X, [m for s in seeds for m in M3T.makers(s)]))
            weekly[(name, g)] = CM.weekly(W)
            per = IB.run_rule(W)
            ex = IB.summary({s: v for s, v in per.items() if s in EXAM})
            ot = IB.summary({s: v for s, v in per.items() if s in OTHER})
            res.setdefault(name, {})[g] = dict(exam_parlays=ex["roi"], exam_stake=ex["stake"], other_parlays=ot["roi"],
                                              exam_singles=IB.singles({sw: r for sw, r in W.items() if sw[0] in EXAM}),
                                              other_singles=IB.singles({sw: r for sw, r in W.items() if sw[0] in OTHER}))
            x_ = res[name][g]
            print(f"{name:<36} {g:<12} EXAM parlays {x_['exam_parlays']:+.1%} on {x_['exam_stake']:.0f} | 2022-24 {x_['other_parlays']:+.1%} | singles exam {x_['exam_singles']:+.1%} 2022-24 {x_['other_singles']:+.1%}", flush=True)
    for name in list(setups)[1:]:
        res[name]["p_better"] = {}
        for g in GROUPS:
            for lab, seas in (("exam", EXAM), ("2022-24", OTHER)):
                a = {k: v for k, v in weekly[("Mark 3", g)].items() if k[0] in seas}
                b = {k: v for k, v in weekly[(name, g)].items() if k[0] in seas}
                res[name]["p_better"][f"{g} {lab}"] = dict(parlays=CM.compare(a, b, "parlays")["p_mark3_better"], singles=CM.compare(a, b, "singles")["p_mark3_better"])
        print(name, "chance better than Mark 3:", res[name]["p_better"], flush=True)
    (config.ROOT / "records" / "tuning" / "comps3_v3_exam.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
