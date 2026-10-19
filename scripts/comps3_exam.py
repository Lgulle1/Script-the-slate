"""Exam of comps scored against the line (comps2/line_comps.py) through Mark 3's picker (2026-10-19).

Fixed on the practice seasons 2022-2023 BEFORE this runs (records/tuning/comps2_ladder.csv, label "vs line: searches"): the defense's recent results vs the
line (dfn, no similarity: receiving 0.079, rushing 0.044) and the league-wide similar offense-defense pairs vs the line (int, h 0.75: receiving 0.048) are the
inputs; the player's own history and the situation search vs the line were flat or negative and are left out. Two setups, both chosen now:
  "Mark 3 + defense vs line + matchup vs line"   dfn_shift, int_shift
  "Mark 3 + defense vs line"                       dfn_shift
Exam = 2025 + 2026 (never used to choose); 2022-2024 reported. Each row's comps use only earlier games.
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
import comps2_exam as CE  # noqa: E402  (loads mark3_tests and friends)

M3T, IB, CM, TN, T2 = CE.M3T, CE.IB, CE.CM, CE.TN, CE.T2
GROUPS, EXAM, OTHER = CE.GROUPS, CE.EXAM, CE.OTHER


def main():
    from comps2 import line_comps as LC
    x = LC.frame()
    c = LC.components(x, (2022, 2023, 2024, 2025, 2026))
    c.write_parquet(LC.OUT)
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X3 = np.hstack([TN.feats(d, layered=True), IB.schedule_feats(d)[0]])
    j = d.select("game_id", "player_id", "market").join(c.select("game_id", "player_id", "market", "dfn_shift", "int_shift"), on=["game_id", "player_id", "market"], how="left")
    F2 = j.select("dfn_shift", "int_shift").to_numpy().astype(float)
    print("rows with a line-comps value:", int((~np.isnan(F2[:, 0])).sum()), "of", d.height, flush=True)
    setups = {"Mark 3": X3, "Mark 3 + defense vs line + matchup vs line": np.hstack([X3, F2]), "Mark 3 + defense vs line": np.hstack([X3, F2[:, :1]])}
    res, weekly = {}, {}
    for name, X in setups.items():
        for g, seeds in GROUPS.items():
            W = IB.week_rows(d, A, IB.hidden_preds(d, A, X, [m for s in seeds for m in M3T.makers(s)]))
            weekly[(name, g)] = CM.weekly(W)
            per = IB.run_rule(W)
            ex = IB.summary({s: v for s, v in per.items() if s in EXAM})
            ot = IB.summary({s: v for s, v in per.items() if s in OTHER})
            res.setdefault(name, {})[g] = dict(exam_parlays=ex["roi"], exam_stake=ex["stake"], other_parlays=ot["roi"], seasons={k: v["roi"] for k, v in IB.summary(per)["seasons"].items()},
                                              exam_singles=IB.singles({sw: r for sw, r in W.items() if sw[0] in EXAM}),
                                              other_singles=IB.singles({sw: r for sw, r in W.items() if sw[0] in OTHER}))
            x_ = res[name][g]
            print(f"{name:<45} {g:<12} EXAM parlays {x_['exam_parlays']:+.1%} on {x_['exam_stake']:.0f} | 2022-24 {x_['other_parlays']:+.1%} | singles exam {x_['exam_singles']:+.1%} 2022-24 {x_['other_singles']:+.1%}", flush=True)
    for name in list(setups)[1:]:
        res[name]["p_better"] = {}
        for g in GROUPS:
            for lab, seas in (("exam", EXAM), ("2022-24", OTHER)):
                a = {k: v for k, v in weekly[("Mark 3", g)].items() if k[0] in seas}
                b = {k: v for k, v in weekly[(name, g)].items() if k[0] in seas}
                res[name]["p_better"][f"{g} {lab}"] = dict(parlays=CM.compare(a, b, "parlays")["p_mark3_better"], singles=CM.compare(a, b, "singles")["p_mark3_better"])
        print(name, "chance better than Mark 3:", res[name]["p_better"], flush=True)
    (config.ROOT / "records" / "tuning" / "comps3_exam.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
