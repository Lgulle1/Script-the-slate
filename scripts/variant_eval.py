"""Evaluate projection-model variants the way Mark 3 is judged (yours, 2026-10-19).

For each variant (data/processed/proj_variants/<name>_wf.parquet, built by scripts/proj_variants.py):
  1. the line study is rebuilt on its projections (as tournament4.build_sel);
  2. Mark 3's picker (the locked inputs plus schedule inputs, six versions x five seeds) is retrained per hidden season, twice with different seeds (1-5 and 11-15);
  3. the locked rule (4 boosted 2-leg parlays from the top 8; straight bets top 8) is applied: return per unit on the hidden seasons 2021-2024 (the seasons a projection variant
     changes; 2025-2026 projections are the stored live ones) and on all six, and the chance it beats the base projections (paired week bootstrap, same weeks);
  4. accuracy: average miss of the projections against the base on the same rows, overall and on rows where a same-position teammate is missing (the Allen case: M >= 6).

  python scripts/variant_eval.py role_ctx role_hurt role_both      # writes records/tuning/variant_eval.json
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


T4, M3T = _load("tournament4"), _load("mark3_tests")
IB, CM, TN, T2 = M3T.IB, M3T.CM, M3T.TN, M3T.T2
TS, VD = TN.TM.T, T4.VD      # the SEL switch must sit on the SAME module copy that TN.build() reads (T4.TS is another copy: every variant ran on base until 2026-10-19)
GROUPS = {"seeds 1-5": (1, 2, 3, 4, 5), "seeds 11-15": (11, 12, 13, 14, 15)}
JUDGE = (2021, 2022, 2023, 2024)


def accuracy(names):
    """Average miss vs base on identical rows, all rows and rows with a teammate missing (M >= 6)."""
    from eval import backtest as bt
    from features import role_features as rf
    from features import volume_features as vf
    data = bt.load_backtest_data()
    tabs = rf.build_tables(data.player_log, vf.week_cutoffs(data.team_log), rf.injury_lookup(), "ctx")
    gid = data.player_log.select("player_id", "season", "week", "game_id").unique(["player_id", "season", "week"])
    mrows = pl.DataFrame([dict(player_id=k[0], game_id=k[1], M=v) for k, v in tabs.M.items()])
    mrows = mrows.join(gid, on=["player_id", "game_id"], how="inner").select(entity="player_id", season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32), M="M")
    base = pl.read_parquet(VD / "base_wf.parquet").filter((pl.col("kind") == "player") & pl.col("season").is_in(JUDGE) & pl.col("market").is_in(T4.T3.FOUR)).select(
        "market", "season", "week", "entity", "actual", pb="prediction").join(mrows, on=["entity", "season", "week"], how="left")
    out = {}
    for n in names:
        v = pl.read_parquet(VD / f"{n}_wf.parquet").filter(pl.col("kind") == "player").select("market", "season", "week", "entity", pv="prediction")
        j = base.join(v, on=["market", "season", "week", "entity"], how="inner").drop_nulls(["pb", "pv", "actual"])
        rows = {}
        for lab, f in (("all rows", pl.lit(True)), ("a teammate missing (M >= 6)", pl.col("M") >= 6), ("nobody missing", pl.col("M").fill_null(0) < 1)):
            x = j.filter(f)
            rows[lab] = {m: dict(n=g.height, base=float((g["pb"] - g["actual"]).abs().mean()), variant=float((g["pv"] - g["actual"]).abs().mean())) for (m,), g in x.group_by("market")}
        out[n] = rows
    return out


def main():
    names = sys.argv[1:]
    if not names:
        raise SystemExit(__doc__)
    allv = ["base"] + names
    for n in allv:
        T4.build_sel(n)
    res, weekly = {}, {}
    for n in allv:
        TS.SEL = VD / f"{n}_sel.parquet"
        d = TN.build()
        A = TN.Arr(d)
        d = d.with_columns(history=pl.Series(T2.history_counts(d)))
        X3 = np.hstack([TN.feats(d, layered=True), IB.schedule_feats(d)[0]])
        for g, seeds in GROUPS.items():
            W = IB.week_rows(d, A, IB.hidden_preds(d, A, X3, [m for s in seeds for m in M3T.makers(s)]))
            weekly[(n, g)] = CM.weekly(W)
            per = IB.run_rule(W)
            s_all = IB.summary(per)
            s_j = IB.summary({s: v for s, v in per.items() if s in JUDGE})
            res.setdefault(n, {})[g] = dict(parlays_all=s_all["roi"], parlays_2021_24=s_j["roi"], stake_2021_24=s_j["stake"], seasons={k: v["roi"] for k, v in s_all["seasons"].items()},
                                            singles_all=IB.singles(W), singles_2021_24=IB.singles({sw: r for sw, r in W.items() if sw[0] in JUDGE}))
            x = res[n][g]
            print(f"{n:<10} {g:<12} parlays 2021-24 {x['parlays_2021_24']:+.1%} on {x['stake_2021_24']:.0f} | all six {x['parlays_all']:+.1%} | singles 2021-24 {x['singles_2021_24']:+.1%}", flush=True)
    for n in names:
        res[n]["p_better_than_base"] = {}
        for g in GROUPS:
            a = {k: v for k, v in weekly[("base", g)].items() if k[0] in JUDGE}
            b = {k: v for k, v in weekly[(n, g)].items() if k[0] in JUDGE}
            res[n]["p_better_than_base"][g] = dict(parlays=CM.compare(a, b, "parlays")["p_mark3_better"], singles=CM.compare(a, b, "singles")["p_mark3_better"])
    res["accuracy"] = accuracy(names)
    (config.ROOT / "records" / "tuning" / "variant_eval.json").write_text(json.dumps(res, indent=1))
    for n in names:
        print(f"\n{n}: chance better than base (2021-24 weeks)", res[n]["p_better_than_base"])
        for lab, rows in res["accuracy"][n].items():
            print("  ", lab, {m: f"{v['variant'] - v['base']:+.3f} ({v['n']})" for m, v in sorted(rows.items())})


if __name__ == "__main__":
    main()
