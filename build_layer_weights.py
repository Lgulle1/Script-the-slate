"""The weight of every Phase 4 layer per market under the one rule for every layer (config.LAYER_WEIGHT_RULE, decision of 2026-10-09): 1.0 on
CLEARS, otherwise 0.0 and flagged; the market always stays in the build. Writes layer_weights.csv (layer, market, verdict, weight, source, note),
the table Phase 6 starts from.

  python build_layer_weights.py [--injury-results FILE] [--pending-4a-rerun]

--pending-4a-rerun: the 4a weights come from the first 4a ablation (335752f), which predates the role-share fixes in models/injuries.py; its
markets that the old rule kept but the new rule does not are flagged "pending 4a rerun".
"""
import argparse
from pathlib import Path

import polars as pl

import config
from eval import compare

OLD_RULE_KEPT = lambda r: r["gain"] > 0 and r["seasons_won"] >= config.MIN_SEASONS_WON          # the 4a / 4b rule before 2026-10-09


def rows_4a(path: Path, pending: bool) -> list:
    out = []
    for r in pl.read_parquet(path).iter_rows(named=True):
        w = compare.layer_weight(r["verdict"])
        note = "pending 4a rerun (EDGE: kept under the old rule, 0 under the new)" if (pending and w == 0.0 and OLD_RULE_KEPT(r)) else ""
        out.append(dict(layer="4a_injuries", market=r["market"], verdict=r["verdict"], weight=w, source=path.name, note=note))
    return out


def rows_4b(path: Path) -> list:
    out = []
    for r in pl.read_parquet(path).iter_rows(named=True):
        v = r["mean_verdict"] if r["mean_verdict"] is not None else r["brier_verdict"]          # moneyline is judged on Brier only
        out.append(dict(layer="4b_simulation", market=r["market"], verdict=v, weight=compare.layer_weight(v), source=path.name, note=""))
    return out


def rows_4c(path: Path) -> list:
    return [dict(layer="4c_comparables", market=r["market"], verdict=r["verdict"], weight=0.0, source=path.name,
                 note="parked for V1 (stopping rule of 2026-10-09: no threshold passes the sweep, fingerprint diagnostic failed)")
            for r in pl.read_parquet(path).iter_rows(named=True)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--injury-results", type=Path, default=config.result_path("injury_ablation_results"))
    ap.add_argument("--pending-4a-rerun", action="store_true")
    a = ap.parse_args()
    rows = (rows_4a(a.injury_results, a.pending_4a_rerun) + rows_4b(config.result_path("sim_backtest_results_anchored"))
            + rows_4c(config.result_path("comps_ablation_results_recency_weighted")))
    t = pl.DataFrame(rows).with_columns(rule=pl.lit(config.LAYER_WEIGHT_RULE))
    t.write_csv(config.ROOT / "layer_weights.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, fmt_str_lengths=70):
        print(t.drop("rule", "source"))
