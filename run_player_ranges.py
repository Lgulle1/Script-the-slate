"""Workstream B: player-specific ranges on the 2020-2024 walk-forward predictions (docs/overnight_plan.md). Writes data/processed/player_ranges_eval.parquet. Development seasons only; byte-identical on a second run."""
import polars as pl

import config
from eval import calibrate as C
from eval import player_ranges as PR
from run_calibration import load

OUT = config.PROCESSED_DIR / "player_ranges_eval.parquet"


def main():
    rows = load()
    cfg = C.Config()
    out = []
    for m in PR.PLAYER_MARKETS:
        for k in (PR.K_SHRINK, 4, 16):
            wf = PR.walk_forward(rows, m, cfg, k_shrink=k)
            if not wf.height:
                continue
            ev = PR.evaluate(wf)
            if not ev["folds"]:
                continue
            out.append(dict(market=m, k_shrink=k, chosen=(k == PR.K_SHRINK), n=ev["n"], gain=ev["gain"], lo=ev["lo"], hi=ev["hi"], folds_better=int(sum(x > 0 for x in ev["fold_gain"])), folds=len(ev["folds"]),
                            fold_gains=",".join(f"{x:.4f}" for x in ev["fold_gain"]), cov80_pool=ev["cov80_pool_all"], cov80_ps=ev["cov80_ps_all"], passes=ev["passes"],
                            mean_r=float(wf["r"].mean()), sd_r=float(wf["r"].std())))
    df = pl.DataFrame(out)
    df.write_parquet(OUT)
    with pl.Config(tbl_rows=40, tbl_cols=20, float_precision=4, tbl_width_chars=220):
        print(df.select("market", "k_shrink", "chosen", "n", "gain", "lo", "hi", "folds_better", "cov80_pool", "cov80_ps", "passes", "mean_r", "sd_r"))
    return df


if __name__ == "__main__":
    main()
