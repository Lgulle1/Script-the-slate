"""Phase 6.1 / 6.2: evaluate the stage-1 shrinkage blend and the stage-2 stacking model on the committed 2020-2024 walk-forward predictions (rules in docs/overnight_plan.md).

Writes data/processed/meta_results.parquet. Development seasons only; no line; 2025 never opened. Verdicts use the one weight rule (eval.compare.verdict)."""
from __future__ import annotations

import numpy as np
import polars as pl

import config
from eval import compare
from eval import meta as M
from live.results import BEST_BASELINE

PRED = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
INJ = config.PROCESSED_DIR / f"injury_ablation_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
METHODS = ["last3", "season_avg", "recency", "role_avg", "blend_70_30"]


def rows_for(market: str) -> pl.DataFrame:
    a = pl.read_parquet(PRED).filter((pl.col("market") == market) & (pl.col("kind") == "player") & (pl.col("season") <= M.LAST)).drop_nulls(["prediction", "actual"])
    key = ["entity", "season", "week"]
    wide = a.pivot(on="method", index=[*key], values="prediction")
    act = a.select(*key, "actual").unique(key)
    d = wide.join(act, on=key).rename({"vol_x_eff": "model"}).with_columns(baseline=pl.col(BEST_BASELINE[market]), market=pl.lit(market))
    inj = pl.read_parquet(INJ).filter((pl.col("market") == market) & (pl.col("kind") == "player")).select(*key, inj=pl.col("prediction")).unique(key)
    d = d.join(inj, on=key, how="left").drop_nulls(["model", "baseline", "actual"])
    return d.with_columns([pl.col(c).fill_null(pl.col("model")) for c in METHODS + ["inj"] if c in d.columns]).with_columns(cluster=pl.col("season") * 100 + pl.col("week"))


def gain(base_abs: np.ndarray, new_abs: np.ndarray, cluster, seasons) -> dict:
    diff = base_abs - new_abs
    lo, hi = compare.cluster_bootstrap(diff, cluster)
    won = int(sum(diff[seasons == s].mean() > 0 for s in np.unique(seasons)))
    return dict(gain=float(diff.mean()), gain_pct=float(100 * diff.mean() / base_abs.mean()), ci_lo=lo, ci_hi=hi, seasons_won=won, verdict=compare.verdict(float(diff.mean()), lo, hi, won))


def main() -> None:
    out = []
    for m in BEST_BASELINE:
        r = rows_for(m)
        s1 = M.stage1(r, m).filter(pl.col("season") >= 2021)
        y, mod = s1["actual"].to_numpy(), s1["model"].to_numpy()
        g1 = gain(np.abs(mod - y), np.abs(s1["final"].to_numpy() - y), s1["cluster"].to_numpy(), s1["season"].to_numpy())
        row = dict(market=m, n=s1.height, baseline=BEST_BASELINE[m], model_mae=float(np.abs(mod - y).mean()), stage1_mae=float(np.abs(s1["final"].to_numpy() - y).mean()),
                   mean_a_used=float(s1["a_used"].mean()), stage1=g1["verdict"], **{f"s1_{k}": v for k, v in g1.items() if k != "verdict"},
                   stage2_active=M.stage2_active(r, m))
        for s in (2021, 2022, 2023, 2024):
            sub = s1.filter(pl.col("season") == s)
            row[f"a_used_{s}"] = float(sub["a_used"].mean()) if sub.height else None
        if row["stage2_active"]:
            s2 = M.stage2(r, m, features=["model", "baseline", "last3", "season_avg", "recency", "role_avg", "blend_70_30", "inj"])
            j = s2.join(s1.select("entity", "season", "week", "final"), on=["entity", "season", "week"], how="inner")
            yy = j["actual"].to_numpy()
            g2 = gain(np.abs(j["final"].to_numpy() - yy), np.abs(j["pred"].to_numpy() - yy), j["cluster"].to_numpy(), j["season"].to_numpy())
            g2m = gain(np.abs(j["model"].to_numpy() - yy), np.abs(j["pred"].to_numpy() - yy), j["cluster"].to_numpy(), j["season"].to_numpy())
            row.update(stage2=g2["verdict"], s2_vs_stage1_gain=g2["gain"], s2_vs_stage1_pct=g2["gain_pct"], s2_ci_lo=g2["ci_lo"], s2_ci_hi=g2["ci_hi"], s2_seasons_won=g2["seasons_won"],
                       s2_vs_model_pct=g2m["gain_pct"], s2_vs_model_verdict=g2m["verdict"], s2_n=j.height)
        out.append(row)
    t = pl.DataFrame(out, infer_schema_length=None)
    t.write_parquet(config.PROCESSED_DIR / "meta_results.parquet")
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(240)
    print(t.select("market", "n", "model_mae", "stage1_mae", pl.col("mean_a_used").round(2), pl.col("s1_gain_pct").round(2), "s1_ci_lo", "s1_ci_hi", "s1_seasons_won", "stage1", "stage2_active",
                   *[c for c in ("s2_vs_stage1_pct", "s2_vs_model_pct", "s2_seasons_won", "stage2", "s2_vs_model_verdict") if c in t.columns]))


if __name__ == "__main__":
    main()
