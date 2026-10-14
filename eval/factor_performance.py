"""Model health table: one row per layer and market, from layer_weights.csv and the ablation result files (plan: "Phase 7 app and Phase 6.1 / 6.2").

Columns: layer, market, verdict, weight, gain_pct, ci_lo, ci_hi, seasons_won, last_season_gain, status (on / off), red_flag (on and no gain in the most recent season).
Gain columns exist for the layers whose result files carry them (4a injuries, 4b simulation); the other layers keep their verdict and weight. Live graded weeks do not exist yet.
"""
from __future__ import annotations

import polars as pl

import config

ROOT = config.ROOT
LAST = max(config.BACKTEST_SEASONS)


def _gain_table() -> pl.DataFrame:
    parts = []
    inj = ROOT / "injury_ablation_results_eligibility_v2_rerun_20261009.parquet"
    if inj.exists():
        d = pl.read_parquet(inj)
        parts.append(d.select(layer=pl.lit("4a_injuries"), market="market", gain_pct="gain_pct", ci_lo="ci_lo", ci_hi="ci_hi", seasons_won="seasons_won", last_season_gain=f"gain_{LAST}"))
    sim = ROOT / "sim_backtest_results_anchored_eligibility_v2.parquet"
    if sim.exists():
        d = pl.read_parquet(sim)
        parts.append(d.select(layer=pl.lit("4b_simulation"), market="market", gain_pct="mean_gain_pct", ci_lo="mean_ci_lo", ci_hi="mean_ci_hi", seasons_won="mean_seasons_won", last_season_gain=f"mean_gain_{LAST}"))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame(schema={"layer": pl.String, "market": pl.String})


def combine(weights: pl.DataFrame, gains: pl.DataFrame) -> pl.DataFrame:
    t = weights.join(gains, on=["layer", "market"], how="left")
    for c in ("gain_pct", "ci_lo", "ci_hi", "last_season_gain"):
        if c not in t.columns:
            t = t.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    if "seasons_won" not in t.columns:
        t = t.with_columns(pl.lit(None, dtype=pl.Int64).alias("seasons_won"))
    t = t.with_columns(status=pl.when(pl.col("weight") > 0).then(pl.lit("on")).otherwise(pl.lit("off")))
    return t.with_columns(red_flag=(pl.col("status") == "on") & (pl.col("last_season_gain") <= 0)).select(
        "layer", "market", "verdict", "weight", "gain_pct", "ci_lo", "ci_hi", "seasons_won", "last_season_gain", "status", "red_flag", "source")


def build() -> pl.DataFrame:
    return combine(pl.read_csv(ROOT / "layer_weights.csv").select("layer", "market", "verdict", "weight", "source"), _gain_table())


if __name__ == "__main__":
    t = build()
    t.write_parquet(config.PROCESSED_DIR / "factor_performance.parquet")
    print(t.filter(pl.col("status") == "on"))
