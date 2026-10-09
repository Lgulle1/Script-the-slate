"""4c.4: standardized residuals and shifts for every 2020-2024 backtest target (build plan 4c.4).

  python build_comp_shifts.py [--window recency_weighted] [--out-dir DIR] [--log-dir DIR]

For each target (player-game or team side) and market, runs the five searches (4c.3) on one window, turns their matches into standardized shifts and
writes, under --out-dir (default data/processed, not committed):
  comp_shifts_<window>.parquet          one row per target and market: shift_vol_S1..S5, shift_eff_S1..S5, n_eff_S1..S5, nomatch_S1..S5, best_sim_S1..S5
  comp_shift_matches_<window>.parquet   the per-match table (similarity, recency, continuity, quality, final weight, capped weights, z)
  comp_shift_detail_<window>.parquet    per target, market and search: matches with an expectation, n_eff, why no_match
  comp_retrieval_change_<window>.parquet per target, market and search: the same search on the HEALTHY target vector -- overlap of the top matches,
                                         shared weight, change in the shifts (build plan 4c.1.4)
and under --log-dir (default the repo root, committed) comp_shift_summary_<window>.parquet: per season, market and search, the no-match rate and the
spread of the shifts, and comp_retrieval_summary_<window>.parquet (how much the lineup adjustment changes retrieval). z = (actual - expected) / sigma uses the comp-free walk-forward predictions (data/processed/walkforward_predictions.parquet).
Walk-forward throughout; 2025 never loaded.
"""
import argparse
import time
from pathlib import Path

import polars as pl

import config
from models import comps as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default=C.LOG_WINDOW)
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--log-dir", type=Path, default=config.ROOT)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    pool = C.load_pool(a.window)
    targets = C.backtest_targets()
    if a.limit:
        targets = targets[: a.limit]
    zt = C.standardized_residuals(pl.read_parquet(config.PROCESSED_DIR / "walkforward_predictions.parquet"))
    feats, matches, detail, retrieval = C.comp_shifts(pool, targets, C.z_lookup(zt))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    w = a.window.replace(":", "_")
    feats.write_parquet(a.out_dir / f"comp_shifts_{w}.parquet")
    matches.write_parquet(a.out_dir / f"comp_shift_matches_{w}.parquet")
    detail.write_parquet(a.out_dir / f"comp_shift_detail_{w}.parquet")
    retrieval.write_parquet(a.out_dir / f"comp_retrieval_change_{w}.parquet")
    summary = C.shift_summary(feats, detail)
    summary.write_parquet(a.log_dir / f"comp_shift_summary_{w}.parquet")
    rsum = C.retrieval_summary(retrieval)
    rsum.write_parquet(a.log_dir / f"comp_retrieval_summary_{w}.parquet")
    print(f"{feats.height:,} target-market rows, {matches.height:,} matches in {time.time() - t0:.0f}s")
    pl.Config.set_tbl_rows(80)
    print(summary.group_by("market", "search").agg(pl.col("nomatch_rate").mean().round(3), pl.col("mean_abs_shift_vol").mean().round(4)).sort("market", "search"))
    print(rsum.group_by("search").agg(pl.col("share_adjusted_differs").mean().round(3), pl.col("mean_overlap_top_k").mean().round(3),
                                      pl.col("mean_abs_shift_vol_change").mean().round(4)).sort("search"))
    print(f"\nholdout {config.HOLDOUT_SEASON} untouched")
