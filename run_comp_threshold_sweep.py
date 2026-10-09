"""The SIM_THRESHOLD sweep (decision of 2026-10-09; the rule is in models/comps_sweep.py and was fixed in docs/overnight_plan.md first).

  python run_comp_threshold_sweep.py [--out-dir DIR] [--log-dir DIR] [--limit N]

Writes comp_threshold_sweep_rows.parquet (per threshold, target, search and side: the real and the random-pairing squared error; data/processed) and,
next to the other results, comp_threshold_sweep.parquet (per threshold: the pooled improvement, its season-week bootstrap interval, the rows) and
comp_threshold_sweep_detail.parquet (per threshold, market, search and side: the match rate and the improvement, for reading only). Prints the
threshold the rule chooses. Base window (recency_weighted), 2020-2024 targets; 2025 is never loaded.
"""
import argparse
import time
from pathlib import Path

import polars as pl

import config
from models import comps as C
from models import comps_sweep as SW

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--log-dir", type=Path, default=config.ROOT)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    pool = C.load_pool(C.LOG_WINDOW)
    targets = C.backtest_targets()[: a.limit] if a.limit else C.backtest_targets()
    wf = pl.read_parquet(config.PROCESSED_DIR / "walkforward_predictions.parquet")
    zl = C.z_lookup(C.standardized_residuals(wf))
    rows, rates = SW.sweep_rows(pool, targets, zl, C.load_efficiency_counts(wf))
    assert rows["season"].is_in(config.BACKTEST_SEASONS).all() and rates["season"].is_in(config.BACKTEST_SEASONS).all(), "holdout reached"
    summary = SW.summarize(rows)
    detail = SW.breakdown(rows, rates)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(a.out_dir / "comp_threshold_sweep_rows.parquet")
    summary.write_parquet(a.log_dir / "comp_threshold_sweep.parquet")
    detail.write_parquet(a.log_dir / "comp_threshold_sweep_detail.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=220, float_precision=4):
        print(summary)
        print(detail.filter(pl.col("side").is_null() | (pl.col("side") == "vol")).pivot(on="threshold", index=["market", "search"], values="match_rate")
              .sort("market", "search"))
    print(f"chosen SIM_THRESHOLD: {SW.choose_threshold(summary)} (rule: the lowest threshold whose pooled improvement is above zero with the interval's "
          f"lower bound above zero; none: {SW.DEFAULT_THRESHOLD})")
    print(f"{len(targets):,} targets, {rows.height:,} scored rows in {time.time() - t0:.0f}s; holdout {config.HOLDOUT_SEASON} untouched")
