"""The per-unit diagnostic (decision of 2026-10-09; rule fixed in docs/overnight_plan.md before any result): for every unit search, side and quantity,
does the shift beat a zero shift AND random pairing from the same observation set, with Bonferroni-corrected season-week intervals and >= 2 seasons?
Reported next to the gate; not used to drop units in this round.

  python run_comp_unit_diagnostic.py [--windows comp_windows.json] [--threshold T] [--out-dir DIR] [--log-dir DIR] [--limit N]

Writes comp_unit_diagnostic_rows_<window>.parquet (data/processed) and comp_unit_diagnostic_<window>.parquet next to the other results ("selected" when
--windows is given, else the base window). 2020-2024 targets; 2025 is never loaded.
"""
import argparse
import time
from pathlib import Path

import polars as pl

import config
from models import comps as C
from models import comps_spec as cs
from models import comps_sweep as SW

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=cs.SIM_THRESHOLD)
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--log-dir", type=Path, default=config.ROOT)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    pool = C.load_windowed_pool(a.windows) if a.windows else C.load_pool(C.LOG_WINDOW)
    w = "selected" if a.windows else C.LOG_WINDOW
    targets = C.backtest_targets()[: a.limit] if a.limit else C.backtest_targets()
    wf = pl.read_parquet(config.PROCESSED_DIR / "walkforward_predictions.parquet")
    zl = C.z_lookup(C.standardized_residuals(wf))
    rows, rates = SW.sweep_rows(pool, targets, zl, C.load_efficiency_counts(wf), thresholds=(a.threshold,))
    assert rows["season"].is_in(config.BACKTEST_SEASONS).all() and rates["season"].is_in(config.BACKTEST_SEASONS).all(), "holdout reached"
    diag = SW.unit_diagnostic(rows, rates)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(a.out_dir / f"comp_unit_diagnostic_rows_{w}.parquet")
    diag.write_parquet(a.log_dir / f"comp_unit_diagnostic_{w}.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=230, float_precision=4, fmt_str_lengths=34):
        print(diag.filter(pl.col("n_rows") > 0).select("search", "side", "quantity", "match_rate", "n_rows", "improvement_zero", "ci_lo_zero",
                                                       "improvement_random", "ci_lo_random", "seasons_won_zero", "seasons_won_random", "label"))
    print(diag.group_by("label").len().sort("label"))
    print(f"threshold {a.threshold}, {diag['bonferroni_m'][0]} cells (interval level {diag['interval_level'][0]:.5f}), {len(targets):,} targets in "
          f"{time.time() - t0:.0f}s; holdout {config.HOLDOUT_SEASON} untouched")
