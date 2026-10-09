"""The fingerprint diagnostic (models/comps_unit_signal.py; fixed in docs/overnight_plan.md before it ran).

  python run_comp_unit_signal.py [--out-dir DIR] [--log-dir DIR]

Writes comp_unit_signal_rows.parquet (data/processed) and comp_unit_signal.parquet (per unit: MAEs, gains over the trailing average and over random
neighbours with intervals, seasons won, PASS) next to the other results. 2020-2024 targets; the ledger stops at 2024, 2025 is never loaded.
"""
import argparse
import time
from pathlib import Path

import polars as pl

import config
from models import comps as C
from models import comps_unit_signal as US

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--log-dir", type=Path, default=config.ROOT)
    a = ap.parse_args()
    t0 = time.time()
    rows = US.unit_rows(C.load_pool(C.LOG_WINDOW), US.single_game_outcomes(US.load_ledger()))
    assert rows["season"].is_in(config.BACKTEST_SEASONS).all(), "holdout reached"
    v = US.verdicts(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(a.out_dir / "comp_unit_signal_rows.parquet")
    v.write_parquet(a.log_dir / "comp_unit_signal.parquet")
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=230, float_precision=4, fmt_str_lengths=36):
        print(v.select("unit", "headline", "n", "mae_a", "mae_k", "mae_r", "gain_vs_a", "ci_lo_vs_a", "ci_hi_vs_a", "gain_vs_random", "ci_lo_vs_random",
                       "seasons_won_vs_a", "passes"))
    print(f"{rows.height:,} rows in {time.time() - t0:.0f}s; holdout {config.HOLDOUT_SEASON} untouched")
