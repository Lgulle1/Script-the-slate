"""The team-level sanity test (models/comps_team_test.py; fixed in docs/overnight_plan.md before it ran).

  python run_comp_team_test.py [--out-dir DIR] [--log-dir DIR]

Writes comp_team_test_rows.parquet (data/processed) and comp_team_test.parquet (per market and row set: the gains of (b) over (a) and over
(b-random) with season-week intervals, the MAEs and PASS) next to the other results. 2020-2024 targets, past games from 2016; 2025 is never loaded.
"""
import argparse
import time
from pathlib import Path

import polars as pl

import config
from models import comps as C
from models import comps_team_test as TT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--log-dir", type=Path, default=config.ROOT)
    a = ap.parse_args()
    t0 = time.time()
    rows = TT.test_rows(C.load_pool(C.LOG_WINDOW), TT.team_outcomes())
    assert rows["season"].is_in(config.BACKTEST_SEASONS).all(), "holdout reached"
    v = TT.verdicts(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(a.out_dir / "comp_team_test_rows.parquet")
    v.write_parquet(a.log_dir / "comp_team_test.parquet")
    with pl.Config(tbl_cols=20, tbl_width_chars=220, float_precision=3):
        print(v)
    print(f"{rows.height:,} rows in {time.time() - t0:.0f}s; holdout {config.HOLDOUT_SEASON} untouched")
