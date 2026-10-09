"""Run the five comparable searches (4c.3) for every 2020-2024 backtest target and log how often each returns no_match.

  python build_comp_searches.py [--window recency_weighted] [--limit N]

Writes data/processed/comp_search_summary.parquet (one row per target and search) and comp_search_log.parquet (per season, market, search, unit).
The window is the one the log is run on; 4c.6 selects a window per unit and market. Walk-forward: a target at week K sees only games before K."""
import argparse
import time

import polars as pl

import config
from models import comps as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default=C.LOG_WINDOW)
    ap.add_argument("--limit", type=int, default=None, help="only the first N targets (development)")
    a = ap.parse_args()
    t0 = time.time()
    pool = C.load_pool(a.window)
    targets = C.backtest_targets()
    if a.limit:
        targets = targets[: a.limit]
    print(f"{len(targets):,} targets; pool ready in {time.time() - t0:.0f}s", flush=True)
    summary = C.run_search_log(pool, targets)
    log = C.aggregate_search_log(summary)
    summary.write_parquet(config.PROCESSED_DIR / "comp_search_summary.parquet")
    log.write_parquet(config.ROOT / "comp_search_log.parquet")
    print(f"done in {time.time() - t0:.0f}s")
    pl.Config.set_tbl_rows(80)
    print(log.filter(pl.col("unit") == "combined").group_by("market", "search").agg(pl.col("no_match_rate").mean().round(3)).sort("market", "search"))
    print(f"\nholdout {config.HOLDOUT_SEASON} untouched")
