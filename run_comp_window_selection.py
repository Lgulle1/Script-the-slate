"""4c.6.1: choose the fingerprint window per unit and market from 2020-2024 walk-forward error (models/comps_windows.py has the criterion).

  python run_comp_window_selection.py [--out-dir DIR] [--log-dir DIR]

Writes comp_window_errors.parquet (per target, market, unit, side and window family; data/processed), and next to the other results
comp_window_selection.parquet (per market, unit and window family: the error and the number of targets compared) and comp_windows.json, the chosen
window per market and unit (the config the comparable searches read). Prints the choice. 2025 is never loaded.
"""
import argparse
import json
import time
from pathlib import Path

import polars as pl

import config
from models import comps as C
from models import comps_windows as W

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--log-dir", type=Path, default=config.ROOT)
    a = ap.parse_args()
    t0 = time.time()
    targets = C.backtest_targets()
    zl = C.z_lookup(C.standardized_residuals(pl.read_parquet(config.PROCESSED_DIR / "walkforward_predictions.parquet")))
    parts = []
    for wf in W.WINDOW_ORDER:
        if wf != "continuity_weighted":
            parts.append(W.unit_retrieval_errors(C.load_pool(wf), targets, zl).with_columns(window_family=pl.lit(wf)))
        else:                                                # each market reads the continuity variant of its own penalty row
            for key in sorted({config.BASELINE_TO_PENALTY_MARKET[m] for ms in C.MARKET_FAMILIES.values() for m in ms}):
                fams = {f for f, ms in C.MARKET_FAMILIES.items() if any(config.BASELINE_TO_PENALTY_MARKET[m] == key for m in ms)}
                tg = [t for t in targets if C.FAMILY_OF_MARKET[t.market] in fams]
                e = W.unit_retrieval_errors(C.load_pool(f"continuity_weighted:{key}"), tg, zl)
                e = e.filter(pl.col("market").replace_strict(config.BASELINE_TO_PENALTY_MARKET) == key)
                parts.append(e.with_columns(window_family=pl.lit(wf)))
        print(f"{wf}: done at {time.time() - t0:.0f}s", flush=True)
    errors = pl.concat(parts).sort("window_family", "market", "unit", "side", "season", "week", "game_id", "team", "player_id", nulls_last=True)
    summary, chosen = W.choose(errors, zl)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    errors.write_parquet(a.out_dir / "comp_window_errors.parquet")
    summary.write_parquet(a.log_dir / "comp_window_selection.parquet")
    choice = {m: {r["unit"]: W.window_name(r["window_family"], m) for r in chosen.filter(pl.col("market") == m).iter_rows(named=True)}
              for m in sorted(chosen["market"].unique().to_list())}
    (a.log_dir / "comp_windows.json").write_text(json.dumps(choice, indent=1, sort_keys=True) + "\n")
    with pl.Config(tbl_rows=120, tbl_width_chars=200):
        print(summary.pivot(on="window_family", index=["market", "unit"], values="error").sort("market", "unit"))
        print(chosen)
    print(f"done in {time.time() - t0:.0f}s; holdout {config.HOLDOUT_SEASON} untouched")
