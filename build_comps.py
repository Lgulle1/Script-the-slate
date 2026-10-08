"""Build the 4c.1 fingerprint vectors (data/processed/comp_vectors_{team,player}.parquet) and their small logs.

  python build_comps.py [--out-dir DIR] [--log-dir DIR] [--no-lineup]

Reads the raw tables (pbp, FTN, PFR, participation, snap counts, rosters, depth charts), the 4a injury layer's expected game-day shares
(fit week by week on earlier games only) and builds, for every team-game 2016-2024 and every player-game, the unit vectors defined in
models/comps_spec.py in every window and space. 2025 is never loaded. Runs from scratch; nothing is cached.
Writes comp_features.parquet (feature, source, quality tag), comp_completeness.parquet (source coverage per season) and
comp_lineup_change.parquet (how far the lineup versions move the vectors) next to the other result files.
"""
import argparse
import time
from pathlib import Path

import polars as pl

import config
from models import comps as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--log-dir", type=Path, default=config.ROOT)
    ap.add_argument("--no-lineup", action="store_true", help="skip the 4a expectations (adjusted vectors then do not exist)")
    a = ap.parse_args()
    t0 = time.time()
    inp = C.load_inputs(lineup_expectations=not a.no_lineup)
    print(f"inputs loaded in {time.time() - t0:.0f}s", flush=True)
    vec = C.build_vectors(inp)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    paths = C.write_vectors(vec, a.out_dir, a.log_dir)
    print(f"built in {time.time() - t0:.0f}s")
    pl.Config.set_tbl_rows(60)
    print(f"\nteam vectors   {vec.team.height:,} rows\n", vec.team.group_by("version").agg(pl.len()).sort("version"))
    print(f"player vectors {vec.player.height:,} rows\n", vec.player.group_by("unit").agg(pl.len()).sort("unit"))
    print("\nlineup versions: games with 4a expectations", vec.notes["adjusted_games"])
    print("NOT lineup-adjusted (adjusted = healthy):", ", ".join(vec.notes["lineup_not_adjusted"]))
    print("\nparticipation completeness by season (the `estimated` features):")
    print(vec.completeness.filter(pl.col("source") == "participation").pivot(on="measure", index="season", values="pct").sort("season").with_columns(pl.exclude("season").round(1)))
    print("\nwritten:", {k: str(v) for k, v in paths.items()})
    print(f"\nholdout {config.HOLDOUT_SEASON} untouched")
