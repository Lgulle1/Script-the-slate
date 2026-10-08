"""Build and persist the Phase 4b tables (DuckDB + data/processed/*.parquet).

  python build_game_state.py

team_ratings (4b.1), game_expectations and game_state_coefficients (4b.2), team_game_state (4b.2 + 4b.3), walk-forward over
2018-2024: 2018 starts the ratings, 2019 is history (never scored), 2020-2024 are the backtest seasons; 2025 is never loaded.
See features/team_ratings.py and models/game_state.py.
"""
import polars as pl

import config
from models import game_state as gs

if __name__ == "__main__":
    t = gs.build_all()
    gs.persist(t)
    ge = t["game_expectations"].filter(~pl.col("is_history"))
    for name, df in t.items():
        print(f"{name}: {df.height:,} rows")
    print("\n2020-2024 margin / total (walk-forward, no market line):")
    print(ge.group_by("season").agg(pl.len().alias("games"), (pl.col("margin") - pl.col("margin_mean")).abs().mean().round(2).alias("margin_mae"),
                                    (pl.col("total") - pl.col("total_mean")).abs().mean().round(2).alias("total_mae"),
                                    pl.col("margin_sd").mean().round(2), pl.col("total_sd").mean().round(2),
                                    ((pl.col("margin") > 0) == (pl.col("margin_mean") > 0)).mean().round(3).alias("winner_acc")).sort("season"))
    print(f"\nholdout {config.HOLDOUT_SEASON} untouched")
