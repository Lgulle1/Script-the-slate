"""Build and persist the four tables Phase 4 reads (DuckDB + data/processed/*.parquet).

  python build_phase4_tables.py [--seasons 2020 2021 ...]

Walk-forward over 2020-2024 with Phase 3 models only (no comparables, injuries or simulation); 2025 is never loaded.
See features/phase4_inputs.py for each table's columns and timing contract.
"""
import argparse

import config
from features import phase4_inputs as p4

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=list(config.BACKTEST_SEASONS))
    a = ap.parse_args()
    tables = p4.build_all(a.seasons)
    p4.persist(tables)
    for name, t in tables.items():
        print(f"{name}: {t.height:,} rows  {t.group_by('season').len().sort('season').rows()}")
