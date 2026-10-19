"""Run the signal ladder for a list of parameter sets on the practice seasons; append every result to records/tuning/comps2_ladder.csv (every structure
tried is logged, kept or not)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

import config
from comps2 import evaluate as V
from comps2 import searches as S

LOG = config.ROOT / "records" / "tuning" / "comps2_ladder.csv"
PRACTICE = (2021, 2022, 2023)


def run(label: str, P: S.Params, searches=("own", "dfn", "int"), e=None):
    e = e if e is not None else S.load()
    tg = e.filter(pl.col("season").is_in(PRACTICE))["rid"].to_numpy()
    t0 = time.time()
    c = S.run(e, P, tg, searches)
    lad = V.ladder(c, searches, PRACTICE).with_columns(label=pl.lit(label), params=pl.lit(json.dumps(P.__dict__)), secs=pl.lit(time.time() - t0))
    old = pl.read_csv(LOG) if LOG.exists() else None
    (pl.concat([old, lad], how="diagonal_relaxed") if old is not None else lad).write_csv(LOG)
    return c, lad


if __name__ == "__main__":
    e = S.load()
    grid = {
        "control (no similarity)": S.Params(h_def=None, h_off=None, h_arch=None, h_pair=None),
        "h=0.5": S.Params(h_def=0.5, h_off=0.5, h_arch=0.5, h_pair=0.5),
        "h=0.75": S.Params(h_def=0.75, h_off=0.75, h_arch=0.75, h_pair=0.75),
        "h=1.0": S.Params(),
        "h=1.5": S.Params(h_def=1.5, h_off=1.5, h_arch=1.5, h_pair=1.5),
    }
    only = sys.argv[1:]
    for lab, P in grid.items():
        if only and lab not in only:
            continue
        _, lad = run(lab, P, e=e)
        print(f"\n=== {lab}")
        print(lad.select("group", "comp", "n", "corr", "corr_min_season", "slope", "mae_gain"))
