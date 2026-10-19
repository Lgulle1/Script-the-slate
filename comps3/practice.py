"""Practice ladder for comps v3 chains (2022-2023), against our projection (relc) and against the line (rline). Logged to records/tuning/comps2_ladder.csv."""
import sys
import time

import numpy as np
import polars as pl

from comps2 import evaluate as V
from comps2 import run_ladder as RL
from comps3 import chains as CH

PRACTICE = (2022, 2023)


def ladder(c, outcome, label):
    y = c.with_columns(relc=pl.col(outcome), expected=pl.col("expected") if outcome == "relc" else pl.col("line"))
    y = y.filter(pl.col("relc").is_not_null())
    lad = V.ladder(y.rename({f"{k}_eff": f"{k}_shift" for k in ("own", "dfn", "team", "int", "sit")}).rename({"chains_mean": "all_shift"}),
                   ("own", "dfn", "team", "int", "sit", "all"), PRACTICE).with_columns(label=pl.lit(label))
    pl.concat([pl.read_csv(RL.LOG), lad], how="diagonal_relaxed").write_csv(RL.LOG)
    return lad


if __name__ == "__main__":
    t0 = time.time()
    x = CH.frame()
    tg = x.filter(pl.col("season").is_in(PRACTICE))["rid"].to_numpy()
    for outcome in sys.argv[1:] or ("relc", "rline"):
        c = CH.run(x, tg, outcome)
        c = CH.with_situation(c, x, tg, outcome)
        c.write_parquet(CH.OUT.with_name(f"comps3_chains_practice_{outcome}.parquet"))
        lad = ladder(c, outcome, f"v3 chains vs {'projection' if outcome == 'relc' else 'line'}")
        print(f"\n=== vs {'projection' if outcome == 'relc' else 'line'}  ({time.time() - t0:.0f}s)")
        print(lad.select("group", "comp", "n", "corr", "corr_min_season", "over_lift"))
