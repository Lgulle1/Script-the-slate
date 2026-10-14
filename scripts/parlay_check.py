"""Three-leg parlays of 70% bars at line-centred proxy prices (docs/overnight_plan.md, addendum to "Ladder pricing proxy"). Descriptive.

  python scripts/parlay_check.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ROWS = config.ROOT / "data" / "processed" / "weeks_review_rows.parquet"
OUT = config.ROOT / "data" / "processed" / "weeks_review_parlays.parquet"
N, SEED, LEG = 20000, 7, 70


def main():
    r = pl.read_parquet(ROWS).sort("game_id", "player_id", "market").with_columns(hit=(pl.col("actual") >= pl.col(f"f{LEG}")).cast(pl.Float64))
    hit, b, e, game, week, player = (r[c].to_numpy() for c in ("hit", f"b{LEG}", f"e{LEG}", "game_id", "week", "player_id"))
    rng = np.random.default_rng(SEED)
    out = []
    for kind in ("different games, same week", "same game"):
        trip = []
        while len(trip) < N:
            i = int(rng.integers(len(r)))
            pool = np.flatnonzero((week == week[i]) & ((game != game[i]) if kind.startswith("different") else (game == game[i])) & (player != player[i]))
            if len(pool) < 2:
                continue
            j, k = rng.choice(pool, size=2, replace=False)
            if kind.startswith("different") and (game[j] == game[k]):
                continue
            if kind == "same game" and player[j] == player[k]:
                continue
            trip.append((i, int(j), int(k)))
        t = np.array(trip)
        h = hit[t].prod(1)
        row = dict(kind=kind, n=len(t), hit=float(h.mean()), product_of_model_chances=float(e[t].prod(1).mean()), product_of_line_chances=float(b[t].prod(1).mean()))
        for mg in (0.0, 0.10):
            price = (b[t] * (1 + mg)).prod(1)
            row[f"roi_{int(mg * 100)}"] = float((h / price - 1).mean())
        out.append(row)
    df = pl.DataFrame(out)
    df.write_parquet(OUT)
    with pl.Config(tbl_cols=20, float_precision=3, tbl_width_chars=200):
        print(df)
    return df


if __name__ == "__main__":
    main()
