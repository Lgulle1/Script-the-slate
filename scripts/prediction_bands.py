"""Display file: each prediction as 'number +/- band -> share of the time the real number landed inside', measured on players with a similar prediction, 2020-2024.

Not a stored record and not a model input; the bands are chosen here per market (no sportsbook line is read)."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

BANDS = {"pass_att": (3, 6), "pass_cmp": (2, 4), "pass_yds": (25, 50), "qb_rush_att": (1, 2), "qb_rush_yds": (10, 20), "rush_att": (2, 4), "rush_yds": (15, 30),
         "targets": (1, 2), "rec": (1, 2), "rec_yds": (15, 30)}
K = {"qb_rush_att": 250, "qb_rush_yds": 250}


def main(src: Path, out: Path) -> int:
    B = pl.read_parquet(config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet").filter(
        (pl.col("method") == "vol_x_eff") & (pl.col("kind") == "player") & (pl.col("season") <= config.HOLDOUT_SEASON - 1)).drop_nulls(["prediction", "actual"])
    ref = {m: (B.filter(pl.col("market") == m)["prediction"].to_numpy(), B.filter(pl.col("market") == m)["actual"].to_numpy()) for m in BANDS}
    rows = []
    for r in pl.read_csv(src).iter_rows(named=True):
        for m, (tight, wide) in BANDS.items():
            p = r.get(m)
            if p is None:
                continue
            x, a = ref[m]
            near = np.argsort(np.abs(x - p))[:K.get(m, 600)]
            miss = np.abs(a[near] - p)
            rows.append(dict(team=r["team"], name=r["name"], market=m, prediction=round(p, 1), tight=tight, tight_share=round(float((miss <= tight).mean()), 2),
                             wide=wide, wide_share=round(float((miss <= wide).mean()), 2)))
    pl.DataFrame(rows).write_csv(out)
    return len(rows)


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])))
