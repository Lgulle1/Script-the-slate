"""Display file: for each player and market of a preview, the chance he reaches round-number marks, from how often players with a similar prediction reached them in 2020-2024.

Not a stored record and not a model input. Marks are round numbers chosen here (no sportsbook line is read)."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

STEP = {"pass_att": 3, "pass_cmp": 3, "pass_yds": 25, "qb_rush_att": 1, "qb_rush_yds": 5, "rush_att": 2, "rush_yds": 10, "targets": 1, "rec": 1, "rec_yds": 10}
K = {"qb_rush_att": 250, "qb_rush_yds": 250}


def main(src: Path, out: Path):
    B = pl.read_parquet(config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet").filter(
        (pl.col("method") == "vol_x_eff") & (pl.col("kind") == "player") & (pl.col("season") <= config.HOLDOUT_SEASON - 1)).drop_nulls(["prediction", "actual"])
    ref = {}
    for m in STEP:
        d = B.filter(pl.col("market") == m)
        ref[m] = (d["prediction"].to_numpy(), d["actual"].to_numpy())
    P = pl.read_csv(src)
    rows = []
    for r in P.iter_rows(named=True):
        for m, step in STEP.items():
            p = r.get(m)
            if p is None:
                continue
            x, a = ref[m]
            near = a[np.argsort(np.abs(x - p))[:K.get(m, 600)]]
            lo, hi = np.percentile(near, [10, 90])
            marks = sorted({max(step, round(p * f / step) * step) for f in (0.6, 0.8, 1.0, 1.2, 1.4)})
            for t in marks:
                rows.append(dict(team=r["team"], name=r["name"], pos=r["pos"], market=m, prediction=round(p, 1), range_10_90=f"{lo:.0f}-{hi:.0f}", at_least=t, chance=round(float((near >= t).mean()), 2)))
    pl.DataFrame(rows).write_csv(out)
    return len(rows)


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])))
