"""Display file: for each prediction, the highest floor the real number reached or beat in 90%, 80% and 70% of games with a similar prediction (2020-2024).

Not a stored record and not a model input; no sportsbook line is read."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

MARKETS = ["pass_att", "pass_cmp", "pass_yds", "qb_rush_att", "qb_rush_yds", "rush_att", "rush_yds", "targets", "rec", "rec_yds"]
LEVELS = (0.9, 0.8, 0.7)
K = {"qb_rush_att": 250, "qb_rush_yds": 250}


def floor_of(near: np.ndarray, level: float) -> float:
    """Largest value x with at least `level` of `near` at or above x (integers: the stats are whole numbers)."""
    return float(np.floor(np.quantile(near, 1 - level, method="lower")))


def reference(through_season: int) -> dict:
    B = pl.read_parquet(config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet").filter(
        (pl.col("method") == "vol_x_eff") & (pl.col("kind") == "player") & (pl.col("season") <= through_season)).drop_nulls(["prediction", "actual"])
    return {m: (B.filter(pl.col("market") == m)["prediction"].to_numpy(), B.filter(pl.col("market") == m)["actual"].to_numpy()) for m in MARKETS}


def floors(ref: dict, m: str, p: float) -> dict:
    x, a = ref[m]
    near = a[np.argsort(np.abs(x - p))[:K.get(m, 600)]]
    return {f"floor_{int(l * 100)}": floor_of(near, l) for l in LEVELS}


def main(src: Path, out: Path) -> int:
    ref = reference(config.HOLDOUT_SEASON - 1)
    rows = []
    for r in pl.read_csv(src).iter_rows(named=True):
        for m in MARKETS:
            p = r.get(m)
            if p is not None:
                rows.append(dict(team=r["team"], name=r["name"], market=m, prediction=round(p, 1), **floors(ref, m, p)))
    pl.DataFrame(rows).write_csv(out)
    return len(rows)


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])))
