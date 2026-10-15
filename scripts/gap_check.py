"""Gap check: was the model's gap against the quoted break-even right? And how did the book's confident prices do? (docs/overnight_plan.md, "Gap check"). Descriptive.

  python scripts/gap_check.py --store records/retro_2026/predictions --season 2026 --tag weeks1_5
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from app import data as D
from eval import calibrate as C
from live import store as ST

spec = importlib.util.spec_from_file_location("price_check", Path(__file__).resolve().parent / "price_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)
GAP_BINS = [(-1e9, -15, "below -15"), (-15, -10, "-15 to -10"), (-10, -5, "-10 to -5"), (-5, 0, "-5 to 0"), (0, 5, "0 to +5"), (5, 10, "+5 to +10"), (10, 15, "+10 to +15"), (15, 1e9, "above +15")]
IMP_BINS = [(0, 0.40, "below 40%"), (0.40, 0.45, "40-45%"), (0.45, 0.50, "45-50%"), (0.50, 0.55, "50-55%"), (0.55, 0.60, "55-60%"), (0.60, 1.01, "60% and above")]


def selections(q: pl.DataFrame, preds: pl.DataFrame, cal: C.Calibrator) -> pl.DataFrame:
    e = pc.edges(q, preds, cal)
    over = e.select(*pc.KEY, "book", "line", side=pl.lit("over"), price="over_price", model_chance="p_over", break_even="be_over")
    under = e.select(*pc.KEY, "book", "line", side=pl.lit("under"), price="under_price", model_chance=(1 - pl.col("p_over")), break_even="be_under")
    return pl.concat([over, under]).with_columns(gap=(pl.col("model_chance") - pl.col("break_even")) * 100, edge=pl.col("model_chance") - pl.col("break_even"))


def slope(gap: np.ndarray, realised: np.ndarray, cl: np.ndarray) -> tuple:
    """OLS slope of the realised gap (hit - break-even, points) on the stated gap (points); game-cluster bootstrap interval."""
    def fit(g, r):
        gc = g - g.mean()
        return float((gc * (r - r.mean())).sum() / (gc ** 2).sum())
    u = np.unique(cl)
    by = {k: np.flatnonzero(cl == k) for k in u}
    rng = np.random.default_rng(pc.SEED)
    b = []
    for _ in range(pc.DRAWS):
        idx = np.concatenate([by[k] for k in rng.choice(u, size=len(u), replace=True)])
        b.append(fit(gap[idx], realised[idx]))
    return fit(gap, realised), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def bucket_table(g: pl.DataFrame, col: str, bins: list, scale: float = 1.0) -> pl.DataFrame:
    rows = []
    for lo, hi, name in bins:
        t = g.filter((pl.col(col) * scale >= lo) & (pl.col(col) * scale < hi)) if col == "gap" else g.filter((pl.col(col) >= lo) & (pl.col(col) < hi))
        if t.height < 10:
            rows.append(dict(bucket=name, n=t.height))
            continue
        roi, rlo, rhi = pc.boot(t["profit"].to_numpy(), t["game_id"].to_numpy())
        rows.append(dict(bucket=name, n=t.height, mean_chance=float(np.mean(t["model_chance"].to_numpy())), mean_break_even=float(np.mean(t["break_even"].to_numpy())), hit=float(np.mean(t["won"].to_numpy())),
                         realised_gap=float(100 * (np.mean(t["won"].to_numpy()) - np.mean(t["break_even"].to_numpy()))), stated_gap=float(np.mean(t["gap"].to_numpy())), roi=roi, lo=rlo, hi=rhi))
    return pl.DataFrame(rows, infer_schema_length=None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    q = pc.quoted(pl.read_csv(pc.LOG, infer_schema_length=0))
    s = selections(q, ST.PredictionStore(a.store).load(), cal)
    g = pc.grade(s.with_columns(edge=pl.col("edge")), a.season).sort("book", *pc.KEY, "line", "side")
    g = g.with_columns(implied=pl.col("break_even"))
    A = bucket_table(g, "gap", GAP_BINS)
    B = bucket_table(g.with_columns(model_chance=pl.col("model_chance")), "implied", IMP_BINS)
    sl = slope(g["gap"].to_numpy(), 100 * (g["won"].to_numpy() - g["break_even"].to_numpy()), g["game_id"].to_numpy())
    names = D.player_names(g["player_id"].unique().to_list())
    ex = g.with_columns(player=pl.col("player_id").replace(names))
    pos = ex.filter(pl.col("gap") > 0)
    top = pos.sort("gap", descending=True, maintain_order=True).head(25)
    near = pos.sort("gap", maintain_order=True).head(25)
    cols = ["player", "game_id", "market", "side", "line", "price", "model_chance", "break_even", "gap", "actual", "won", "profit"]
    out = Path(config.PROCESSED_DIR)
    A.write_parquet(out / f"gap_check_{a.tag}_gap.parquet")
    B.write_parquet(out / f"gap_check_{a.tag}_confidence.parquet")
    top.select(cols).write_parquet(out / f"gap_check_{a.tag}_top.parquet")
    near.select(cols).write_parquet(out / f"gap_check_{a.tag}_near.parquet")
    pl.DataFrame([dict(selections=g.height, slope=sl[0], slope_lo=sl[1], slope_hi=sl[2], corr=float(np.corrcoef(g["gap"].to_numpy(), g["won"].to_numpy() - g["break_even"].to_numpy())[0, 1]))]).write_parquet(out / f"gap_check_{a.tag}_slope.parquet")
    with pl.Config(tbl_rows=40, tbl_cols=20, float_precision=3, tbl_width_chars=210):
        print("A. by gap"); print(A)
        print(f"slope of realised gap on stated gap: {sl[0]:.3f} [{sl[1]:.3f}, {sl[2]:.3f}] on {g.height} selections")
        print("B. by the book's implied chance"); print(B)
        print("C. positive gaps, largest"); print(top.select(cols).head(12))
        print("C. positive gaps, closest to zero"); print(near.select(cols).head(12))
    return g


if __name__ == "__main__":
    main()
