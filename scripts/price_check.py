"""Price check: the model's chance against the price actually quoted, line by line, and (with --grade) what taking the better side would have returned.

  python scripts/price_check.py --store records/retro_2026/predictions --season 2026 --ledger /tmp/ledger.parquet [--grade]
Rules: docs/overnight_plan.md, "Price check". Descriptive only; sets no weight and recommends no bet.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from app import data as D
from eval import calibrate as C
from live import store as ST

LOG = config.ROOT / "records" / "lines" / "prop_lines_log.csv"
THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
DRAWS, SEED = 2000, 7
KEY = ["game_id", "player_id", "market"]


def breakeven(price):
    price = np.asarray(price, float)
    return np.where(price > 0, 100.0 / (price + 100.0), -price / (-price + 100.0))


def quoted(log: pl.DataFrame) -> pl.DataFrame:
    """Latest quoted over and under price per (book, game, player, market, line); rows with both sides and a price."""
    d = log.with_row_index("_i").with_columns(line=pl.col("line").cast(pl.Float64), price=pl.col("price").cast(pl.Float64, strict=False)).sort("captured_at", "_i")
    if "is_alt" in d.columns:                                               # a main line that moved is one bet at its latest line; alternate lines stand side by side
        alt = d.filter(pl.col("is_alt") == "true")
        main = d.filter(pl.col("is_alt") != "true")
        last_line = main.group_by(["book", *KEY, "side"], maintain_order=True).agg(line=pl.col("line").last())
        main = main.join(last_line, on=["book", *KEY, "side", "line"], how="semi")
        d = pl.concat([main, alt]).sort("captured_at", "_i")
    k = ["book", *KEY, "line"]
    o = d.filter(pl.col("side") == "over").group_by(k, maintain_order=True).last().select(*k, over_price="price", captured_at="captured_at")
    u = d.filter(pl.col("side") == "under").group_by(k, maintain_order=True).last().select(*k, under_price="price")
    return o.join(u, on=k, how="inner").drop_nulls(["over_price", "under_price"])


def edges(q: pl.DataFrame, preds: pl.DataFrame, cal: C.Calibrator) -> pl.DataFrame:
    p = D.latest(preds).filter(pl.col("player_id").is_not_null()).select(*KEY, mean="mean")
    x = q.join(p, on=KEY, how="inner").sort("book", *KEY, "line")
    rows = []
    for r in x.iter_rows(named=True):
        s = cal.sample(r["market"], r["mean"])
        if s is None:
            continue
        p_over = float((s > r["line"]).mean())
        be_o, be_u = float(breakeven(r["over_price"])), float(breakeven(r["under_price"]))
        e_o, e_u = p_over - be_o, (1 - p_over) - be_u
        side = "over" if e_o >= e_u else "under"
        rows.append(dict(**r, p_over=p_over, be_over=be_o, be_under=be_u, edge_over=e_o, edge_under=e_u, side=side, edge=max(e_o, e_u),
                         price=r["over_price"] if side == "over" else r["under_price"], model_chance=p_over if side == "over" else 1 - p_over))
    return pl.DataFrame(rows)


def grade(e: pl.DataFrame, season: int) -> pl.DataFrame:
    from live import results as RS
    act, _, _ = RS.played(season)
    g = e.join(act.filter(pl.col("player_id").is_not_null()), on=KEY, how="inner").filter(pl.col("actual") != pl.col("line"))
    won = np.where(g["side"].to_numpy() == "over", g["actual"].to_numpy() > g["line"].to_numpy(), g["actual"].to_numpy() < g["line"].to_numpy())
    dec = np.where(g["price"].to_numpy() > 0, 1 + g["price"].to_numpy() / 100.0, 1 + 100.0 / -g["price"].to_numpy())
    return g.with_columns(won=pl.Series(won.astype(float)), profit=pl.Series(np.where(won, dec - 1, -1.0)), week=pl.col("game_id").str.split("_").list.get(1).cast(pl.Int64))


def boot(v: np.ndarray, cl: np.ndarray):
    u, inv = np.unique(cl, return_inverse=True)
    s, n = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    idx = np.random.default_rng(SEED).integers(0, len(u), size=(DRAWS, len(u)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def summary(g: pl.DataFrame) -> pl.DataFrame:
    g = g.sort("book", *KEY, "line")
    out = []
    for th in THRESHOLDS:
        for name, lo, hi in (("all", 0.0, 1.01), ("chance < 50%", 0.0, 0.5), ("chance 50-70%", 0.5, 0.7), ("chance >= 70%", 0.7, 1.01)):
            t = g.filter((pl.col("edge") >= th) & (pl.col("model_chance") >= lo) & (pl.col("model_chance") < hi))
            if t.height < 10:
                continue
            roi, rlo, rhi = boot(t["profit"].to_numpy(), t["game_id"].to_numpy())
            h1, h2 = t.filter(pl.col("week") <= 3), t.filter(pl.col("week") >= 4)
            out.append(dict(threshold=th, bucket=name, n=t.height, hit=float(t["won"].mean()), mean_break_even=float(breakeven(t["price"].to_numpy()).mean()), roi=roi, lo=rlo, hi=rhi,
                            roi_first=float(h1["profit"].mean()) if h1.height else None, n_first=h1.height, roi_second=float(h2["profit"].mean()) if h2.height else None, n_second=h2.height))
    return pl.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--grade", action="store_true")
    a = ap.parse_args(argv)
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    log = pl.read_csv(LOG, infer_schema_length=0)
    q = quoted(log)
    e = edges(q, ST.PredictionStore(a.store).load(), cal)
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    e.write_parquet(a.ledger)
    print(f"{e.height} priced lines matched to a stored prediction; books {sorted(e['book'].unique().to_list())}")
    if a.grade:
        g = grade(e, a.season)
        s = summary(g)
        s.write_parquet(a.ledger.with_name(a.ledger.stem + "_graded.parquet"))
        with pl.Config(tbl_rows=40, tbl_cols=20, float_precision=3, tbl_width_chars=200):
            print(s)
    return e


if __name__ == "__main__":
    main()
