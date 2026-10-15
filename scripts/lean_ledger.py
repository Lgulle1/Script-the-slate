"""Lean ledger: take the model's pick (over if its number is above the line, else under) on every FanDuel main line, flat 1 unit at the quoted price. docs/overnight_plan.md, "Lean ledger".

  python scripts/lean_ledger.py --store records/retro_2026/predictions --season 2026 --tag weeks1_5
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
from live import store as ST

spec = importlib.util.spec_from_file_location("price_check", Path(__file__).resolve().parent / "price_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)
MARKETS = ["pass_yds", "rush_yds", "rec", "rec_yds", "qb_rush_yds"]
LABEL = {"pass_yds": "Pass yards", "rush_yds": "Rush yards", "rec": "Receptions", "rec_yds": "Receiving yards", "qb_rush_yds": "QB rush yards", "ALL": "ALL MARKETS"}


def build(season: int, store: Path) -> pl.DataFrame:
    q = pc.quoted(pl.read_csv(pc.LOG, infer_schema_length=0)).filter(pl.col("market").is_in(MARKETS))
    p = D.latest(ST.PredictionStore(store).load()).filter(pl.col("player_id").is_not_null()).select(*pc.KEY, model="mean")
    x = q.join(p, on=pc.KEY, how="inner").filter(pl.col("model") != pl.col("line")).sort("book", *pc.KEY, "line")
    out = []
    for side in ("model_pick", "over", "under"):
        s = x.with_columns(side=pl.when(pl.col("model") > pl.col("line")).then(pl.lit("over")).otherwise(pl.lit("under")) if side == "model_pick" else pl.lit(side))
        s = s.with_columns(price=pl.when(pl.col("side") == "over").then(pl.col("over_price")).otherwise(pl.col("under_price")), edge=pl.lit(0.0), model_chance=pl.lit(0.5), strategy=pl.lit(side))
        out.append(pc.grade(s.select(*pc.KEY, "line", "side", "price", "edge", "model_chance", "strategy"), season))
    return pl.concat(out)


def tally(g: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for m in [*MARKETS, "ALL"]:
        t = g if m == "ALL" else g.filter(pl.col("market") == m)
        row = dict(market=LABEL[m])
        for strat, nm in (("model_pick", "pick"), ("over", "allover"), ("under", "allunder")):
            u = t.filter(pl.col("strategy") == strat)
            row[f"{nm}_bets"] = u.height
            row[f"{nm}_wins"] = int(u["won"].sum()) if u.height else 0
            row[f"{nm}_hit"] = float(np.mean(u["won"].to_numpy())) if u.height else None
            row[f"{nm}_units"] = float(np.sum(u["profit"].to_numpy())) if u.height else 0.0
            row[f"{nm}_roi"] = float(np.mean(u["profit"].to_numpy())) if u.height else None
        rows.append(row)
    return pl.DataFrame(rows)


def running(g: pl.DataFrame) -> pl.DataFrame:
    u = g.filter(pl.col("strategy") == "model_pick").sort("week", *pc.KEY)
    wk = [dict(week=int(w), bets=t.height, wins=int(t["won"].sum()), units=float(np.sum(t["profit"].to_numpy()))) for (w,), t in sorted(u.partition_by("week", as_dict=True).items())]
    d = pl.DataFrame(wk)
    return d.with_columns(running_units=pl.col("units").cum_sum(), running_bets=pl.col("bets").cum_sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    g = build(a.season, a.store).sort("strategy", "week", *pc.KEY, "line")
    t, r = tally(g), running(g)
    t.write_parquet(config.PROCESSED_DIR / f"lean_ledger_{a.tag}_tally.parquet")
    r.write_parquet(config.PROCESSED_DIR / f"lean_ledger_{a.tag}_running.parquet")
    with pl.Config(tbl_rows=40, tbl_cols=30, float_precision=3, tbl_width_chars=220):
        print(t.select("market", "pick_bets", "pick_wins", "pick_hit", "pick_units", "pick_roi", "allover_hit", "allover_units", "allunder_hit", "allunder_units"))
        print(r)
    return g


if __name__ == "__main__":
    main()
