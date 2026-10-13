"""Build a self-contained HTML snapshot of one graded game (predictions, ranges, floors, results) for viewing as a page. Display only; reads the stored records and display files."""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from app import data as D

GAME = "2026_05_BUF_LA"


def build() -> dict:
    g = pl.read_parquet(config.ROOT / "records" / "graded" / "graded_predictions.parquet").filter(pl.col("game_id") == GAME)
    names = D.player_names([x for x in g["player_id"].unique().to_list() if x])
    ranges = pl.read_csv(config.ROOT / "previews" / "monday_night_BUF_at_LA_chances.csv").unique(subset=["name", "market"]).select("name", "market", "range_10_90")
    floors = pl.read_csv(config.ROOT / "previews" / "monday_night_BUF_at_LA_floors.csv").select("name", "market", "floor_90", "floor_80", "floor_70", "team")
    pl_rows = g.filter(pl.col("player_id").is_not_null()).with_columns(name=pl.col("player_id").replace_strict(names, default=pl.col("player_id")))
    j = pl_rows.join(ranges, on=["name", "market"], how="left").join(floors, on=["name", "market"], how="left")
    rows = []
    for r in j.iter_rows(named=True):
        lo, hi = (float(x) for x in r["range_10_90"].split("-")) if r["range_10_90"] else (None, None)
        rows.append(dict(name=r["name"], team=r["team"], market=r["market"], model=round(r["mean"], 1), lo=lo, hi=hi, f90=r["floor_90"], f80=r["floor_80"], f70=r["floor_70"], actual=r["actual"], dnp=bool(r["did_not_play"])))
    game = {r["market"]: dict(model=r["mean"], actual=r["actual"]) for r in g.filter(pl.col("player_id").is_null()).iter_rows(named=True)}
    return dict(rows=rows, game=game)


if __name__ == "__main__":
    out = build()
    Path(sys.argv[1]).write_text(json.dumps(out))
    print(len(out["rows"]), out["game"])
