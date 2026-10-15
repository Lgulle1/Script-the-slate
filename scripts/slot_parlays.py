"""Slot parlays: one 3-leg parlay per slot of each week (Thursday night, Sunday 1 pm, Sunday 4 pm, Sunday night, Monday night), with and without a 30% profit boost. docs/overnight_plan.md, "Slot parlays".

  python scripts/slot_parlays.py --store records/retro_2026/predictions --season 2026 --tag weeks1_5
A: the model's three most confident picks (side with the higher model chance) on the real FanDuel main lines, real prices.
B: the model's 80% bars (over) of the players who have a main line, proxy price from the line-centred distribution at a margin of 0 and 10%.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from app import data as D
from eval import calibrate as C
from live import results as RS
from live import store as ST
from live import targets as TG

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("price_check", HERE / "price_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)
spec2 = importlib.util.spec_from_file_location("gap_check", HERE / "gap_check.py")
gc = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(gc)
MARKETS = ["pass_yds", "rush_yds", "rec_yds", "rec"]
SLOTS = ["Thursday night", "Sunday 1 pm", "Sunday 4 pm", "Sunday night", "Monday night"]
BOOST = 0.30
MIN_BAR = {"pass_yds": 10, "rush_yds": 10, "rec_yds": 10, "rec": 1}
LEGS = 3


def slot_of(kick_utc) -> str:
    et = kick_utc.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
    if et.weekday() == 0:
        return "Monday night"
    if et.weekday() == 6:
        return "Sunday 1 pm" if et.hour < 15 else "Sunday 4 pm" if et.hour < 19 else "Sunday night"
    return "Thursday night"                                                  # the Wednesday or Thursday opener and any other non-Sunday, non-Monday game


def slots(season: int) -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    sf = TG.schedule_frame(sched).filter(pl.col("season") == season)
    return pl.DataFrame([dict(game_id=r["game_id"], week=r["week"], slot=slot_of(r["kickoff_time"])) for r in sf.iter_rows(named=True)])


def decimal(price: float) -> float:
    return 1 + price / 100.0 if price > 0 else 1 + 100.0 / -price


def settle(legs: pl.DataFrame, dec_col: str) -> dict:
    """legs: won (1/0/None = void), decimal odds. Void legs drop out. Returns hit, odds, units without and with the 30% boost; None when every leg is void."""
    live = legs.filter(pl.col("won").is_not_null())
    if live.is_empty():
        return dict(hit=None, odds=None, units=None, units_boost=None)
    odds = float(np.prod(live[dec_col].to_numpy()))
    hit = bool(np.all(live["won"].to_numpy() == 1.0))
    return dict(hit=hit, odds=odds, units=(odds - 1) if hit else -1.0, units_boost=(1 + BOOST) * (odds - 1) if hit else -1.0)


def pick(c: pl.DataFrame, rank_cols: list) -> pl.DataFrame:
    """Top three legs of distinct players by the pre-game ranking (descending), deterministic ties."""
    c = c.sort([*rank_cols, "player_id"], descending=[True] * len(rank_cols) + [False])
    return c.unique(subset=["player_id"], keep="first", maintain_order=True).head(LEGS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    preds = ST.PredictionStore(a.store).load()
    q = pc.quoted(pl.read_csv(pc.LOG, infer_schema_length=0)).filter(pl.col("market").is_in(MARKETS))
    sl = slots(a.season)
    act, _, _ = RS.played(a.season)
    act = act.filter(pl.col("player_id").is_not_null()).select(*pc.KEY, "actual")
    names = D.player_names(q["player_id"].unique().to_list())
    # ---- A: real prices, the most confident side of every line
    s = gc.selections(q, preds, cal).join(sl, on="game_id", how="inner").join(act, on=pc.KEY, how="left")
    s = s.with_columns(dec=pl.col("price").map_elements(decimal, return_dtype=pl.Float64), won=pl.when(pl.col("actual").is_null() | (pl.col("actual") == pl.col("line"))).then(None).when(pl.col("side") == "over").then((pl.col("actual") > pl.col("line")).cast(pl.Float64)).otherwise((pl.col("actual") < pl.col("line")).cast(pl.Float64)))
    # ---- B: the model's 80% bar, proxy price
    p = D.latest(preds).filter(pl.col("player_id").is_not_null()).select(*pc.KEY, mean="mean")
    lines = q.join(p, on=pc.KEY, how="inner").join(sl, on="game_id", how="inner").join(act, on=pc.KEY, how="left").sort("game_id", "player_id", "market")
    bl = []
    for r in lines.iter_rows(named=True):
        smp = cal.sample(r["market"], r["mean"])
        sml = cal.sample(r["market"], r["line"])
        if smp is None or sml is None:
            continue
        bar = float(np.floor(np.quantile(smp, 0.2, method="lower")))
        if bar < MIN_BAR[r["market"]]:
            continue
        e, b = float((smp >= bar).mean()), float(min(0.995, max(0.005, (sml >= bar).mean())))
        won = None if r["actual"] is None else float(r["actual"] >= bar)
        bl.append(dict(game_id=r["game_id"], player_id=r["player_id"], market=r["market"], week=r["week"], slot=r["slot"], bar=bar, e=e, b=b, won=won, actual=r["actual"]))
    B = pl.DataFrame(bl)
    rows = []
    for wk in sorted(sl["week"].unique().to_list()):
        for slot in SLOTS:
            ca = s.filter((pl.col("week") == wk) & (pl.col("slot") == slot))
            ta = pick(ca, ["model_chance"]) if ca.height else ca
            if ta.height == LEGS:
                st = settle(ta, "dec")
                rows.append(dict(version="A real prices", week=wk, slot=slot, margin=None, legs=" | ".join(f"{names.get(r['player_id'], r['player_id'])} {r['market']} {r['side']} {r['line']} ({r['price']:+.0f})" for r in ta.iter_rows(named=True)),
                                 leg_results="".join("v" if r["won"] is None else "W" if r["won"] == 1.0 else "L" for r in ta.iter_rows(named=True)), **st))
            cb = B.filter((pl.col("week") == wk) & (pl.col("slot") == slot)) if B.height else B
            tb = pick(cb, ["e", "b"]) if cb.height else cb
            if tb.height == LEGS:
                for mg in (0.0, 0.10):
                    t2 = tb.with_columns(dec=1.0 / pl.min_horizontal(pl.lit(0.995), pl.col("b") * (1 + mg)))                  # the implied chance of a leg cannot pass 99.5%
                    st = settle(t2, "dec")
                    rows.append(dict(version="B ladder, 80% bars", week=wk, slot=slot, margin=mg, legs=" | ".join(f"{names.get(r['player_id'], r['player_id'])} {r['market']} {r['bar']:.0f}+" for r in tb.iter_rows(named=True)),
                                     leg_results="".join("v" if r["won"] is None else "W" if r["won"] == 1.0 else "L" for r in tb.iter_rows(named=True)), **st))
    out = pl.DataFrame(rows, infer_schema_length=None)
    out.write_parquet(config.PROCESSED_DIR / f"slot_parlays_{a.tag}.parquet")
    tot = []
    for v in out["version"].unique(maintain_order=True).to_list():
        for mg in sorted(set(out.filter(pl.col("version") == v)["margin"].to_list()), key=lambda x: -1 if x is None else x):
            t = out.filter((pl.col("version") == v) & (pl.col("margin") == mg if mg is not None else pl.col("margin").is_null()) & pl.col("hit").is_not_null())
            tot.append(dict(version=v, margin=mg, parlays=t.height, hit=int(t["hit"].sum()), units=float(t["units"].sum()), units_boost=float(t["units_boost"].sum())))
    tot = pl.DataFrame(tot, infer_schema_length=None)
    tot.write_parquet(config.PROCESSED_DIR / f"slot_parlays_{a.tag}_totals.parquet")
    with pl.Config(tbl_rows=120, tbl_cols=20, float_precision=2, tbl_width_chars=240, fmt_str_lengths=120):
        print(tot)
        print(out.filter((pl.col("margin").is_null()) | (pl.col("margin") == 0.10)).select("version", "week", "slot", "leg_results", "odds", "hit", "units", "units_boost"))
    return out


if __name__ == "__main__":
    main()
