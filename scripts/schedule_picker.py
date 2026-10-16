"""Tracked candidate S (yours, 2026-10-16): the locked picker plus schedule inputs (rest days for both teams and the difference, short week, division game,
primetime, grass/turf; ideas batch 1: +54.4% on hidden seasons vs +50.2%, 4 of 6 seasons better, straight bets +3.7% vs -1.6%). Not bet: tracked next to the
locked strategy until live weeks show whether it is really better.

Builds, exactly like the locked replay pool (picker retrained every week on every line before that week, 4+ edge, 3+ games with the team):
  data/processed/replay_pool_schedule.parquet     2024 week 4 - 2026 week 5
  records/cards/<season>_wkNN_card_schedule.csv    the live week (FanDuel markets only)

  python scripts/schedule_picker.py --season 2026 --week 6
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


CR, IB = _load("card_replay"), _load("ideas_batch")
CF, TN = CR.CF, CR.TN
POOL = config.PROCESSED_DIR / "replay_pool_schedule.parquet"
COLS = ["season", "week", "game_id", "player_id", "player_name", "team", "market", "line", "dec_u", "mean", "actual", "picker_under_chance", "won", "profit"]
LABEL = {"rec": "Receptions", "rec_yds": "Receiving yds", "rush_yds": "Rush yds", "pass_yds": "Pass yds"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=6)
    a = ap.parse_args(argv)
    d = TN.build()
    A = TN.Arr(d)
    sched, _ = IB.schedule_feats(d)
    X = np.hstack([TN.feats(d, layered=True), sched])
    d = d.with_columns(history=pl.Series(CR.history_counts(d)))
    out = []
    for s, ws in CR.WEEKS.items():
        for w in ws:
            x = CF.picks_for(d, A, X, s, w, True).filter(pl.col("history") >= CF.MIN_HISTORY)
            out.append(x.select(COLS))
        print(s, "done", flush=True)
    pl.concat(out).write_parquet(POOL)
    x = CF.picks_for(d, A, X, a.season, a.week, True).filter((pl.col("history") >= CF.MIN_HISTORY) & pl.col("market").is_in(list(LABEL)))
    card = x.select("player_name", "team", pl.col("market").replace_strict(LABEL).alias("market"), "line", pl.col("price").cast(pl.Float64),
                    pl.col("mean").round(1).alias("our_prediction"), pl.col("picker_under_chance").round(3), pl.col("edge").round(3), "game_id")
    f = config.ROOT / "records" / "cards" / f"{a.season}_wk{a.week:02d}_card_schedule.csv"
    card.write_csv(f)
    print("wrote", POOL, "and", f, card.height, "picks")


if __name__ == "__main__":
    main()
