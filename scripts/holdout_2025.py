"""The one-time 2025 benchmark (docs/overnight_plan.md, "Improvement plan from weeks 1-5", workstream A; P5.4 item 2). Run ONCE; the result is recorded in the plan and not repeated.

Every regular-season week of 2025 is replayed as if unplayed: it sees only games before its first kickoff, the target week and everything after (2026 included) is hidden, models and calibrator are the
2020-2024 ones, and the simulated-spread scaling floor is 2025*100+1. Predictions go to records/holdout_2025/predictions (version "holdout2025").

  python scripts/holdout_2025.py [--push]
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import calibrate as CAL
from live import layers as LY
from live import publish
from live import targets as TG
from live import weekly as W
from models import injuries as ij

SEASON = 2025
WEEKS = list(range(1, 19))
STORE = config.ROOT / "records" / "holdout_2025" / "predictions"
FLOOR_KEY = config.HOLDOUT_SEASON * 100 + 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    sf = TG.schedule_frame(sched).filter(pl.col("season") == SEASON)
    models = W.train_models(LY.read_active())
    cal = CAL.Calibrator.load(W.CALIBRATOR_PATH)
    for wk in WEEKS:
        g = sf.filter(pl.col("week") == wk)
        first_day = g["gameday"].min()
        now = g["kickoff_time"].min() - timedelta(hours=6)
        res = W.run(now, store_dir=STORE, models=models, run_id=f"holdout_w{SEASON}_{wk:02d}", version="holdout2025", work_dir=config.ROOT / "data" / "processed" / "holdout_live", calibrator=cal,
                    replay=dict(hide_outcomes_from=first_day, status_clock=ij.main_run_as_of, floor_key=FLOOR_KEY))
        m = res["manifest"]
        assert (res["season"], res["week"]) == (SEASON, wk), (res["season"], res["week"], wk)
        assert date.fromisoformat(m["history"]["newest_game"]) < first_day, (wk, m["history"]["newest_game"], first_day)       # a week never sees itself or later
        print(f"week {wk}: {m['n_games']} games, written {res['written']}, refused {res['refused']}, newest history game {m['history']['newest_game']}", flush=True)
    if a.push:
        print("published:", publish.publish(config.ROOT, [p for p in STORE.rglob("*") if p.is_file()], "2025 holdout benchmark predictions (version holdout2025)"))


if __name__ == "__main__":
    main()
