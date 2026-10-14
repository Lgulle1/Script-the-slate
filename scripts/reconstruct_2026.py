"""Run the live path for 2026 weeks 1-5 as if each week had not been played (docs/overnight_plan.md, "2026 weeks 1-5 reconstruction"). Code and models frozen; nothing tuned.

  python scripts/reconstruct_2026.py                 # writes records/retro_2026/predictions (version "retro")
  python scripts/reconstruct_2026.py --push          # also publishes the store to data-snapshots (the outside timestamp)
"""
import argparse
import sys
from datetime import timedelta
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

SEASON = 2026
WEEKS = [1, 2, 3, 4, 5]
STORE = config.ROOT / "records" / "retro_2026" / "predictions"
HOLDOUT_FLOOR_KEY = config.HOLDOUT_SEASON * 100 + 1        # scaling terms fit on 2020-2024 games only, as in a live run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--weeks", type=int, nargs="*", default=WEEKS)
    a = ap.parse_args(argv)
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    sf = TG.schedule_frame(sched).filter(pl.col("season") == SEASON)
    active = LY.read_active()
    models = W.train_models(active)
    cal = CAL.Calibrator.load(W.CALIBRATOR_PATH)
    out = {}
    for wk in a.weeks:
        g = sf.filter(pl.col("week") == wk)
        first = g["kickoff_time"].min()
        now = first - timedelta(hours=6)
        res = W.run(now, store_dir=STORE, models=models, run_id=f"retro_w{SEASON}_{wk:02d}", version="retro", work_dir=config.ROOT / "data" / "processed" / "retro_live", calibrator=cal,
                    replay=dict(hide_outcomes_from=g["gameday"].min(), status_clock=ij.main_run_as_of, floor_key=HOLDOUT_FLOOR_KEY))
        assert (res["season"], res["week"]) == (SEASON, wk), (res["season"], res["week"], wk)
        m = res["manifest"]
        print(f"week {wk}: {m['n_games']} games, written {res['written']}, refused {res['refused']}, newest history game {m['history']['newest_game']}, stacks {m['stacks']}")
        out[wk] = res
    if a.push:
        files = [p for p in STORE.rglob("*") if p.is_file()]
        print("published:", publish.publish(config.ROOT, files, f"2026 weeks {a.weeks[0]}-{a.weeks[-1]} reconstruction (version retro), before any line is imported"))
    return out


if __name__ == "__main__":
    main()
