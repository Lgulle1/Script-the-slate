"""Phase 5.2: grade the stored predictions whose games are final (plan: "Prediction schedule, calibration and grading").

  python grade_week.py              # grade everything gradable, append to records/graded/graded_predictions.parquet, print the summary per stack
  python grade_week.py --bootstrap --refresh --push   # a fresh container: pull the results first, grade, publish
  python grade_week.py --push       # also publish the graded table and the summary to the data-snapshots branch

Idempotent (a prediction_id is graded once), pure track (no sportsbook line), did-not-play rows kept apart, stacks never blended.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

import config
from ingest import refresh_live
from live import grade as GR
from live import publish
from live import results as RS
from live import store as ST


def main(argv=None) -> pl.DataFrame:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=config.ROOT / "records" / "predictions")
    ap.add_argument("--graded", type=Path, default=config.ROOT / "records" / "graded" / "graded_predictions.parquet")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--now")
    ap.add_argument("--bootstrap", action="store_true", help="pull 2020 to now into a fresh raw database (a container without one)")
    ap.add_argument("--refresh", action="store_true", help="append the new rows (results of the games just played) before grading")
    a = ap.parse_args(argv)
    if not a.no_sync:
        publish.sync_records(config.ROOT, ["records/predictions", "records/graded"])
    if a.bootstrap or not config.RAW_DUCKDB_PATH.exists():
        print("bootstrap:", refresh_live.bootstrap())
    elif a.refresh:
        print("refresh:", refresh_live.refresh())
    now = datetime.fromisoformat(a.now).astimezone(timezone.utc) if a.now else datetime.now(timezone.utc)
    gradable = ST.PredictionStore(a.store).gradable()
    seasons = sorted({int(g[:4]) for g in gradable["game_id"].unique().to_list()})
    graded = None
    for season in seasons:
        actuals, plog, _ = RS.played(season)
        graded = GR.grade(gradable.filter(pl.col("game_id").str.starts_with(str(season))), actuals, a.graded, graded_at=now)
    if graded is None:
        print("nothing gradable yet")
        return pl.DataFrame()
    summ = GR.summary(graded)
    cmp = RS.baseline_mae(graded, plog)
    summ = summ.join(cmp, on=["stack", "market"], how="left")
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)
    print(summ)
    out = a.graded.with_name("graded_summary.csv")
    summ.write_csv(out)
    if a.push:
        print("published:", publish.publish(config.ROOT, [a.graded, out], f"graded {now.strftime('%Y-%m-%d')}: {graded.height} predictions graded in total"))
    return summ


if __name__ == "__main__":
    main()
