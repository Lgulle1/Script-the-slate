"""Phase 5 weekly live run (docs/overnight_plan.md "Phase 5 gates").

  python run_weekly.py [--now ISO8601] [--bootstrap | --refresh] [--push] [--store DIR]

  --bootstrap   pull 2020 to the current season into a fresh raw database (a container without one)
  --refresh     append the current season's new rows (schedules, player stats, depth charts, play-by-play)
  --push        publish this run's records (predictions, manifest, refusals, lines) to the data-snapshots branch
Predictions are stored before kickoff only; a game that has started is refused and logged. The clock defaults to now (UTC).
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import config
from ingest import refresh_live
from live import publish
from live import weekly


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", default=None)
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-sync", action="store_true", help="do not bring the published records of data-snapshots into this checkout first (only for tests / a clean first run)")
    ap.add_argument("--store", type=Path, default=config.ROOT / "records" / "predictions")
    ap.add_argument("--lines", type=Path, default=config.ROOT / "records" / "lines" / "lines_log.csv")
    a = ap.parse_args(argv)
    fixed = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else None
    if fixed is not None and fixed.tzinfo is None:
        raise SystemExit("--now needs a timezone (e.g. 2026-10-16T20:00:00Z)")
    if a.bootstrap or not config.RAW_DUCKDB_PATH.exists():
        print("bootstrap:", refresh_live.bootstrap())
    elif a.refresh:
        print("refresh:", refresh_live.refresh())
    now = fixed or datetime.now(timezone.utc)          # the clock is read AFTER the pull: rows are stamped pulled_at = the pull time, and a run only sees rows pulled by its clock
    if not a.no_sync:           # continue the published version numbers; a failed sync stops the run rather than restarting at version 1
        print("synced records:", publish.sync_records(config.ROOT, ["records/predictions", "records/lines"]))
    before = set(a.store.rglob("*")) if a.store.exists() else set()
    out = weekly.run(now, store_dir=a.store, lines_path=a.lines)
    print({k: v for k, v in out.items() if k not in ("manifest", "predictions", "debug", "explanations", "margin_terms")})
    if a.push and out.get("status") == "ok":
        new = sorted(p for p in set(a.store.rglob("*")) - before if p.is_file()) + [p for p in (a.store / "refusals.csv", a.lines) if p.exists()]
        sha = publish.publish(config.ROOT, new, f"predictions {out['manifest']['run_id']}: {out['written']} written, {out['refused']} refused, {out['unchanged']} unchanged")
        print("published:", sha)
    return out


if __name__ == "__main__":
    main()
