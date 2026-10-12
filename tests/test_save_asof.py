"""As-of snapshots of the 2026 tables that cannot be pulled again as they stood (ingest/save_asof.py)."""
from datetime import datetime

import polars as pl
import pytest

import config
from ingest import save_asof as SA


def _frames():
    return {
        "injuries": pl.DataFrame({"season": [2026, 2026, 2025], "week": [1, 2, 18], "gsis_id": ["a", "b", "c"]}),
        "rosters_weekly": pl.DataFrame({"season": [2026, 2025], "week": [1, 1], "gsis_id": ["a", "z"]}),
        "schedules": pl.DataFrame({"season": [2026, 2026, 2024], "game_id": ["g1", "g2", "g0"]}),
        "depth_charts": pl.DataFrame({"season": [None, None, None], "dt": ["2026-10-07T06:02:29Z", "2025-12-30T06:00:00Z", "2026-01-02T00:00:00Z"],
                                      "gsis_id": ["a", "b", "c"]}, schema_overrides={"season": pl.Int32}),
    }


def test_only_live_season_rows_are_kept_and_the_holdout_never_is():
    out = SA.live_rows(_frames())
    assert out["injuries"]["gsis_id"].to_list() == ["a", "b"] and out["rosters_weekly"]["gsis_id"].to_list() == ["a"] and out["schedules"]["game_id"].to_list() == ["g1", "g2"]
    assert sorted(out["depth_charts"]["gsis_id"].to_list()) == ["a", "c"]                    # the 2025-dated depth chart row (holdout) is dropped
    for t, f in out.items():
        assert (f["season"].drop_nulls() > config.HOLDOUT_SEASON).all() if "season" in f.columns and t != "depth_charts" else True


def test_a_snapshot_is_written_once_and_never_overwritten(tmp_path):
    now = datetime(2026, 10, 14, 12, 5)
    d = SA.save(_frames(), tmp_path, now)
    assert sorted(p.name for p in d.iterdir()) == ["depth_charts.parquet", "injuries.parquet", "rosters_weekly.parquet", "schedules.parquet"]
    assert d.name == "2026-10-14T12" and pl.read_parquet(d / "injuries.parquet").height == 2
    first = (d / "injuries.parquet").read_bytes()
    SA.save({**_frames(), "injuries": pl.DataFrame({"season": [2026], "week": [9], "gsis_id": ["q"]})}, tmp_path, now)       # same hour again
    assert (d / "injuries.parquet").read_bytes() == first                                    # the earlier snapshot is untouched


def test_the_weekly_slots():
    assert SA.is_asof_slot(datetime(2026, 10, 14, 12, 3))          # Wednesday 12:00 UTC
    assert SA.is_asof_slot(datetime(2026, 10, 17, 18, 40))         # Saturday 18:00 UTC
    assert not SA.is_asof_slot(datetime(2026, 10, 14, 18, 0)) and not SA.is_asof_slot(datetime(2026, 10, 18, 12, 0))


def test_the_scheduled_run_records_forecasts_always_and_the_asof_tables_only_in_a_weekly_slot(tmp_path):
    from ingest import scheduled_snapshot as SS
    calls = []
    kw = dict(forecasts=lambda now: calls.append("forecasts") or 3, asof=lambda now: calls.append("asof") or tmp_path)
    SS.run(datetime(2026, 10, 14, 6, 7), **kw)                      # Wednesday 06:07: forecasts only
    SS.run(datetime(2026, 10, 14, 12, 7), **kw)                     # Wednesday 12:07: both
    assert calls == ["forecasts", "forecasts", "asof"]
    assert SS.run(datetime(2026, 10, 14, 6, 7), **kw)["asof"] is None
