"""The 2025+ depth-chart layout (a snapshot timestamp `dt`, no season / week) read back into the shape the Phase 3-4 loaders use (features/depth_live.py)."""
import polars as pl
import pytest

import config
from features import depth_live as DL


def _row(dt, team, abb, slot, rank, gid, grp="3WR 1TE"):
    return dict(dt=dt, team=team, pos_abb=abb, pos_slot=slot, pos_rank=rank, gsis_id=gid, pos_grp=grp, season=None, week=None, club_code=None, position=None, depth_team=None, formation=None)


def _frame():
    r = []
    for dt in ("2026-09-30T06:00:00Z", "2026-10-07T06:00:00Z"):
        r += [_row(dt, "KC", "QB", 9, 1, "qb1"), _row(dt, "KC", "LT", 3, 1, "lt1"), _row(dt, "KC", "RT", 7, 1, "rt1"), _row(dt, "KC", "LG", 4, 1, "lg1"),
              _row(dt, "KC", "RG", 6, 1, "rg1"), _row(dt, "KC", "C", 5, 1, "c1"), _row(dt, "KC", "WR", 1, 1, "wr1"), _row(dt, "KC", "WR", 2, 2, "wr2"),
              _row(dt, "KC", "WR", 8, 3, "wr3"), _row(dt, "KC", "TE", 10, 1, "te1"), _row(dt, "KC", "RB", 11, 1, "rb1"), _row(dt, "KC", "RB", 11, 2, "rb2"),
              _row(dt, "KC", "FB", 12, 1, "fb1"), _row(dt, "KC", "LDE", 1, 1, "de1", grp="Base 4-3 D"), _row(dt, "KC", "KR", 5, 3, "kr3", grp="Special Teams")]
    r.append(_row("2026-10-07T06:00:00Z", "BUF", "QB", 9, 1, "bq1"))
    r.append(_row("2026-02-03T06:00:00Z", "KC", "QB", 9, 1, "old_playoff_qb"))                      # a February 2026 chart is still the 2025 season
    r.append(_row("2025-12-30T06:00:00Z", "KC", "QB", 9, 1, "holdout_qb"))
    return pl.DataFrame(r, schema_overrides={"season": pl.Int32, "week": pl.Int32, "club_code": pl.String, "position": pl.String, "depth_team": pl.String, "formation": pl.String})


def test_a_chart_belongs_to_the_season_that_started_in_march():
    s = pl.Series(["2025-08-03T10:09:07Z", "2026-01-02T00:00:00Z", "2026-02-28T23:59:59Z", "2026-03-01T00:00:00Z", "2026-10-07T06:02:29Z"])
    assert s.to_frame("dt").select(DL.dt_season(pl.col("dt")))["dt"].to_list() == [2025, 2025, 2025, 2026, 2026]


def test_the_latest_chart_before_the_cutoff_in_the_old_layout():
    d = DL.chart_as_of(_frame(), "2026-10-08T00:00:00Z", week=5)
    kc = d.filter(pl.col("team") == "KC")
    assert set(d.columns) >= {"season", "week", "team", "gsis_id", "position", "depth_team", "formation", "dt"}
    assert (d["season"] == 2026).all() and (d["week"] == 5).all() and (d["dt"] == "2026-10-07T06:00:00Z").all()           # the newest chart per team, not the one a week earlier
    assert (d["formation"] == "Offense").all() and set(d["position"]) == {"QB", "T", "G", "C", "WR", "TE", "RB", "FB"}       # defense and special teams are not the offensive chart
    assert sorted(kc.filter(pl.col("position") == "T")["gsis_id"].to_list()) == ["lt1", "rt1"] and set(kc.filter(pl.col("position") == "T")["depth_team"]) == {"1"}
    assert kc.filter(pl.col("position") == "WR").sort("depth_team")["gsis_id"].to_list() == ["wr1", "wr2", "wr3"]            # the rank over the three wide-receiver slots is the old depth_team
    assert d.filter(pl.col("team") == "BUF")["gsis_id"].to_list() == ["bq1"]


def test_nothing_after_the_cutoff_is_seen():
    d = DL.chart_as_of(_frame(), "2026-10-01T00:00:00Z", week=4)
    assert (d["dt"] == "2026-09-30T06:00:00Z").all() and "BUF" not in d["team"].to_list()                                # BUF's first chart is after the cutoff: no row, never a later one


def test_the_holdout_season_is_refused_and_its_rows_never_returned():
    with pytest.raises(config.HoldoutError):
        DL.chart_as_of(_frame(), "2025-12-31T00:00:00Z", week=17)
    with pytest.raises(config.HoldoutError):
        DL.chart_as_of(_frame(), "2026-02-10T00:00:00Z", week=1)                                                       # February 2026 is still the 2025 season
    assert not {"old_playoff_qb", "holdout_qb"} & set(DL.chart_as_of(_frame(), "2026-10-08T00:00:00Z", week=5)["gsis_id"].to_list())
    assert not {"old_playoff_qb", "holdout_qb"} & set(DL.live_charts(_frame())["gsis_id"].to_list())


def test_a_chart_row_without_a_player_id_is_dropped_not_zeroed():
    f = _frame().with_columns(gsis_id=pl.when(pl.col("pos_abb") == "FB").then(None).otherwise(pl.col("gsis_id")))
    d = DL.chart_as_of(f, "2026-10-08T00:00:00Z", week=5)
    assert d["gsis_id"].null_count() == 0 and "FB" not in d["position"].to_list()


def test_a_frame_of_nothing_but_holdout_charts_is_refused_and_old_layout_rows_are_ignored():
    only_holdout = _frame().filter(pl.col("gsis_id") == "holdout_qb")
    with pytest.raises(config.HoldoutError):
        DL.live_charts(only_holdout)
    old = _frame().head(3).with_columns(season=pl.lit(2024, dtype=pl.Int32), week=pl.lit(1, dtype=pl.Int32))             # a Phase 3 row (season set) is not this module's input
    assert DL.live_charts(pl.concat([old, _frame().filter(pl.col("gsis_id") == "bq1")])).height == 1


def test_the_asof_snapshot_is_read_only_through_depth_live(tmp_path):
    d = tmp_path / "2026-10-07T00"
    d.mkdir()
    _frame().write_parquet(d / "depth_charts.parquet")
    out = DL.read_asof_charts(d)
    assert not {"old_playoff_qb", "holdout_qb"} & set(out["gsis_id"].to_list()) and out["chart_season"].min() == 2026         # the Jan-Feb 2026 rows of the real file never come back


def test_no_code_outside_the_ingest_savers_and_depth_live_touches_the_asof_snapshots():
    """records/asof/<stamp>/depth_charts.parquet holds 2025-season rows (plan, 'Phase 5 blockers'): feature, model and eval code must not open it."""
    from pathlib import Path
    root = Path(config.ROOT)
    allowed = {root / "ingest" / "save_asof.py", root / "ingest" / "scheduled_snapshot.py", root / "features" / "depth_live.py"}
    bad = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if rel.parts[0] in ("tests", ".git", "data") or path in allowed:
            continue
        text = path.read_text()
        if "records/asof" in text or 'records" / "asof"' in text or "ASOF_DIR" in text or "save_asof" in text or "asof/" in text:
            bad.append(str(rel))
    assert not bad, f"read the asof snapshots only through features.depth_live: {bad}"
