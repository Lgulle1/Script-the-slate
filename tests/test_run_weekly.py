"""run_weekly.py: the run's clock is read after the data pull (a fresh container's first run saw 'no upcoming game' because its clock preceded its own pull)."""
from datetime import datetime, timezone

import run_weekly as RW


def test_the_clock_is_read_after_the_pull(monkeypatch, tmp_path):
    seen = {}
    pulled = {}

    def fake_refresh():
        pulled["at"] = datetime.now(timezone.utc)
        return {}

    def fake_run(now, **kw):
        seen["now"] = now
        return dict(status="no upcoming game", written=0)

    monkeypatch.setattr(RW.refresh_live, "refresh", fake_refresh)
    monkeypatch.setattr(RW.weekly, "run", fake_run)
    monkeypatch.setattr(RW.config, "RAW_DUCKDB_PATH", tmp_path / "exists.duckdb")
    (tmp_path / "exists.duckdb").write_bytes(b"")
    RW.main(["--refresh", "--no-sync", "--store", str(tmp_path / "s"), "--lines", str(tmp_path / "l.csv")])
    assert seen["now"] >= pulled["at"]


def test_a_fixed_now_is_used_as_given(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(RW.weekly, "run", lambda now, **kw: (seen.update(now=now), dict(status="no upcoming game", written=0))[1])
    monkeypatch.setattr(RW.config, "RAW_DUCKDB_PATH", tmp_path / "exists.duckdb")
    (tmp_path / "exists.duckdb").write_bytes(b"")
    RW.main(["--now", "2026-10-16T20:00:00Z", "--no-sync", "--store", str(tmp_path / "s"), "--lines", str(tmp_path / "l.csv")])
    assert seen["now"] == datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)
