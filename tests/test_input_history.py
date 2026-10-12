"""The scoped input-history unlock (config.input_history) and its one user, live/history.py (plan "Phase 5 gates", P5.4 guardrails 1-3)."""
import re
from datetime import date
from pathlib import Path

import polars as pl
import pytest

import config
from live import history as H

ROOT = Path(config.ROOT)


def test_outside_the_block_the_holdout_stays_locked_inside_it_only_up_to_the_requested_season():
    with pytest.raises(config.HoldoutError):
        config.cap_season(2025)
    with config.input_history(2026):
        assert config.cap_season(2025) == 2025 and config.cap_season(2026) == 2026
        with pytest.raises(config.HoldoutError):
            config.cap_season(2027)                                                         # nothing beyond what was asked for
        assert config.cap_season(None) == max(config.BACKTEST_SEASONS)                       # "None" never means "everything", even inside
    with pytest.raises(config.HoldoutError):
        config.cap_season(2025)                                                             # and the unlock is gone after the block, also after an exception
    with pytest.raises(RuntimeError):
        with config.input_history(2026):
            raise RuntimeError("boom")
    with pytest.raises(config.HoldoutError):
        config.cap_season(2026)


def test_only_live_history_opens_the_unlock():
    users = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py") if p.relative_to(ROOT).parts[0] not in ("tests", ".git", "data")
             and re.search(r"(?<![\w])input_history\(", p.read_text()) and p.name != "config.py"]
    assert users == ["live/history.py"], users


def test_history_before_a_2025_cutoff_never_sees_2026_nor_the_cutoff_week():
    cutoff = date(2025, 11, 9)                                                               # a week-10 Sunday of 2025
    h = H.read_input_history(cutoff)
    assert h.player_log["gameday"].max() < cutoff and h.team_log["gameday"].max() < cutoff
    assert h.team_log["season"].max() == 2025 and h.team_log.filter(pl.col("season") == 2026).is_empty()
    wk10 = h.team_log.filter((pl.col("season") == 2025) & (pl.col("week") >= 10))
    assert wk10.filter(pl.col("gameday") >= cutoff).is_empty()


def test_history_for_a_2026_target_reaches_2025_and_2026_to_date_and_stops_at_the_cutoff():
    cutoff = date(2026, 10, 4)
    h = H.read_input_history(cutoff)
    seasons = set(h.team_log["season"].unique().to_list())
    assert {2024, 2025, 2026} <= seasons and h.team_log["gameday"].max() < cutoff


def test_the_history_object_is_marked_input_only_and_is_not_a_backtest_object():
    h = H.read_input_history(date(2026, 10, 4))
    assert h.input_only is True and h.cutoff == date(2026, 10, 4)
    from eval import backtest as bt
    assert not isinstance(h, (bt.History, bt.BacktestData))
    with pytest.raises(RuntimeError, match="holdout"):
        bt.assert_no_holdout(h.team_log)                                                      # the walk-forward harness still refuses these rows if they are ever handed to it


def test_a_cutoff_before_the_live_seasons_is_just_the_dev_path_and_needs_no_unlock():
    h = H.read_input_history(date(2024, 10, 6))
    assert h.team_log["season"].max() == 2024 and h.team_log["gameday"].max() < date(2024, 10, 6)


def test_a_2025_cutoff_asks_the_loaders_for_nothing_beyond_2025(monkeypatch):
    seen = []
    real = H.bl.build_team_game_log
    monkeypatch.setattr(H.bl, "build_team_game_log", lambda raw_db, max_season=None: (seen.append(max_season), real(raw_db, max_season=max_season))[1])
    H.read_input_history(date(2025, 11, 9))
    assert seen and set(seen) == {2025}                                                                   # the unlock reaches exactly the cutoff's season, so a 2026 row is never even loaded
