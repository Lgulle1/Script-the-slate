import pytest

import config


def test_abbreviations_match_nflverse_schedules():
    nflreadpy = pytest.importorskip("nflreadpy")
    try:
        s = nflreadpy.load_schedules(config.BACKTEST_SEASONS)
    except Exception as e:  # network unavailable
        pytest.skip(f"schedules unavailable: {e}")
    teams = set(s["home_team"].to_list()) | set(s["away_team"].to_list())
    assert teams == set(config.TEAM_ABBR_TO_NAME)
