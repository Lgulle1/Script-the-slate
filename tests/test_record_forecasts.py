"""Forecast recorder (option C of the 4e decision): a tracked, append-only record of forecasts as they stood before kickoff."""
from datetime import datetime, timedelta

import polars as pl
import pytest

import config
from ingest import record_forecasts as RF

NOW = datetime(2026, 10, 14, 12, 0)


def _req(gid, hours_ahead, indoor=False):
    return dict(game_id=gid, lat=40.0, lon=-75.0, kickoff=NOW + timedelta(hours=hours_ahead), indoor=indoor)


def _fake_fetch(reqs, now):
    return [dict(game_id=r["game_id"], temp_f=60.0 + i, wind_mph=8.0, precip_in=0.0, precip_prob_pct=None, source="meteostat_forecast", station_id="X",
                 station_km=5.0, indoor=r.get("indoor"), hours_before_kickoff=(r["kickoff"] - now).total_seconds() / 3600) for i, r in enumerate(reqs)]


def test_only_future_live_season_games_inside_the_horizon_are_selected():
    reqs = [_req("2026_07_A_B", 30), _req("2026_07_C_D", 200), _req("2026_06_E_F", -2), _req("2026_07_G_H", 0), _req("2025_07_I_J", 30)]
    got = [r["game_id"] for r in RF.select_requests(reqs, NOW, horizon_hours=96)]
    assert got == ["2026_07_A_B"]                              # past, kickoff == now, beyond 96 h, and a holdout-season game are all out


def test_snapshots_are_appended_with_their_lead_time_and_never_rewritten(tmp_path):
    p = tmp_path / "log.csv"
    assert RF.record([_req("2026_07_A_B", 30), _req("2026_07_C_D", 50)], NOW, p, fetch=_fake_fetch) == 2
    first = p.read_text()
    assert RF.record([_req("2026_07_A_B", 24)], NOW + timedelta(hours=6), p, fetch=_fake_fetch) == 1       # a later snapshot of the same game is added
    log = pl.read_csv(p)
    assert log.height == 3 and p.read_text().startswith(first)                                            # the earlier rows are byte-for-byte unchanged
    a = log.filter(pl.col("game_id") == "2026_07_A_B")
    assert a["hours_before_kickoff"].to_list() == pytest.approx([30.0, 18.0]) and a["pulled_at"].n_unique() == 2


def test_a_snapshot_taken_again_within_the_gap_is_not_duplicated(tmp_path):
    p = tmp_path / "log.csv"
    RF.record([_req("2026_07_A_B", 30)], NOW, p, fetch=_fake_fetch)
    assert RF.record([_req("2026_07_A_B", 30)], NOW + timedelta(hours=1), p, fetch=_fake_fetch, min_gap_hours=3) == 0
    assert pl.read_csv(p).height == 1


def test_a_failed_lookup_is_recorded_as_missing_with_null_values_not_zero(tmp_path):
    def failing(reqs, now):
        return [dict(game_id=r["game_id"], temp_f=None, wind_mph=None, precip_in=None, precip_prob_pct=None, source="missing", station_id=None,
                     station_km=None, indoor=r.get("indoor"), hours_before_kickoff=None) for r in reqs]
    p = tmp_path / "log.csv"
    RF.record([_req("2026_07_A_B", 30)], NOW, p, fetch=failing)
    r = pl.read_csv(p).row(0, named=True)
    assert r["source"] == "missing" and r["temp_f"] is None and r["wind_mph"] is None


def test_the_recorder_refuses_the_holdout_season(tmp_path):
    with pytest.raises(config.HoldoutError):
        RF.record([_req("2025_07_A_B", 30)], NOW, tmp_path / "log.csv", fetch=_fake_fetch, select=False)


def test_the_live_schedule_comes_from_nflverse_when_there_is_no_raw_database(tmp_path):
    """A fresh container has no data/raw/raw.duckdb: the recorder must still find the live season's games."""
    seen = []

    def loader(seasons):
        seen.append(seasons)
        return pl.DataFrame({"game_id": ["2026_07_A_B"], "gameday": ["2026-10-18"], "gametime": ["13:00"], "home_team": ["BUF"], "stadium_id": [None],
                             "location": ["Home"], "roof": ["outdoors"], "season": [2026]}, schema_overrides={"stadium_id": pl.String})
    games = RF.live_schedule(tmp_path / "missing.duckdb", loader=loader)
    assert seen == [[config.HOLDOUT_SEASON + 1]] and games[0]["game_id"] == "2026_07_A_B"
