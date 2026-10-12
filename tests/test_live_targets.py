"""Target rows of a live run (live/targets.py): kickoff times, team-game numbers, the eligible player set (plan P5.1/P5.2/P5.6)."""
from datetime import date, datetime, timezone

import duckdb
import polars as pl
import pytest

import config
from eval import backtest as bt
from live import targets as TG

needs_db = pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="no local raw db")


def test_kickoff_times_are_eastern_converted_to_utc():
    assert TG.kickoff_utc("2026-10-18", "13:00") == datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)       # EDT
    assert TG.kickoff_utc("2026-11-08", "13:00") == datetime(2026, 11, 8, 18, 0, tzinfo=timezone.utc)        # EST from 1 Nov
    assert TG.kickoff_utc("2026-10-04", "09:30") == datetime(2026, 10, 4, 13, 30, tzinfo=timezone.utc)       # the London game


def _sched():
    rows = [("2026_01_AAA_BBB", 1, "2026-09-10", "20:20", "BBB", "AAA"), ("2026_02_BBB_CCC", 2, "2026-09-20", "13:00", "CCC", "BBB"),
            ("2026_03_AAA_CCC", 3, "2026-09-27", "13:00", "AAA", "CCC"), ("2026_04_BBB_AAA", 4, "2026-10-04", None, "AAA", "BBB")]
    return pl.DataFrame([dict(game_id=g, season=2026, game_type="REG", week=w, gameday=d, gametime=t, home_team=h, away_team=a) for g, w, d, t, h, a in rows])


def test_a_game_without_a_kickoff_time_is_dropped_never_guessed():
    s = TG.schedule_frame(_sched())
    assert "2026_04_BBB_AAA" not in s["game_id"].to_list() and s.height == 3


def test_team_game_numbers_count_the_team_s_games_in_the_season_by_date():
    games = TG.target_games(_sched(), 2026, 3)
    t = TG.team_rows(_sched(), games).sort("team")
    assert t.select("team", "team_game_num").rows() == [("AAA", 2), ("CCC", 2)]                           # AAA played week 1, CCC week 2; this is each one's second game
    assert t["pf"].null_count() == 2 and t["pa"].null_count() == 2 and t["is_home"].to_list() == [True, False]


def _chart(rows):
    return pl.DataFrame([dict(team=t, gsis_id=g, position=p, depth_team=str(d)) for t, g, p, d in rows])


def _hist(rows):
    return pl.DataFrame([dict(player_id=p, game_id=g, team=t, family=f, gameday=date(2026, 9, d), carries=c, targets=tg, rush_att_ex_kneel=0.0, season=2026, week=w) for p, g, t, f, d, c, tg, w in rows])


def test_chart_slots_make_players_eligible_by_the_one_rule():
    games = TG.target_games(_sched(), 2026, 3)
    trows = TG.team_rows(_sched(), games)
    chart = _chart([("AAA", "qb1", "QB", 1), ("AAA", "rb1", "RB", 1), ("AAA", "rb2", "RB", 2), ("AAA", "rb3", "RB", 3), ("AAA", "fb1", "FB", 1),
                    ("AAA", "wr1", "WR", 1), ("AAA", "wr4", "WR", 4), ("AAA", "te1", "TE", 1), ("AAA", "te2", "TE", 2), ("CCC", "qb9", "QB", 1)])
    empty_h = _hist([]) if False else pl.DataFrame(schema={"player_id": pl.String, "game_id": pl.String, "team": pl.String, "family": pl.String, "gameday": pl.Date, "carries": pl.Float64,
                                                           "targets": pl.Float64, "rush_att_ex_kneel": pl.Float64, "season": pl.Int32, "week": pl.Int32})
    th = pl.DataFrame(schema={"game_id": pl.String, "team": pl.String, "gameday": pl.Date})
    p = TG.player_rows(games, chart, empty_h, th, trows).filter(pl.col("team") == "AAA")
    el = {r["player_id"]: r["elig"] for r in p.iter_rows(named=True)}
    assert el["rb1"] == "rush_att,rush_yds,targets,rec,rec_yds" and el["rb2"] == el["rb1"] and el["fb1"] == el["rb1"]
    assert el["rb3"] == "" and el["wr1"] == "targets,rec,rec_yds" and el["wr4"] == "" and el["te1"] == "targets,rec,rec_yds" and el["te2"] == ""
    assert el["qb1"] == "pass_att,pass_cmp,pass_yds"                                                       # no rushing history: no QB-rush markets, never assumed
    assert p["pf"].null_count() if "pf" in p.columns else True
    assert all(p[c].null_count() == p.height for c in TG.STATS)                                           # no outcome in any stat


def test_usage_makes_an_unlisted_player_eligible_and_a_qb_who_runs_gets_the_qb_rush_markets():
    games = TG.target_games(_sched(), 2026, 3)
    trows = TG.team_rows(_sched(), games)
    chart = _chart([("AAA", "qb1", "QB", 1)])
    hist = _hist([("rbx", "2026_01_AAA_BBB", "AAA", "RB", 10, 12.0, 1.0, 1), ("qb1", "2026_01_AAA_BBB", "AAA", "QB", 10, 6.0, 0.0, 1)])
    hist = hist.with_columns(rush_att_ex_kneel=pl.col("carries"))
    th = pl.DataFrame({"game_id": ["2026_01_AAA_BBB"], "team": ["AAA"], "gameday": [date(2026, 9, 10)]})
    p = TG.player_rows(games, chart, hist, th, trows).filter(pl.col("team") == "AAA")
    el = {r["player_id"]: r["elig"] for r in p.iter_rows(named=True)}
    assert el["rbx"] == "rush_att,rush_yds" and "qb_rush_att,qb_rush_yds" in el["qb1"]                       # 12 carries a game clears the 5-carry clause; 6 QB rushes clear the 4-attempt clause


@needs_db
def test_replay_of_2024_week_10_finds_the_standard_eligible_set():
    data = bt.load_backtest_data()
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("select * from (select distinct on (game_id) * from schedules order by game_id, pulled_at desc) where season=2024").pl()
    old = con.execute("select club_code as team, gsis_id, position, depth_team from (select distinct on (season,week,club_code,gsis_id,depth_team,position) * from depth_charts "
                      "where season=2024 and week=10 and formation='Offense' and game_type='REG' order by season,week,club_code,gsis_id,depth_team,position,pulled_at desc) "
                      "where gsis_id is not null").pl()
    games = TG.target_games(sched, 2024, 10)
    cut = games["gameday"].min()
    ht, hp = data.team_log.filter(pl.col("gameday") < cut), data.player_log.filter(pl.col("gameday") < cut)
    trows = TG.team_rows(sched, games)
    std_t = data.team_log.filter((pl.col("season") == 2024) & (pl.col("week") == 10)).sort("game_id", "team")
    assert trows.sort("game_id", "team").select("team", "opponent", "is_home", "team_game_num").equals(std_t.select("team", "opponent", "is_home", "team_game_num"))
    pr = TG.player_rows(games, old, hp, ht, trows)
    std = data.player_log.filter((pl.col("season") == 2024) & (pl.col("week") == 10) & (pl.col("elig") != ""))
    live = {(r["player_id"], r["game_id"]): r["elig"] for r in pr.filter(pl.col("elig") != "").iter_rows(named=True)}
    ref = {(r["player_id"], r["game_id"]): r["elig"] for r in std.iter_rows(named=True)}
    assert all(live[k] == ref[k] for k in live.keys() & ref.keys())                                          # the same player gets the same markets
    assert len(ref.keys() - live.keys()) / len(ref) < 0.03                                                   # recall: a usage-eligible player back from a long absence can be missed
    played = set(zip(data.player_log.filter((pl.col("season") == 2024) & (pl.col("week") == 10))["player_id"], data.player_log.filter((pl.col("season") == 2024) & (pl.col("week") == 10))["game_id"]))
    assert len(live.keys() - played) > 0                                                                     # and eligible players who did not play are predicted too (did-not-play rows)


def test_a_traded_player_on_the_new_teams_chart_is_not_also_a_recent_player_of_his_old_team():
    games = TG.target_games(_sched(), 2026, 3)
    trows = TG.team_rows(_sched(), games)
    chart = _chart([("CCC", "wrx", "WR", 1), ("AAA", "qb1", "QB", 1)])                               # wrx now on CCC's chart
    hist = _hist([("wrx", "2026_01_AAA_BBB", "AAA", "WR", 10, 0.0, 6.0, 1)])                          # but his last games were with AAA
    th = pl.DataFrame({"game_id": ["2026_01_AAA_BBB"], "team": ["AAA"], "gameday": [date(2026, 9, 10)]})
    p = TG.player_rows(games, chart, hist, th, trows)
    assert p.filter(pl.col("player_id") == "wrx")["team"].to_list() == ["CCC"]
