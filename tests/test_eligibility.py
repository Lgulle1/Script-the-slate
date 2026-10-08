"""The one eligibility rule (config.ELIGIBLE_PLAYER_RULE) for the rush and receiving markets."""
import copy
from datetime import date

import duckdb
import polars as pl
import pytest

import config
from eval import baselines as bl
from eval import eligibility as el

RUSH, RECV = {"rush_att", "rush_yds"}, {"targets", "rec", "rec_yds"}


def frame(rows):
    base = dict(family="WR", slot=0, chart_rb=False, chart_fb=False, chart_wr=False, chart_te=False,
                avg_carries_prev=None, avg_targets_prev=None, avg_qb_rush_prev=None)
    return pl.DataFrame([base | r for r in rows], schema={
        "family": pl.String, "slot": pl.Int32, "chart_rb": pl.Boolean, "chart_fb": pl.Boolean, "chart_wr": pl.Boolean,
        "chart_te": pl.Boolean, "avg_carries_prev": pl.Float64, "avg_targets_prev": pl.Float64, "avg_qb_rush_prev": pl.Float64})


def pool(rows, rule=None):
    return [set(el.markets_of(s)) for s in el.add_eligibility(frame(rows), rule)["elig"].to_list()]


def test_each_clause_of_the_rule_and_the_markets_it_makes_a_player_eligible_for():
    got = pool([
        dict(family="RB", chart_rb=True),                      # RB1 / RB2 on the chart
        dict(family="RB", chart_fb=True),                      # FB1 on the chart
        dict(family="WR", chart_wr=True),                      # WR1-3
        dict(family="TE", chart_te=True),                      # TE1
        dict(family="RB", avg_carries_prev=5.0),               # RB3+ / unlisted RB or fullback, exactly at the carries bar
        dict(family="RB", avg_carries_prev=4.99),              # just below
        dict(family="WR", avg_targets_prev=2.0),               # exactly at the targets bar
        dict(family="WR", avg_targets_prev=1.99),
        dict(family="TE", avg_targets_prev=None),              # no history -> usage clause false
    ])
    assert got[0] == got[1] == RUSH | RECV                      # backs and fullbacks: rush AND receiving
    assert got[2] == got[3] == RECV                             # WR / TE chart slots: receiving only
    assert got[4] == RUSH and got[5] == set()                   # carries clause: rushing markets only
    assert got[6] == RECV and got[7] == set()                   # targets clause: receiving markets only
    assert got[8] == set()


def test_rushing_markets_are_rb_group_only_no_qb_wr_or_te_ever():
    got = pool([
        dict(family="QB", slot=1, avg_carries_prev=9.0),            # a very mobile QB: NOT in the RB rush pool (he has his own markets)
        dict(family="WR", chart_wr=True, avg_carries_prev=8.0),     # a WR with jet-sweep volume
        dict(family="TE", chart_te=True, avg_carries_prev=6.0),
        dict(family="RB", chart_rb=True, avg_carries_prev=0.5),     # but the RB group is in
    ])
    assert not (got[0] & RUSH) and not (got[1] & RUSH) and not (got[2] & RUSH)
    assert got[1] == RECV and got[2] == RECV                    # they still count for receiving via their chart slot
    assert got[3] == RUSH | RECV


def test_qb_rushing_pool_is_qb1_who_averaged_four_kneel_free_attempts():
    got = pool([
        dict(family="QB", slot=1, avg_qb_rush_prev=4.0),       # exactly at the bar
        dict(family="QB", slot=1, avg_qb_rush_prev=3.99),
        dict(family="QB", slot=2, avg_qb_rush_prev=9.0),       # a backup is never in
        dict(family="QB", slot=1, avg_qb_rush_prev=None),      # no history
        dict(family="RB", chart_rb=True, avg_qb_rush_prev=8.0),
    ])
    qbm = {"qb_rush_att", "qb_rush_yds"}
    assert got[0] == {"pass_att", "pass_cmp", "pass_yds"} | qbm
    assert got[1] == {"pass_att", "pass_cmp", "pass_yds"}
    assert got[2] == set() and got[3] == {"pass_att", "pass_cmp", "pass_yds"} and not (got[4] & qbm)


def test_qb_rush_threshold_is_the_config_constant(monkeypatch):
    monkeypatch.setattr(config, "QB_RUSH_MIN_ATT", 6)
    got = pool([dict(family="QB", slot=1, avg_qb_rush_prev=5.0), dict(family="QB", slot=1, avg_qb_rush_prev=6.0)])
    assert "qb_rush_att" not in got[0] and "qb_rush_att" in got[1]


def test_all_markets_scope_is_the_literal_reading_and_switchable():
    rule = copy.deepcopy(config.ELIGIBLE_PLAYER_RULE)
    rule["scope"] = "all_markets"
    got = pool([dict(family="RB", slot=0, avg_carries_prev=6.0), dict(family="WR", avg_targets_prev=3.0)], rule)
    assert got[0] == RUSH | RECV                                # any clause -> all five markets, for the RB group
    assert got[1] == RECV                                       # a WR still can never enter the rush markets
    with pytest.raises(ValueError):
        rule["scope"] = "nope"
        pool([dict()], rule)


def test_usage_average_is_the_previous_four_games_played_and_never_the_current_game():
    df = pl.DataFrame({"player_id": ["a"] * 6 + ["b"] * 2, "gameday": [date(2024, 9, i + 1) for i in range(6)] + [date(2024, 9, 1), date(2024, 9, 2)],
                       "carries": [10, 0, 2, 4, 6, 100, 50, 50], "targets": [1, 1, 1, 1, 1, 9, 0, 0],
                       "rush_att_ex_kneel": [9.0, 0, 2, 4, 6, 100, 49, 50]}).sort("player_id", "gameday")
    out = el.add_usage_averages(df)
    assert out["avg_carries_prev"].to_list()[:6] == [None, 10.0, 5.0, 4.0, 4.0, 3.0]          # window of up to 4 earlier games, min 1
    assert out["avg_carries_prev"].to_list()[6:] == [None, 50.0]                              # players do not mix
    assert out["avg_qb_rush_prev"].to_list()[:3] == [None, 9.0, 4.5]                          # the QB-rushing average uses the kneel-free column
    # changing a game's own (or any later) stats never changes its own eligibility inputs
    bumped = df.with_columns(carries=pl.when(pl.col("player_id") == "a").then(pl.col("carries") + 1000).otherwise(pl.col("carries")))
    assert el.add_usage_averages(bumped)["avg_carries_prev"].to_list()[0] is None


def _chart_db(tmp_path, rows):
    db = tmp_path / "raw.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE depth_charts (season INTEGER, week INTEGER, game_type VARCHAR, formation VARCHAR, club_code VARCHAR, "
                "gsis_id VARCHAR, position VARCHAR, depth_position VARCHAR, depth_team VARCHAR, pulled_at TIMESTAMP)")
    for gsis, pos, dpos, dt in rows:
        con.execute("INSERT INTO depth_charts VALUES (2023, 5, 'REG', 'Offense', 'KC', ?, ?, ?, ?, TIMESTAMP '2023-10-05')", [gsis, pos, dpos, dt])
    con.close()
    return db


def test_chart_flags_rank_backs_and_fullbacks_separately_and_apply_the_slot_numbers(tmp_path):
    db = _chart_db(tmp_path, [
        ("fb1", "RB", "FB", "1"),      # a fullback listed under RB, taking depth 1 -> must not make the back behind him RB2
        ("rb1", "RB", "RB", "2"), ("rb2", "RB", "RB", "3"), ("rb3", "RB", "RB", "4"),
        ("fb_pos", "FB", "FB", "1"), ("fb2", "FB", "FB", "2"),
        ("wr1", "WR", "WR", "1"), ("wr3", "WR", "WR", "3"), ("wr4", "WR", "WR", "4"),
        ("te1", "TE", "TE", "1"), ("te2", "TE", "TE", "2"),
    ])
    f = {r["gsis_id"]: r for r in bl.build_chart_flags(db).iter_rows(named=True)}
    assert f["rb1"]["chart_rb"] and f["rb2"]["chart_rb"] and not f["rb3"]["chart_rb"]        # RB1, RB2 only
    assert not f["fb1"]["chart_rb"] and f["fb1"]["chart_fb"]                                   # FB under the RB group is FB1
    assert f["fb_pos"]["chart_fb"] and not f["fb2"]["chart_fb"]                                # position FB: FB1 only
    assert f["wr1"]["chart_wr"] and f["wr3"]["chart_wr"] and not f["wr4"]["chart_wr"]          # WR1-WR3
    assert f["te1"]["chart_te"] and not f["te2"]["chart_te"]                                   # TE1 only


def test_chart_flags_refuse_the_holdout(tmp_path):
    with pytest.raises(config.HoldoutError):
        bl.build_chart_flags(tmp_path / "x.duckdb", max_season=2025)


def test_rule_config_and_result_naming():
    rule = config.ELIGIBLE_PLAYER_RULE
    assert rule["usage"] == {"window_games": 4, "min_carries": 5.0, "min_targets": 2.0}
    assert rule["depth_chart"] == {"RB": (1, 2), "FB": (1,), "WR": (1, 2, 3), "TE": (1,)}
    assert rule["version"] == "eligibility_v2" and rule["scope"] == "per_market"
    assert config.result_path("baseline_results").name == "baseline_results_eligibility_v2.parquet"
    assert config.result_path("baseline_results", "slotpool_v1").name == "baseline_results_slotpool_v1.parquet"
    assert config.result_path("baseline_results") != config.result_path("baseline_results", "slotpool_v1")


@pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="no local raw db")
def test_real_log_pools_cover_more_of_the_work_and_never_look_ahead():
    log = bl.build_player_game_log()
    assert {"elig", "chart_rb", "chart_fb", "chart_wr", "chart_te", "avg_carries_prev", "avg_targets_prev", "avg_qb_rush_prev",
            "rush_att_ex_kneel", "rush_yds_ex_kneel"} <= set(log.columns)
    rush = log.filter(pl.col("elig").str.split(",").list.contains("rush_att"))
    assert set(rush["family"]) == {"RB"}                         # no QB, WR or TE in the rushing pool
    qb = log.filter(pl.col("elig").str.split(",").list.contains("qb_rush_att"))
    assert set(qb["family"]) == {"QB"} and (qb["slot"] == 1).all() and (qb["avg_qb_rush_prev"] >= config.QB_RUSH_MIN_ATT).all()
    recv = log.filter(pl.col("elig").str.contains("targets"))
    assert rush["carries"].sum() / log["carries"].sum() > 0.65 and recv["targets"].sum() / log["targets"].sum() > 0.92
    # every usage-only member really did average >= the bar over EARLIER games (check one player by hand)
    row = rush.filter(~(pl.col("chart_rb") | pl.col("chart_fb"))).row(0, named=True)  # a usage-only RB-group member
    earlier = log.filter((pl.col("player_id") == row["player_id"]) & (pl.col("gameday") < row["gameday"])).sort("gameday").tail(4)
    assert earlier["carries"].mean() == pytest.approx(row["avg_carries_prev"]) and row["avg_carries_prev"] >= 5.0


def test_kneels_are_removed_from_qb_rushing_attempts_and_yards(tmp_path):
    db = tmp_path / "raw.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE pbp (game_id VARCHAR, play_id INTEGER, season INTEGER, qb_kneel DOUBLE, rusher_player_id VARCHAR, "
                "yards_gained DOUBLE, pulled_at TIMESTAMP)")
    con.execute("INSERT INTO pbp VALUES ('g1',1,2023,1,'q1',-1,TIMESTAMP '2023-01-01'), ('g1',2,2023,1,'q1',-2,TIMESTAMP '2023-01-01'),"
                " ('g1',3,2023,0,'q1',7,TIMESTAMP '2023-01-01'), ('g1',2,2023,1,'q1',-2,TIMESTAMP '2023-02-01')")  # play 2 pulled twice
    con.close()
    k = bl.build_kneels(db).row(0, named=True)
    assert (k["game_id"], k["player_id"], k["kneels"], k["kneel_yards"]) == ("g1", "q1", 2.0, -3.0)   # duplicates collapsed, runs not counted
    with pytest.raises(config.HoldoutError):
        bl.build_kneels(db, max_season=2025)


@pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="no local raw db")
def test_real_qb_kneel_free_stats_match_the_play_by_play():
    log = bl.build_player_game_log()
    qb = log.filter(pl.col("family") == "QB")
    assert (qb["rush_att_ex_kneel"] <= qb["carries"]).all() and (qb["rush_att_ex_kneel"] >= 0).all()
    assert qb["rush_att_ex_kneel"].sum() < qb["carries"].sum()                   # kneels really were removed
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    truth = con.execute("SELECT game_id, rusher_player_id AS player_id, sum(rush_attempt) AS att FROM (SELECT DISTINCT ON (game_id, play_id) * "
                        "FROM pbp WHERE season BETWEEN 2020 AND 2024 ORDER BY game_id, play_id, pulled_at DESC) "
                        "WHERE rush_attempt = 1 AND qb_kneel = 0 AND rusher_player_id IS NOT NULL GROUP BY 1, 2").pl()
    m = qb.join(truth, on=["game_id", "player_id"], how="left").with_columns(pl.col("att").fill_null(0.0))
    assert (m["rush_att_ex_kneel"] == m["att"]).mean() > 0.97
