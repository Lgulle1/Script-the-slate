"""4b.2 expected margin / total and 4b.3 state shares, dropback rates and play accounting."""
import numpy as np
import polars as pl
import pytest

import config
from features import team_ratings as tr
from models import game_state as gs

db = pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="raw database not present")
COEF = {"c": 0.4, "x_rating": 0.75, "x_home": 1.6, "x_rest": 0.05, "x_qb": 0.35}


def _side(x_rating, x_home, x_rest, x_qb, league=22.0):
    return dict(league_pts=league, x_rating=x_rating, x_home=x_home, x_rest=x_rest, x_qb=x_qb)


# ---------------------------------------------------------------- 4b.2
def test_swapping_home_and_away_flips_the_margin_and_keeps_the_total():
    a = dict(x_rating=3.1, x_rest=2.0, x_qb=-1.2)          # team A's view of the matchup
    b = dict(x_rating=-0.8, x_rest=-2.0, x_qb=0.4)         # team B's view (rest difference is antisymmetric)
    # neutral site: swapping the labels flips the margin exactly and keeps the total
    m1, t1 = gs.game_prediction(_side(**a, x_home=0.0), _side(**b, x_home=0.0), COEF)
    m2, t2 = gs.game_prediction(_side(**b, x_home=0.0), _side(**a, x_home=0.0), COEF)
    assert m1 == pytest.approx(-m2) and t1 == pytest.approx(t2)
    # with home field: moving the game to the other stadium moves the margin by twice the home edge, the total not at all
    m_a_home, t_a_home = gs.game_prediction(_side(**a, x_home=0.5), _side(**b, x_home=-0.5), COEF)
    m_b_home, t_b_home = gs.game_prediction(_side(**b, x_home=0.5), _side(**a, x_home=-0.5), COEF)
    assert m_a_home - COEF["x_home"] == pytest.approx(-(m_b_home - COEF["x_home"]))
    assert t_a_home == pytest.approx(t_b_home)


def test_no_market_line_in_the_schedule_columns():
    assert not set(gs.SCHEDULE_COLUMNS) & set(tr.FORBIDDEN_COLUMNS)
    assert not {"home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name", "result", "total"} & set(gs.SCHEDULE_COLUMNS)


@pytest.fixture(scope="module")
def inp():
    return gs.load_inputs(max_season=2022)


@db
def test_no_market_line_anywhere_in_the_inputs(inp):
    forbidden = set(tr.FORBIDDEN_COLUMNS) | {"home_qb_id", "away_qb_id"}
    for name in ("games", "plays", "qb", "chart", "status", "roster"):
        assert not forbidden & set(getattr(inp, name).columns), name
    assert tuple(inp.games.columns) == gs.SCHEDULE_COLUMNS


def _truncate(inp, season, week):
    before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    upto = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") <= week))
    games = pl.concat([inp.games.filter(before),
                       inp.games.filter((pl.col("season") == season) & (pl.col("week") == week)).with_columns(
                           home_score=pl.lit(None, dtype=inp.games["home_score"].dtype), away_score=pl.lit(None, dtype=inp.games["away_score"].dtype))])
    return gs.Inputs(games, inp.plays.filter(before), inp.qb.filter(before), inp.chart.filter(upto), inp.status.filter(upto), inp.roster.filter(upto))


@db
def test_walk_forward_only_later_data_never_changes_a_weeks_prediction(inp):
    season, week = 2022, 6
    full_r = tr.compute_ratings(inp.plays, n_boot=0)
    tp_full, _ = gs.walk_forward(gs.pregame_features(inp, full_r))
    cut = _truncate(inp, season, week)
    tp_cut, _ = gs.walk_forward(gs.pregame_features(cut, tr.compute_ratings(cut.plays, n_boot=0)))
    pick = lambda t: t.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("game_id", "team").select("game_id", "team", "pts_mean")
    a, b = pick(tp_full), pick(tp_cut)
    assert a.height > 20 and a.select("game_id", "team").equals(b.select("game_id", "team"))
    # identical up to floating-point summation order (the truncated history is a shorter array)
    assert np.allclose(a["pts_mean"].to_numpy(), b["pts_mean"].to_numpy(), rtol=0, atol=1e-9)


@db
def test_sds_are_the_rms_of_earlier_walk_forward_errors(inp):
    r = tr.compute_ratings(inp.plays, n_boot=0)
    tp, _ = gs.walk_forward(gs.pregame_features(inp, r))
    ge = gs.game_expectations(tp, inp.games)
    row = ge.filter((pl.col("season") == 2022) & (pl.col("week") == 10)).row(0, named=True)
    prior = ge.filter((pl.col("season") * 100 + pl.col("week")) < 202210)
    assert row["margin_sd"] == pytest.approx(float(np.sqrt(((prior["margin"] - prior["margin_mean"]) ** 2).mean())))
    assert row["total_sd"] == pytest.approx(float(np.sqrt(((prior["total"] - prior["total_mean"]) ** 2).mean())))
    assert ge["margin_sd"].drop_nulls().n_unique() > 10            # changes week to week: not a constant


def test_qb_quality_is_shrunk_toward_the_league_mean():
    qb = pl.DataFrame({"season": [2021] * 3, "week": [1, 1, 2], "game_id": ["g1", "g1", "g2"], "team": ["A", "A", "A"],
                       "qb_id": ["x", "y", "x"], "dropbacks": [40.0, 360.0, 60.0], "epa": [20.0, 0.0, 10.0]})
    book = gs.QBBook(qb)
    L = 20.0 / 400.0                                                     # league mean before week 2
    assert book.league_mean(2021, 2) == pytest.approx(L)
    assert book.quality("x", 2021, 2) == pytest.approx((20.0 + gs.QB_K * L) / (40.0 + gs.QB_K))
    assert book.quality("x", 2021, 1) == 0.0                             # nothing before week 1: league mean of nothing
    assert book.quality("nobody", 2021, 2) == pytest.approx(L)
