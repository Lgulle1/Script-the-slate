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


# ---------------------------------------------------------------- 4b.3
from fractions import Fraction


def test_play_accounting_identity_holds_exactly_on_synthetic_data():
    rng = np.random.default_rng(7)
    for _ in range(200):
        plays = Fraction(int(rng.integers(45, 80)))
        raw = [Fraction(int(v)) for v in rng.integers(1, 100, 5)]
        shares = [v / sum(raw) for v in raw]
        rates = [Fraction(int(v), 1000) for v in rng.integers(300, 900, 5)]
        sack, scr = Fraction(int(rng.integers(20, 120)), 1000), Fraction(int(rng.integers(10, 80)), 1000)
        a = gs.play_accounting(plays, shares, rates, sack, scr)
        assert sum(shares) == 1
        assert a["dropbacks"] == plays * sum(s * r for s, r in zip(shares, rates))
        assert a["pass_att"] + a["sacks"] + a["scrambles"] == a["dropbacks"]            # dropbacks = attempts + sacks + scrambles
        assert a["rush_att"] + a["dropbacks"] - a["scrambles"] == a["plays"]            # plays = designed runs + dropbacks
        assert a["pass_att"] + a["sacks"] + a["rush_att"] == a["plays"]                 # every play is a pass, a sack or a rush
        assert a["pass_att"] == a["dropbacks"] - a["sacks"] - a["scrambles"]
        assert a["rush_att"] == a["plays"] - a["dropbacks"] + a["scrambles"]
    # the same function on numpy floats: identities to rounding
    f = gs.play_accounting(np.array([64.0, 58.5]), [np.array([0.2, 0.3])] * 5, [np.array([0.6, 0.55])] * 5, np.array([0.06, 0.07]), np.array([0.04, 0.05]))
    assert np.allclose(f["pass_att"] + f["sacks"] + f["rush_att"], f["plays"], atol=1e-12)


def _synthetic_state_rows(n=3000, seed=3):
    """Team-games whose plays per state follow a known softmax in the expected margin."""
    rng = np.random.default_rng(seed)
    b0 = np.array([-0.2, 0.3, 0.0, 0.2, -0.5])
    b1 = np.array([-1.6, -0.6, 0.0, 0.6, 1.5])
    x = rng.normal(0, 6, n)
    z = b0 + np.outer(x / gs.MARGIN_SCALE, b1)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    counts = np.array([rng.multinomial(65, pi) for pi in p]).astype(float)
    cols = {"exp_margin": x, **{f"plays_{s}": counts[:, i] for i, s in enumerate(gs.STATES)}}
    return pl.DataFrame(cols), p, x


def test_state_shares_sum_to_one_and_recover_a_planted_curve():
    rows, p_true, x = _synthetic_state_rows()
    m = gs.fit_state_shares(rows, "exp_margin")
    sh = gs.state_shares(m, x)
    assert sh.shape == (len(x), 5)
    assert np.allclose(sh.sum(axis=1), 1.0, atol=1e-12)
    assert np.abs(sh - p_true).max() < 0.03
    grid = gs.state_shares(m, np.linspace(-40, 40, 81))
    assert np.allclose(grid.sum(axis=1), 1.0, atol=1e-12) and (grid > 0).all()
    assert np.all(np.diff(grid[:, -1]) > 0) and np.all(np.diff(grid[:, 0]) < 0)    # lead 9+ rises, trail 9+ falls with the margin


def test_dropback_rate_is_shrunk_toward_the_league_with_k_100():
    assert gs.shrunk_rate(30.0, 50.0, 0.5, gs.STATE_K) == pytest.approx((30 + 100 * 0.5) / 150)
    assert gs.shrunk_rate(0.0, 0.0, 0.42, gs.STATE_K) == pytest.approx(0.42)        # no plays: the league rate
    big = gs.shrunk_rate(7000.0, 10000.0, 0.5, gs.STATE_K)
    assert abs(big - 0.7) < 0.003                                                   # many plays: his own rate


def test_team_rates_shrink_each_state_with_effective_plays():
    clock = {(2021, w): float(w) for w in range(1, 20)}
    rows = pl.DataFrame({"game_id": ["g1", "g1"], "season": [2021, 2021], "week": [1, 1], "team": ["A", "B"], "plays": [60.0, 60.0],
                         "dropbacks": [40.0, 20.0], "sacks": [4.0, 0.0], "scrambles": [2.0, 1.0],
                         **{f"plays_{s}": [12.0, 12.0] for s in gs.STATES}, **{f"db_{s}": [8.0, 4.0] for s in gs.STATES}})
    r = gs.team_rates(rows, clock, 2021, 2).sort("team")
    w = 0.5 ** (1 / config.RECENCY_HALF_LIFE_GAMES)
    league = (8 + 4) / (12 + 12)
    assert r["db_rate_tied"][0] == pytest.approx((w * 8 + gs.STATE_K * league) / (w * 12 + gs.STATE_K))
    assert r["db_rate_tied"][1] == pytest.approx((w * 4 + gs.STATE_K * league) / (w * 12 + gs.STATE_K))
    lsk = 4 / 60
    assert r["sack_rate"][0] == pytest.approx((w * 4 + gs.RATE_K * lsk) / (w * 40 + gs.RATE_K))


@db
def test_state_rows_match_the_team_game_rates_table():
    from features import phase4_inputs as p4
    sr = gs.load_state_rows(max_season=2021)
    tgr = p4.build_team_game_rates(max_season=2021)
    j = tgr.join(sr, on=["game_id", "team"], how="inner", suffix="_gs")
    assert j.height == tgr.height
    for c in ("plays", "dropbacks", "sacks", "scrambles"):
        assert (j[c].cast(pl.Float64) - j[f"{c}_gs"]).abs().max() == 0
    for s in gs.STATES:
        assert (j[f"share_{s}"] - j[f"plays_{s}"] / j["plays_gs"]).abs().max() < 1e-12


@db
def test_state_model_for_a_week_uses_only_earlier_team_games(inp):
    r = tr.compute_ratings(inp.plays, n_boot=0)
    tp, _ = gs.walk_forward(gs.pregame_features(inp, r))
    sr = gs.load_state_rows(max_season=2022)
    full, models = gs.state_inputs(sr, tp, inp.games)
    key = (2022, 8)
    before = lambda df: df.filter((pl.col("season") < key[0]) | ((pl.col("season") == key[0]) & (pl.col("week") <= key[1])))
    part, models2 = gs.state_inputs(before(sr), before(tp), inp.games)
    assert np.allclose(models[key].coef, models2[key].coef) and np.allclose(models[key].intercept, models2[key].intercept)
    cols = ["exp_margin", *[f"share_{s}" for s in gs.STATES], "exp_dropbacks", "exp_pass_att", "exp_rush_att"]
    a = full.filter((pl.col("season") == key[0]) & (pl.col("week") == key[1])).sort("game_id", "team").select(cols).to_numpy()
    b = part.filter((pl.col("season") == key[0]) & (pl.col("week") == key[1])).sort("game_id", "team").select(cols).to_numpy()
    assert a.shape == b.shape and np.allclose(a, b, atol=1e-9)
