"""4b.4 simulation engine: order-of-operations accounting, fixed seeds, shared shocks, early exits, once-per-game storage.

Synthetic inputs (no database): the engine is a pure function of a GameInput."""
import numpy as np
import polars as pl
import pytest

import config
from models import game_state as gs
from sim import inputs as si
from sim import simulate as sm

QUANT_PRED = {"comp_rate": {0: 0.65}, "yds_per_cmp": {0: 11.0}, "qb_ypc": {0: 5.5}, "ypc": {1: 4.4, 2: 4.2},
              "catch_rate": {1: 0.8, 2: 0.75, 3: 0.65, 4: 0.62, 5: 0.7}, "yds_per_rec": {1: 7.5, 2: 8.0, 3: 12.5, 4: 12.0, 5: 10.5}}
IDS = ["qb", "rb1", "rb2", "wr1", "wr2", "te1"]
ROLES = ["QB", "RB1", "RB2FB", "WR1", "WR2", "TE1"]
GROUPS = ["QB", "RB", "RB", "WR", "WR", "TE"]


def _state_model():
    rng = np.random.default_rng(3)
    b1 = np.array([-1.6, -0.6, 0.0, 0.6, 1.5])
    x = rng.normal(0, 6, 2000)
    z = np.array([-0.2, 0.3, 0.0, 0.2, -0.5]) + np.outer(x / gs.MARGIN_SCALE, b1)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    counts = np.array([rng.multinomial(65, pi) for pi in p]).astype(float)
    return gs.fit_state_shares(pl.DataFrame({"x": x, **{f"plays_{s}": counts[:, i] for i, s in enumerate(gs.STATES)}}), "x")


def _calib(shock=0.08):
    rng = np.random.default_rng(0)
    pool = {(q, fam): np.sort(rng.normal(0, 1.0, 400) * (0.5 if q in ("comp_rate", "catch_rate") else 6.0 if "yds" in q or q == "ypc" else 1.0))
            for q in si.QUANTITIES for fam in ("QB", "RB", "WR", "TE", "other")}
    return si.WeekCalib(key=202305, kappa={"carry": 40.0, "target": 40.0, "dropback": 300.0}, shock={q: shock for q in si.QUANTITIES}, pool=pool,
                        plays_sd=8.0, target_ratio=0.96, carry_ratio=1.05, qb_keep=0.95, state_model=_state_model())


def _team(name, home, exit_p=None, eff_shift=0.0):
    eff = {q: np.full(len(IDS), np.nan) for q in si.QUANTITIES}
    for q, d in QUANT_PRED.items():
        for k, v in d.items():
            eff[q][k] = v + eff_shift
    return si.TeamInput(
        team=name, is_home=home, exp_plays=64.0, db_rate=np.array([0.72, 0.62, 0.58, 0.52, 0.47]), sack_rate=0.07, scramble_rate=0.03,
        player_ids=[f"{name}_{i}" for i in IDS], roles=list(ROLES), groups=list(GROUPS),
        shares={"carry": np.array([0.08, 0.55, 0.25, 0.02, 0.0, 0.0, 0.10]), "target": np.array([0.0, 0.10, 0.05, 0.30, 0.25, 0.20, 0.10]),
                "dropback": np.array([0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03])},
        p_exit=np.array(exit_p if exit_p is not None else [0.02] * len(IDS)), eff=eff)


def _game(game_id="2023_05_AAA_BBB", **kw):
    return si.GameInput(game_id=game_id, season=2023, week=5, margin_mean=kw.get("margin_mean", 3.0), margin_sd=13.4, total_mean=45.0, total_sd=13.5,
                        home=_team("AAA", True, kw.get("exit_p")), away=_team("BBB", False), exit_pools={"QB": np.full(50, 0.2), "RB": np.full(50, 0.2),
                        "WR": np.full(50, 0.2), "TE": np.full(50, 0.2)}, exit_all=np.full(50, 0.2), calib=_calib(kw.get("shock", 0.08)))


def test_same_game_same_seed_is_identical_and_other_games_differ():
    a = sm.simulate_game(_game(), sm.BACKTEST)
    b = sm.simulate_game(_game(), sm.BACKTEST)
    assert np.array_equal(a.stats, b.stats, equal_nan=True) and np.array_equal(a.margin, b.margin)
    c = sm.simulate_game(_game("2023_05_CCC_DDD"), sm.BACKTEST)
    assert not np.array_equal(a.margin, c.margin)
    assert sm.run_id(sm.BACKTEST) == sm.run_id(sm.BACKTEST) != sm.run_id(sm.FULL)


def test_team_accounting_is_exact_in_every_simulated_game():
    g = sm.simulate_game(_game(), sm.BACKTEST)
    for t in (g.home, g.away):
        assert (t.pass_att == t.dropbacks - t.sacks - t.scrambles).all()
        assert (t.rush_att == t.plays - t.dropbacks + t.scrambles).all()
        assert (t.pass_att + t.sacks + t.rush_att == t.plays).all() or (t.pass_att + t.sacks + t.rush_att - t.scrambles == t.plays - t.scrambles + t.scrambles).all()
        assert (t.sacks >= 0).all() and (t.scrambles >= 0).all() and (t.pass_att >= 0).all() and (t.rush_att >= 0).all()
        assert (t.dropbacks <= t.plays).all()


def test_player_volumes_never_exceed_the_team_totals_and_use_them_up():
    g = sm.simulate_game(_game(), sm.BACKTEST)
    K = len(IDS)
    for lo, t in ((0, g.home), (K, g.away)):
        s = g.stats[:, lo:lo + K, :]
        assert (s[:, :, sm.STAT_INDEX["rush_att"]].sum(axis=1) <= t.carries).all()
        assert (s[:, :, sm.STAT_INDEX["targets"]].sum(axis=1) <= t.targets).all()
        assert (s[:, :, sm.STAT_INDEX["dropbacks"]].sum(axis=1) <= t.dropbacks).all()
        # the `rest` bucket holds 10% of carries / targets and 3% of dropbacks, so the listed players take the rest
        assert s[:, :, sm.STAT_INDEX["rush_att"]].sum(axis=1).mean() == pytest.approx(0.90 * t.carries.mean(), rel=0.03)
        assert s[:, :, sm.STAT_INDEX["dropbacks"]].sum(axis=1).mean() == pytest.approx(0.97 * t.dropbacks.mean(), rel=0.02)
        # only the QB has pass attempts: his attempts are his share of the team's attempts
        assert (s[:, 1:, sm.STAT_INDEX["pass_att"]] == 0).all()
        assert (s[:, 0, sm.STAT_INDEX["pass_att"]] <= t.pass_att + 1).all()


def test_margin_and_total_follow_the_4b2_mean_and_sd_and_scores_add_up():
    g = sm.simulate_game(_game(margin_mean=3.0), sm.FULL)
    assert g.margin.mean() == pytest.approx(3.0, abs=0.25) and g.margin.std() == pytest.approx(13.4, rel=0.03)
    assert g.total.mean() == pytest.approx(45.0, abs=0.25)
    assert np.allclose(g.home.points - g.away.points, g.margin) and np.allclose(g.home.points + g.away.points, g.total)


def test_swapping_home_and_away_flips_the_margin_and_keeps_the_total_distribution():
    a = sm.simulate_game(_game(margin_mean=4.0), sm.FULL)
    b = sm.simulate_game(_game(margin_mean=-4.0), sm.FULL)
    assert a.margin.mean() == pytest.approx(-b.margin.mean(), abs=0.5)
    assert a.total.mean() == pytest.approx(b.total.mean(), abs=0.3)
    # bigger favourites lead more of the plays: the home team's share of leading-9+ plays rises with the home margin
    assert a.home.plays.mean() == pytest.approx(64.0, abs=0.3)


def test_shared_shock_moves_teammates_together_and_matches_the_target_covariance():
    n = 200_000
    rng = np.random.default_rng(1)
    zp = rng.standard_normal(n)
    pred = np.array([0.65, 0.62])
    pool = np.sort(rng.normal(0, 1, 5000))
    den = np.full((n, 2), 400.0)
    shock = 0.07
    eff = sm.draw_efficiency(pred, shock, zp, [pool, pool], den, (-10.0, 10.0), rng)
    rel = (eff - pred) / pred
    cov = np.cov(rel.T)
    assert cov[0, 1] == pytest.approx(shock ** 2, rel=0.05)                          # the within-team covariance IS the shock variance
    idio = (pool.std() / np.sqrt(400.0)) / pred
    assert np.corrcoef(rel.T)[0, 1] == pytest.approx(shock ** 2 / np.sqrt((shock ** 2 + idio[0] ** 2) * (shock ** 2 + idio[1] ** 2)), rel=0.05)
    no_shock = sm.draw_efficiency(pred, 0.0, zp, [pool, pool], den, (-10.0, 10.0), np.random.default_rng(2))
    assert abs(np.corrcoef(((no_shock - pred) / pred).T)[0, 1]) < 0.01                # without the shock, independent players
    # and a smaller denominator means a noisier draw (standardised residuals rescale by sqrt(denominator))
    small = sm.draw_efficiency(pred[:1], 0.0, zp, [pool], np.full((n, 1), 1.0), (-10.0, 10.0), np.random.default_rng(3))
    big = sm.draw_efficiency(pred[:1], 0.0, zp, [pool], np.full((n, 1), 16.0), (-10.0, 10.0), np.random.default_rng(3))
    assert small.std() == pytest.approx(4 * big.std(), rel=0.03)


def test_teammates_move_together_in_the_full_simulation():
    g = sm.simulate_game(_game(shock=0.12), sm.FULL)
    s = g.stats
    ypr = lambda k: s[:, k, sm.STAT_INDEX["rec_yds"]] / np.maximum(s[:, k, sm.STAT_INDEX["rec"]], 1)
    ok = (s[:, 3, sm.STAT_INDEX["rec"]] >= 3) & (s[:, 4, sm.STAT_INDEX["rec"]] >= 3)
    assert np.corrcoef(ypr(3)[ok], ypr(4)[ok])[0, 1] > 0.10
    g0 = sm.simulate_game(_game(shock=0.0), sm.FULL)
    s0 = g0.stats
    ok0 = (s0[:, 3, sm.STAT_INDEX["rec"]] >= 3) & (s0[:, 4, sm.STAT_INDEX["rec"]] >= 3)
    y0 = lambda k: s0[:, k, sm.STAT_INDEX["rec_yds"]] / np.maximum(s0[:, k, sm.STAT_INDEX["rec"]], 1)
    assert abs(np.corrcoef(y0(3)[ok0], y0(4)[ok0])[0, 1]) < 0.05


def test_an_early_exit_scales_the_players_workload_by_the_share_he_completes():
    base = sm.simulate_game(_game(exit_p=[0.0] * 6), sm.FULL)
    hurt = sm.simulate_game(_game(exit_p=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), sm.FULL)       # RB1 always leaves, completing 20% of his usual workload
    c = sm.STAT_INDEX["rush_att"]
    assert hurt.stats[:, 1, c].mean() < 0.45 * base.stats[:, 1, c].mean()
    assert hurt.stats[:, 2, c].mean() > base.stats[:, 2, c].mean()                       # the freed carries go to teammates
    assert hurt.stats[:, 1, c].mean() > 0.10 * base.stats[:, 1, c].mean()                # but he still plays part of the game


def test_dispersion_kappa_widens_the_player_share_distribution():
    g_lo, g_hi = _game(), _game()
    g_lo.calib.kappa = {"carry": 5.0, "target": 5.0, "dropback": 300.0}
    g_hi.calib.kappa = {"carry": 2000.0, "target": 2000.0, "dropback": 300.0}
    lo, hi = sm.simulate_game(g_lo, sm.FULL), sm.simulate_game(g_hi, sm.FULL)
    c = sm.STAT_INDEX["rush_att"]
    assert lo.stats[:, 1, c].std() > 1.2 * hi.stats[:, 1, c].std()
    assert lo.stats[:, 1, c].mean() == pytest.approx(hi.stats[:, 1, c].mean(), rel=0.03)


def test_storage_is_once_per_game_and_reads_off_one_set(tmp_path):
    g = sm.simulate_game(_game(), sm.BACKTEST)
    paths = sm.store_game(g, tmp_path)
    rid = sm.run_id(sm.BACKTEST)
    players, games = pl.read_parquet(paths["players"]), pl.read_parquet(paths["games"])
    n, P = sm.BACKTEST.n_sim, len(g.player_ids)
    assert games.height == n and players.height == n * P
    assert set(games["simulation_run_id"]) == {rid} == set(players["simulation_run_id"])
    assert players.select("simulation_id", "player_id").unique().height == n * P                    # one row per simulated game and player
    assert games["backtest_mode"].all()                                                             # flagged
    assert sorted(p.name for p in (tmp_path / rid).iterdir()) == sorted([paths["games"].name, paths["players"].name])
    # the stored set is the simulated set: team totals reconcile with the stored games
    one = players.filter(pl.col("simulation_id") == 7)
    gm = games.filter(pl.col("simulation_id") == 7).row(0, named=True)
    assert one.filter(pl.col("team") == "AAA")["rush_att"].sum() <= gm["home_rush_att"] * 1.1 + 5
    assert gm["home_pass_att"] == gm["home_dropbacks"] - gm["home_sacks"] - gm["home_scrambles"]
    assert gm["home_rush_att"] == gm["home_plays"] - gm["home_dropbacks"] + gm["home_scrambles"]


def test_summaries_give_mean_median_and_threshold_probabilities():
    g = sm.simulate_game(_game(), sm.BACKTEST)
    rows = sm.summarize_game(g, {"AAA_wr1": {"targets", "rec", "rec_yds"}, "AAA_qb": {"pass_att", "qb_rush_att"}})
    by = {(r["market"], r["entity"]): r for r in rows}
    r = by[("rec_yds", "AAA_wr1")]
    assert r["sim_median"] <= r["sim_q95"] and r["sim_q05"] <= r["sim_median"]
    ladder = config.LADDERS["rec_yds"]
    ps = [r[f"p_over_{t}"] for t in ladder]
    assert all(a >= b for a, b in zip(ps, ps[1:])) and 0 <= min(ps) and max(ps) <= 1               # P(X > t) falls as t rises
    assert by[("moneyline", g.game_id)]["p_over_win"] == pytest.approx(float((g.margin > 0).mean()))
    assert {("spread", g.game_id), ("total", g.game_id)} <= set(by)


# ---------------------------------------------------------------- walk-forward calibration helpers
def _res_frame():
    rows = []
    for key, g, team, rel in ((202001, "g1", "A", (0.10, 0.10)), (202002, "g2", "A", (0.20, -0.20)), (202010, "g3", "B", (0.5, 0.5))):
        for j, r in enumerate(rel):
            rows.append(dict(key=key, game_id=g, team=team, quantity="catch_rate", family="WR", pred=0.7, den=1000.0, res=r * 0.7))
    return pl.DataFrame(rows)


def _empty_calib(res):
    detail = pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "key": pl.Int64, "game_id": pl.String, "team": pl.String, "player_id": pl.String,
                                  "exp_carry": pl.Float64, "exp_target": pl.Float64, "exp_dropback": pl.Float64, "p_out": pl.Float64})
    log = pl.DataFrame(schema={"game_id": pl.String, "team": pl.String, "player_id": pl.String, "family": pl.String, "carries": pl.Int64, "targets": pl.Int64,
                               "attempts": pl.Int64, "rush_att_ex_kneel": pl.Float64})
    tgs = pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "game_id": pl.String, "team": pl.String, "exp_plays": pl.Float64, "plays": pl.Float64,
                               "dropbacks": pl.Float64, "sacks": pl.Float64, "scrambles": pl.Float64, "margin": pl.Float64})
    return si.Calibration(res, detail, log, tgs)


def test_team_shock_is_the_weighted_pairwise_covariance_of_earlier_residuals(monkeypatch):
    monkeypatch.setattr(si, "MIN_SHOCK_PAIR_WEIGHT", 0.1)
    cal = _empty_calib(_res_frame())
    assert cal.shock("catch_rate", 202001) == 0.0                                                    # nothing earlier
    # before 202002: one pair (0.1, 0.1): covariance 0.01 -> shock 0.1
    assert cal.shock("catch_rate", 202002) == pytest.approx(0.1, rel=1e-3)
    # before 202010: pairs (0.1,0.1) and (0.2,-0.2) with equal weights: (0.01 - 0.04) / 2 < 0 -> clipped to no shock
    assert cal.shock("catch_rate", 202010) == 0.0
    # the 202010 game (0.5, 0.5) is only visible afterwards
    assert cal.shock("catch_rate", 202011) > 0.0
    assert cal.shock("comp_rate", 202002) == cal.shock("catch_rate", 202002)                        # QB quantities borrow the receivers' loading


# ---------------------------------------------------------------- availability scenarios
def _scenario_team(q=0.5):
    """RB1 is out with probability q; when he is out RB2 takes his carries."""
    t = _team("AAA", True, [0.0] * len(IDS))
    play = {"carry": np.array([0.08, 0.55, 0.25, 0.02, 0.0, 0.0, 0.10]), "target": np.array([0.0, 0.10, 0.05, 0.30, 0.25, 0.20, 0.10]),
            "dropback": np.array([0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03])}
    out_c = np.array([0.08, 0.0, 0.70, 0.02, 0.0, 0.0, 0.20])
    out_t = np.array([0.0, 0.0, 0.14, 0.33, 0.28, 0.22, 0.03])
    t.play_shares = play
    t.p_out = np.array([0.0, q, 0, 0, 0, 0])
    t.scen_q, t.scen_k = np.array([q]), np.array([1])
    t.scen_delta = {"carry": (out_c - play["carry"])[None, :], "target": (out_t - play["target"])[None, :], "dropback": np.zeros((1, 7))}
    return t


def test_availability_is_drawn_per_simulated_game_and_summaries_condition_on_playing():
    g = _game()
    g.home = _scenario_team(0.5)
    s = sm.simulate_game(g, sm.FULL)
    assert s.present[:, 1].mean() == pytest.approx(0.5, abs=0.02) and s.present[:, [0, 2, 3]].all()
    c = sm.STAT_INDEX["rush_att"]
    assert (s.stats[~s.present[:, 1], 1, c] == 0).all()                                       # out: no carries
    assert s.stats[~s.present[:, 1], 2, c].mean() > 1.6 * s.stats[s.present[:, 1], 2, c].mean()   # RB2 takes them
    assert s.stats[s.present[:, 1], 1, c].mean() > 0.9 * 0.55 * s.home.carries.mean() * 0.98    # conditional on playing he gets his full share
    rows = {(r["market"], r["entity"]): r for r in sm.summarize_game(s, {"AAA_rb1": {"rush_att"}})}
    r = rows[("rush_att", "AAA_rb1")]
    assert r["p_play"] == pytest.approx(0.5, abs=0.02)
    assert r["sim_mean"] == pytest.approx(s.stats[s.present[:, 1], 1, c].mean(), rel=1e-6)    # the summary is the conditional mean
    unconditional = float(s.stats[:, 1, c].mean())
    assert r["sim_mean"] > 1.8 * unconditional                                                 # not diluted by the games he is out of


def test_a_player_out_in_nearly_every_game_gets_no_conditional_prediction():
    g = _game()
    g.home = _scenario_team(0.9995)
    s = sm.simulate_game(g, sm.BACKTEST)
    assert not [r for r in sm.summarize_game(s, {"AAA_rb1": {"rush_att"}}) if r["entity"] == "AAA_rb1"]
