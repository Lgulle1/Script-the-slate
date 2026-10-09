import math

import numpy as np
import pytest

import config
from eval import compare as cp
from models import combine as cb


def test_combine_player_products_and_missing_components():
    v = {"pass_att": 30.0, "rush_att": 12.0, "targets": 6.0}
    e = {"comp_pct": 0.6, "yds_per_cmp": 11.0, "ypc": 4.5, "catch_pct": 0.7, "yds_per_rec": 10.0}
    c = cb.combine_player(v, e)
    assert c["pass_cmp"] == pytest.approx(18.0) and c["pass_yds"] == pytest.approx(198.0)
    assert c["rush_yds"] == pytest.approx(54.0)
    assert c["rec"] == pytest.approx(4.2) and c["rec_yds"] == pytest.approx(42.0)
    assert (c["pass_att"], c["rush_att"], c["targets"]) == (30.0, 12.0, 6.0)
    part = cb.combine_player(v, {"comp_pct": 0.6})   # yards-per-completion missing: no partial product
    assert part["pass_cmp"] == pytest.approx(18.0) and part["pass_yds"] is None and part["rush_yds"] is None
    assert cb.combine_player({}, e)["rec_yds"] is None


def test_combine_game():
    g = cb.combine_game({"plays_home": 64.0, "plays_away": 60.0}, {"ppp_home": 0.4, "ppp_away": 0.3})
    assert g["total"] == pytest.approx(25.6 + 18.0) and g["spread"] == pytest.approx(7.6)
    assert g["moneyline"] == pytest.approx(0.5 * (1 + math.erf(7.6 / config.GAME_MARGIN_SD / math.sqrt(2))))
    assert cb.combine_game({"plays_home": 64.0, "plays_away": None}, {"ppp_home": 0.4, "ppp_away": 0.3}) == \
        {"spread": None, "moneyline": None, "total": None}
    assert set(cb.combine_player({}, {})) == set(cb.PLAYER_MARKETS) and set(cb.GAME_MARKETS) >= {"spread", "total"}


def test_bootstrap_is_deterministic_and_centred():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 3.0, 4000)
    cl = np.repeat(np.arange(80), 50)
    a, b = cp.cluster_bootstrap(d, cl, n_boot=2000), cp.cluster_bootstrap(d, cl, n_boot=2000)
    assert a == b
    lo, hi = a
    assert lo < d.mean() < hi and 0.4 < (hi - lo) / (2 * 1.96 * 3.0 / math.sqrt(4000)) < 1.6   # about the analytic width


def test_ci_excludes_zero_for_a_real_gain_and_covers_zero_for_noise():
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(60), 40)
    real = rng.normal(0.8, 2.0, cl.size)
    assert cp.cluster_bootstrap(real, cl, n_boot=2000)[0] > 0 and cp.iid_bootstrap(real, n_boot=500)[0] > 0
    noise = rng.normal(0.0, 2.0, cl.size)
    lo, hi = cp.cluster_bootstrap(noise, cl, n_boot=2000)
    assert lo < 0 < hi


def test_cluster_interval_is_wider_than_iid_when_clusters_are_correlated():
    rng = np.random.default_rng(2)
    cl = np.repeat(np.arange(60), 40)
    d = rng.normal(0, 1, 60)[cl] + rng.normal(0, 1, cl.size)   # shared slate effect
    cw, iw = (lambda t: t[1] - t[0])(cp.cluster_bootstrap(d, cl, n_boot=2000)), (lambda t: t[1] - t[0])(cp.iid_bootstrap(d, n_boot=500))
    assert cw > 1.5 * iw


def test_verdicts():
    assert cp.verdict(0.5, 0.1, 0.9, 3) == "CLEARS"
    assert cp.verdict(0.62, -0.02, 1.27, 4) == "EDGE"      # the Oct 6 RB rushing-yards result
    assert cp.verdict(0.5, 0.1, 0.9, 1) == "EDGE"          # positive pooled gain but only one season
    assert cp.verdict(-0.2, -0.6, 0.1, 1) == "NO" and cp.verdict(-0.5, -0.9, -0.1, 0) == "NO"


def test_one_layer_weight_rule_keeps_a_layer_only_on_clears():
    """Decision of 2026-10-09: every Phase 4 layer keeps weight 1 in a market only on CLEARS (interval excluding zero, >= 2 seasons); EDGE -> 0."""
    from eval import compare
    assert compare.layer_weight(compare.verdict(0.02, 0.001, 0.04, 2)) == 1.0
    assert compare.layer_weight(compare.verdict(0.02, -0.001, 0.04, 4)) == 0.0          # EDGE: wins seasons but the interval touches zero
    assert compare.layer_weight(compare.verdict(0.02, 0.001, 0.04, 1)) == 0.0           # one season only
    assert compare.layer_weight("NO") == 0.0
