"""The SIM_THRESHOLD sweep (models/comps_sweep.py): one search serves every threshold, the random pairing keeps the real weights, the rule."""
import numpy as np
import polars as pl
import pytest

from models import comps as C
from models import comps_spec as cs
from models import comps_sweep as SW
from tests.test_comps import _player_target, _pool, _team_target


@pytest.fixture(scope="module")
def pool3():
    return _pool()


def test_restrict_reproduces_a_search_run_at_the_higher_threshold(pool3):
    pool, inp, _ = pool3
    for tg in (_team_target(inp, week=9), _player_target(inp, week=9), _player_target(inp, "targets", week=10)):
        low = pool.search(tg, C.SEARCHES, keep=True, sim_threshold=0.2, min_neff=cs.MIN_NEFF)
        seen = 0
        for t in (0.3, 0.5, 0.7):
            got = SW.restrict(low, t)
            want = pool.search(tg, C.SEARCHES, keep=True, sim_threshold=t, min_neff=cs.MIN_NEFF)
            assert set(got) == set(want)
            for s in want:
                g, w = got[s].summary, want[s].summary
                assert (g["no_match"], g["reason"], g["n_matches"]) == (w["no_match"], w["reason"], w["n_matches"]), (tg, t, s)
                assert g["n_eff"] == pytest.approx(w["n_eff"]) and np.array_equal(g["best_similarity"], w["best_similarity"], equal_nan=True)
                if w["n_matches"]:
                    assert got[s].matches.equals(want[s].matches)
                    seen += 1
        assert seen > 0


def test_random_pairing_keeps_the_real_weights_and_shrinkage():
    rng = np.random.default_rng(0)
    assert SW.random_shift(np.array([1.0, 3.0]), 1.6, np.full(5, 2.0), rng) == pytest.approx(2.0 * 1.6 / (1.6 + cs.SHRINK_K))
    draws = {SW.random_shift(np.array([1.0]), 1.0, np.array([1.0, 2.0, 3.0]), rng) * (1 + cs.SHRINK_K) for _ in range(60)}
    assert draws == {1.0, 2.0, 3.0}                                                      # every candidate can be drawn
    with pytest.raises(ValueError):
        SW.random_shift(np.ones(3), 3.0, np.ones(2), rng)                                # never more matches than the set holds


def test_the_rule_needs_both_random_and_zero_beaten_at_the_lowest_threshold_else_the_default():
    """Tightened rule (2026-10-09): beating random pairing is not enough; the comps must also beat a zero shift, each interval above zero."""
    s = pl.DataFrame({"threshold": [0.4, 0.5, 0.6, 0.7], "improvement": [0.02, 0.03, 0.04, 0.05], "ci_lo": [0.01, 0.001, 0.02, 0.01],
                      "improvement_zero": [-0.01, 0.02, 0.03, 0.02], "ci_lo_zero": [-0.02, -0.001, 0.01, 0.005]})
    assert SW.choose_threshold(s) == 0.6                                                 # 0.4 loses to zero, 0.5's zero interval touches it
    assert SW.choose_threshold(s.with_columns(ci_lo_zero=pl.lit(-1.0))) == SW.DEFAULT_THRESHOLD == 0.70
    assert SW.choose_threshold(s.with_columns(ci_lo=pl.lit(-1.0))) == 0.70
    nulls = {c: pl.lit(None, dtype=pl.Float64) for c in ("improvement", "ci_lo", "improvement_zero", "ci_lo_zero")}
    assert SW.choose_threshold(s.with_columns(**nulls)) == 0.70


def test_summary_reports_the_improvement_over_a_zero_shift():
    rows = pl.DataFrame({"threshold": [0.5] * 4, "season": [2021, 2021, 2022, 2022], "week": [1, 2, 1, 2], "game_id": list("abcd"), "team": ["A"] * 4,
                         "player_id": [None] * 4, "market": ["spread"] * 4, "search": ["S5"] * 4, "side": ["vol"] * 4,
                         "sq_error_real": [1.0, 2.0, 1.0, 2.0], "sq_error_zero": [2.0, 2.0, 2.0, 2.0], "sq_error_random": [1.5, 2.5, 1.5, 2.5]},
                        schema_overrides={"player_id": pl.String})
    s = SW.summarize(rows, (0.5,)).row(0, named=True)
    assert s["improvement"] == pytest.approx(0.5) and s["improvement_zero"] == pytest.approx(0.5) and s["zero"] == pytest.approx(2.0)


def test_each_quantity_counts_once_per_family():
    sm = SW.side_markets()
    assert sm["receiving"] == [("targets", "vol", "targets"), ("rec", "eff", "catch_rate"), ("rec_yds", "eff", "yds_per_rec")]
    assert sm["game"] == [(C.MARKET_FAMILIES["game"][0], "vol", "team_plays"), (C.MARKET_FAMILIES["game"][0], "eff", "pts_per_play")]
    for fam, rows in sm.items():
        assert len({q for _, _, q in rows}) == len(rows)


def test_sweep_rows_are_deterministic_and_score_only_matched_searches(pool3):
    pool, inp, _ = pool3
    tgs = [_team_target(inp, week=9), _player_target(inp, "rush_att", week=9), _player_target(inp, "targets", week=10)]
    rng = np.random.default_rng(5)
    zl, counts = {}, {}
    for q in ("team_plays", "pts_per_play", "rush_att", "ypc", "targets", "catch_rate", "yds_per_rec"):
        zl[q] = {**{(r["game_id"], r["team"]): float(rng.normal()) for r in inp.games.iter_rows(named=True)},
                 **{(r["game_id"], r["player_id"]): float(rng.normal()) for r in inp.player.iter_rows(named=True)}}
        counts[q] = {k: float(rng.integers(1, 20)) for k in zl[q]}
    th = (0.2, 0.4, 0.6)
    a, ra = SW.sweep_rows(pool, tgs, zl, counts, thresholds=th, min_neff=1.0, n_draws=3)
    b, rb = SW.sweep_rows(pool, tgs, zl, counts, thresholds=th, min_neff=1.0, n_draws=3)
    assert a.equals(b) and ra.equals(rb) and a.height > 0
    assert (a["sq_error_real"] >= 0).all() and (a["sq_error_random"] >= 0).all()
    n = a.group_by("threshold").len().sort("threshold")["len"].to_list()
    assert n == sorted(n, reverse=True)                                                  # a higher threshold scores fewer searches
    assert a.select("threshold", "game_id", "team", "player_id", "search", "quantity").is_unique().all()
    summ = SW.summarize(a, th)
    assert summ["n_rows"].to_list() == n and SW.choose_threshold(summ) in (*th, SW.DEFAULT_THRESHOLD)


def test_the_unit_diagnostic_needs_both_improvements_with_bonferroni_intervals():
    cells = SW.diagnostic_cells()
    assert len(cells) == len(set(cells)) and ("S4:run_offense*run_defense", "vol", "rush_att") in cells
    assert not any(u.startswith("S3") and q in ("team_plays", "pts_per_play") for u, _, q in cells)      # S3: player markets only
    rng = np.random.default_rng(1)
    rows = []
    for s in (2021, 2022, 2023, 2024):
        for w in range(1, 18):
            for i in range(6):
                z = float(rng.normal())
                # a unit whose shift tracks z (beats zero and random), and one that is pure noise
                rows.append(dict(season=s, week=w, game_id=f"{s}{w}{i}", team="A", player_id=f"P{i}", market="rush_att", search="S4:run_offense*run_defense",
                                 side="vol", quantity="rush_att", sq_error_real=(0.6 * z - z) ** 2, sq_error_zero=z ** 2, sq_error_random=z ** 2 + 0.05))
                rows.append(dict(season=s, week=w, game_id=f"{s}{w}{i}", team="A", player_id=f"P{i}", market="rush_att", search="S1:run_defense",
                                 side="vol", quantity="rush_att", sq_error_real=z ** 2 + 0.3 * float(rng.normal()), sq_error_zero=z ** 2,
                                 sq_error_random=z ** 2 + 0.3 * float(rng.normal())))
    rates = pl.DataFrame([dict(search=r["search"], side="vol", quantity="rush_att", market="rush_att", matched=True) for r in rows])
    d = SW.unit_diagnostic(pl.DataFrame(rows), rates)
    cell = d.filter((pl.col("side") == "vol") & (pl.col("quantity") == "rush_att"))
    lab = dict(zip(cell["search"].to_list(), cell["label"].to_list()))
    assert lab["S4:run_offense*run_defense"] == "WORKS" and lab["S1:run_defense"] in ("NO", "EDGE")
    assert d["interval_level"][0] == pytest.approx(1 - 0.05 / len(cells)) and d.height == len(cells)
    assert d.filter(pl.col("n_rows") == 0)["label"].unique().to_list() == ["NO"]
