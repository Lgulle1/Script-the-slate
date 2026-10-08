import numpy as np
import polars as pl
import pytest

import config
from eval import backtest as bt
from eval import baselines as bl
from features import lineups as lu
from features import weights
from features import volume_features as vf
from models import volume as vm

PEN = config.CONTINUITY_PENALTIES["rush_att"]


def _continuity(lin_past, tgt, slot_p, slot_t, fam_p, fam_t, team_changed):
    return weights.continuity_weight(weights.continuity_flags(lin_past, tgt, slot_p, slot_t, fam_p, fam_t, team_changed), "rush_att")


def test_continuity_weights_apply_only_changed_factors():
    ids = lambda **k: {c: np.array(k.get(c, [5, 5, 5])) for c in lu.COMPONENTS}  # noqa: E731
    lin_past = ids(OL=[5, 9, -1])                      # 2nd game: OL differs; 3rd: chart missing
    tgt = {c: 5 for c in lu.COMPONENTS}
    same = np.array([0, 0, 0])
    w = _continuity(lin_past, tgt, np.array([1, 1, 1]), 1, np.array([1, 1, 1]), 1, same.astype(bool))
    assert w[0] == 1.0 and w[2] == 1.0                 # unchanged; missing chart is never a change
    assert w[1] == pytest.approx(PEN["OL"])
    # role change and a team change
    w = _continuity(ids(), tgt, np.array([2, 1, 1]), 1, np.array([1, 1, 1]), 1, np.array([False, False, True]))
    assert w[0] == pytest.approx(PEN["role"])
    assert w[2] == pytest.approx(np.prod([PEN[f] for f in config.CONTINUITY_FACTORS]))  # team change flags everything


def test_team_volume_is_float_and_rates_are_fractions():
    if not config.RAW_DUCKDB_PATH.exists():
        pytest.skip("no raw db")
    tv = vf.build_team_volume(max_season=2024)
    assert tv.schema["plays"] == pl.Float64          # DuckDB sums are DECIMAL; integer-dividing them gave 0/1 rates
    pr = tv["dropbacks"] / tv["plays"]
    assert 0.5 < pr.mean() < 0.65 and 0.0 < pr.min() and pr.max() < 1.0  # real extremes exist (NE threw 3 times in the 2021 wind game)
    assert 55 < tv["plays"].mean() < 70


@pytest.fixture(scope="module")
def real():
    if not config.RAW_DUCKDB_PATH.exists():
        pytest.skip("no raw db")
    data = bt.load_backtest_data()
    tv = vf.build_team_volume(max_season=2024)
    ln = lu.build_lineups(max_season=2024)
    return data, tv, ln, vf.build_feature_tables(data.player_log, data.team_log, tv, ln)


def test_recency_feature_matches_phase2_baseline(real):
    """rec_<stat> must equal the phase-2 recency baseline for the same target and cutoff."""
    data, _, _, ft = real
    cutoffs = {(s, w): c for s, w, c in data.weeks}
    df = ft.players["rush_att"].filter((pl.col("season") == 2023) & (pl.col("n_hist") > 5)).sample(25, seed=1)
    for r in df.iter_rows(named=True):
        c = cutoffs[(r["season"], r["week"])]
        hist = data.player_log.filter(pl.col("gameday") < c)
        fam = {v: k for k, v in vf.FAMILY_CODE.items()}[r["family"]]
        t = bl.PlayerTarget(r["player_id"], r["gameday"], r["season"], r["team_game_num"], r["opponent"], fam, r["slot"],
                            r["team"], r["week"])
        assert r["rec_carries"] == pytest.approx(bl.player_recency(hist, t, c)["rush_att"], rel=1e-9)


def test_features_never_see_the_future(real):
    """Scramble every stat, lineup and score from week-X onward: features of rows at or before X must not move."""
    data, tv, ln, ft = real
    D = next(c for s, w, c in data.weeks if (s, w) == (2023, 10))
    late_games = set(data.team_log.filter(pl.col("gameday") >= D)["game_id"])
    wk = {(s, w): c for s, w, c in data.weeks}
    stat_cols = list(bl.PLAYER_MARKETS.values())
    pl2 = data.player_log.with_columns([pl.when(pl.col("gameday") >= D).then(pl.col(c) * 3 + 11).otherwise(pl.col(c)).alias(c)
                                        for c in stat_cols])
    tl2 = data.team_log.with_columns([pl.when(pl.col("gameday") >= D).then(pl.col(c) * 2 + 7).otherwise(pl.col(c)).alias(c)
                                      for c in ("pf", "pa")])
    tv2 = tv.with_columns([pl.when(pl.col("game_id").is_in(list(late_games))).then(pl.col(c) * 2 + 5).otherwise(pl.col(c)).alias(c)
                           for c in ("rush_att", "pass_att", "dropbacks", "plays")])
    late_wk = [(s, w) for (s, w), c in wk.items() if c >= D]
    ln2 = ln.with_columns([pl.when(pl.struct("season", "week").map_elements(lambda r: (r["season"], r["week"]) in late_wk,
                                                                          return_dtype=pl.Boolean))
                           .then(pl.col(c) + 1).otherwise(pl.col(c)).alias(c) for c in lu.COMPONENTS])
    ft2 = vf.build_feature_tables(pl2, tl2, tv2, ln2)
    for q in vf.PLAYER_SPECS:
        a, b = ft.players[q], ft2.players[q]
        cols = [c for c in vm.feature_columns(a)]
        early = lambda df: df.filter(pl.col("gameday") < D).sort("player_id", "game_id")  # noqa: E731
        ea, eb = early(a), early(b)
        assert ea.height == eb.height > 300   # the QB-rushing table is small (a few hundred games)
        for c in cols:
            x, y = ea[c].to_numpy().astype(float), eb[c].to_numpy().astype(float)
            assert np.allclose(x, y, equal_nan=True), f"{q}.{c} changed when only the future was altered"
    a, b = ft.teams.filter(pl.col("gameday") < D).sort("team", "gameday"), ft2.teams.filter(pl.col("gameday") < D).sort("team", "gameday")
    for c in vm.feature_columns(a):
        assert np.allclose(a[c].to_numpy().astype(float), b[c].to_numpy().astype(float), equal_nan=True), c


def test_volume_predictors_through_the_harness_are_deterministic_and_capped(real):
    from dataclasses import replace
    data, tv, ln, ft = real
    d = replace(data, game_extras=vm.plays_extras(data.team_log, tv))
    pp, gp = vm.volume_predictors(ft)
    a = bt.walk_forward("v", d, pp, gp, seasons=[2022])
    b = bt.walk_forward("v", d, pp, gp, seasons=[2022])
    assert a.equals(b) and a.height > 5000
    assert set(a["market"].unique()) == {"pass_att", "rush_att", "targets", "qb_rush_att", "plays_home", "plays_away"}
    assert a["season"].max() == 2022
    assert a["prediction"].min() >= 0
    with pytest.raises(ValueError, match="locked"):
        bt.walk_forward("v", d, pp, gp, seasons=[2025])
    assert d.player_log["season"].max() == 2024 and ft.teams["season"].max() == 2024
