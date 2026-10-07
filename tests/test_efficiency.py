from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest

import config
from eval import backtest as bt
from eval import baselines as bl
from features import lineups as lu
from features import volume_features as vf
from models import efficiency as ef
from models import volume as vm


def test_allowed_points_are_what_the_defence_allowed_not_what_its_offence_scored():
    """Regression: def_pts once held the defence's OWN scoring. Team B scores 10 and allows 30."""
    rows = []
    for wk, day in ((1, date(2024, 9, 8)), (2, date(2024, 9, 15))):
        rows += [dict(game_id=f"g{wk}", season=2024, week=wk, gameday=day, team="A", opponent="B", pf=30, pa=10,
                      is_home=True, team_game_num=wk),
                 dict(game_id=f"g{wk}", season=2024, week=wk, gameday=day, team="B", opponent="A", pf=10, pa=30,
                      is_home=False, team_game_num=wk)]
    tl = pl.DataFrame(rows)
    tv = tl.select("game_id", "team", rush_att=pl.lit(25.0), pass_att=pl.lit(35.0), dropbacks=pl.lit(37.0), plays=pl.lit(62.0),
                   rush_yds=pl.lit(100.0), pass_yds=pl.lit(250.0), completions=pl.lit(22.0))
    wk = {(2024, 1): date(2024, 9, 8), (2024, 2): date(2024, 9, 15)}
    side = vf.build_team_side_features(tl, tv, wk, vf._Clock([2024]))
    b = side.filter((pl.col("team") == "B") & (pl.col("week") == 2)).row(0, named=True)
    assert b["off_pts"] == pytest.approx(10.0) and b["def_pts"] == pytest.approx(30.0)
    a = side.filter((pl.col("team") == "A") & (pl.col("week") == 2)).row(0, named=True)
    assert a["off_pts"] == pytest.approx(30.0) and a["def_pts"] == pytest.approx(10.0)


@pytest.fixture(scope="module")
def real():
    if not config.RAW_DUCKDB_PATH.exists():
        pytest.skip("no raw db")
    data = bt.load_backtest_data()
    tv = vf.build_team_volume(max_season=2024)
    ln = lu.build_lineups(max_season=2024)
    return data, tv, ln, ef.build_efficiency_tables(data.player_log, data.team_log, tv, ln)


def test_labels_are_ratios_and_zero_denominators_are_untrainable(real):
    _, _, _, et = real
    for q, spec in ef.EFFICIENCY_SPECS.items():
        df = et.players[q]
        ok = df.filter(pl.col("den") > 0)
        assert ok["label"].is_nan().sum() == 0
        assert ok["label"].min() >= (0 if q in ("comp_pct", "catch_pct") else -15)  # yards per carry can be negative
        zero = df.filter(pl.col("den") == 0)
        assert zero.height == 0 or zero["label"].is_nan().all()
    assert et.players["comp_pct"]["label"].drop_nans().max() <= 1.0
    assert 0.25 < et.teams["label"].mean() < 0.45  # points per play


def test_pooled_rate_feature_matches_a_manual_recomputation(real):
    """rec_ypc = sum(w * yards) / sum(w * carries) with the phase-2 recency weights."""
    data, _, _, et = real
    cutoffs = {(s, w): c for s, w, c in data.weeks}
    rows = et.players["ypc"].filter((pl.col("season") == 2023) & (pl.col("n_hist") > 5)).sample(20, seed=3)
    for r in rows.iter_rows(named=True):
        c = cutoffs[(r["season"], r["week"])]
        hist = data.player_log.filter((pl.col("player_id") == r["player_id"]) & (pl.col("gameday") < c))
        fam = {v: k for k, v in vf.FAMILY_CODE.items()}[r["family"]]
        t = bl.PlayerTarget(r["player_id"], r["gameday"], r["season"], r["team_game_num"], r["opponent"], fam, r["slot"],
                            r["team"], r["week"])
        w = np.array([bl.recency_weight(bl._games_back(s, g, tm, wk, t)) for s, g, tm, wk in
                      zip(hist["season"], hist["team_game_num"], hist["team"], hist["week"])])
        yds, car = hist["rushing_yards"].to_numpy().astype(float), hist["carries"].to_numpy().astype(float)
        want = (w * yds).sum() / (w * car).sum() if (w * car).sum() > 0 else np.nan
        assert r["rec_ypc"] == pytest.approx(want, nan_ok=True, rel=1e-9)
        assert r["rec_ypc_mass"] == pytest.approx((w * car).sum(), rel=1e-9)


def test_efficiency_features_never_see_the_future(real):
    data, tv, ln, et = real
    D = next(c for s, w, c in data.weeks if (s, w) == (2023, 10))
    late = list(data.team_log.filter(pl.col("gameday") >= D)["game_id"])
    pl2 = data.player_log.with_columns([pl.when(pl.col("gameday") >= D).then(pl.col(c) * 3 + 11).otherwise(pl.col(c)).alias(c)
                                        for c in bl.PLAYER_MARKETS.values()])
    tl2 = data.team_log.with_columns([pl.when(pl.col("gameday") >= D).then(pl.col(c) * 2 + 7).otherwise(pl.col(c)).alias(c)
                                      for c in ("pf", "pa")])
    tv2 = tv.with_columns([pl.when(pl.col("game_id").is_in(late)).then(pl.col(c) * 2 + 5).otherwise(pl.col(c)).alias(c)
                           for c in tv.columns if c not in ("game_id", "team")])
    et2 = ef.build_efficiency_tables(pl2, tl2, tv2, ln)
    for q in ef.EFFICIENCY_SPECS:
        a = et.players[q].filter(pl.col("gameday") < D).sort("player_id", "game_id")
        b = et2.players[q].filter(pl.col("gameday") < D).sort("player_id", "game_id")
        assert a.height == b.height > 500
        for c in vm.feature_columns(a):
            assert np.allclose(a[c].to_numpy().astype(float), b[c].to_numpy().astype(float), equal_nan=True), f"{q}.{c}"
    a, b = (t.teams.filter(pl.col("gameday") < D).sort("team", "gameday") for t in (et, et2))
    for c in vm.feature_columns(a):
        assert np.allclose(a[c].to_numpy().astype(float), b[c].to_numpy().astype(float), equal_nan=True), c


def test_efficiency_predictors_through_the_harness(real):
    data, tv, ln, et = real
    d = replace(data, game_extras=ef.ppp_extras(data.team_log, tv), player_extras=ef.efficiency_player_actuals)
    pp, gp = ef.efficiency_predictors(et)
    a = bt.walk_forward("e", d, pp, gp, seasons=[2022])
    assert a.equals(bt.walk_forward("e", d, pp, gp, seasons=[2022]))
    assert set(a["market"].unique()) == set(ef.EFFICIENCY_SPECS) | {"ppp_home", "ppp_away"}
    assert a["season"].max() == 2022 and a["prediction"].drop_nulls().min() >= 0
    assert a.filter(pl.col("market").is_in(["comp_pct", "catch_pct"]))["prediction"].drop_nulls().max() <= 1.0
    with pytest.raises(ValueError, match="locked"):
        bt.walk_forward("e", d, pp, gp, seasons=[2025])
    assert et.teams["season"].max() == 2024 and all(df["season"].max() == 2024 for df in et.players.values())
