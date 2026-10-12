"""Replay test of the live scoring path (plan P5.2 / P5.6): a past week, scored as if it had not been played, must reproduce the walk-forward harness.

For 2024 week 10 the outcomes of the week are nulled, the feature tables are built from history before the week's cutoff plus those placeholder rows, the models
are fit on rows before the cutoff, and the predictions are compared with the committed walk-forward predictions of the same week. Identical = the live path
adds nothing of its own to a prediction and sees nothing from the future.
"""
import numpy as np
import polars as pl
import pytest

import config
from eval import backtest as bt
from features import lineups as lu
from features import volume_features as vf
from live import predict as LP
from models import efficiency as ef

PRED = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
needs_db = pytest.mark.skipif(not (config.RAW_DUCKDB_PATH.exists() and PRED.exists()), reason="needs the raw database and the committed-run predictions")
S, W = 2024, 10
STATS = ["attempts", "completions", "passing_yards", "carries", "rushing_yards", "targets", "receptions", "receiving_yards", "rush_att_ex_kneel", "rush_yds_ex_kneel"]


@pytest.fixture(scope="module")
def world():
    data = bt.load_backtest_data()
    tv = vf.build_team_volume(max_season=2024)
    ln = lu.build_lineups(max_season=2024)
    full_vol = vf.build_feature_tables(data.player_log, data.team_log, tv, ln)
    full_eff = ef.build_efficiency_tables(data.player_log, data.team_log, tv, ln)
    wk = lambda df: df.filter((pl.col("season") == S) & (pl.col("week") == W))
    cut = wk(data.team_log)["gameday"].min()
    hist_t, hist_p = data.team_log.filter(pl.col("gameday") < cut), data.player_log.filter(pl.col("gameday") < cut)
    wk_p, wk_t = wk(data.player_log), wk(data.team_log)
    blank_p = wk_p.with_columns([pl.lit(None, dtype=wk_p[c].dtype).alias(c) for c in STATS])           # the week as if unplayed: no outcome in any stat
    blank_t = wk_t.with_columns(pf=pl.lit(None, dtype=wk_t["pf"].dtype), pa=pl.lit(None, dtype=wk_t["pa"].dtype))
    tvh = tv.join(hist_t.select("game_id").unique(), on="game_id", how="inner")                            # team volume only of games already played
    pl_, tl_ = pl.concat([hist_p, blank_p]), pl.concat([hist_t, blank_t])
    live_vol, live_eff = vf.build_feature_tables(pl_, tl_, tvh, ln), ef.build_efficiency_tables(pl_, tl_, tvh, ln)
    return dict(data=data, full_vol=full_vol, full_eff=full_eff, live_vol=live_vol, live_eff=live_eff, cut=cut, wk=wk)


@needs_db
def test_feature_rows_of_an_unplayed_week_equal_the_standard_ones(world):
    wk = world["wk"]
    for kind, full, live in (("vol", world["full_vol"], world["live_vol"]), ("eff", world["full_eff"], world["live_eff"])):
        for q in full.players:
            a, b = wk(full.players[q]).sort("player_id", "game_id"), wk(live.players[q]).sort("player_id", "game_id")
            assert a.height == b.height > 0, (kind, q)
            for c in [c for c in a.columns if c not in ("label", "den")]:
                x, y = a[c].to_numpy(), b[c].to_numpy()
                assert (np.allclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-9, atol=1e-12) if x.dtype.kind == "f" else (x == y).all()), (kind, q, c)
        a, b = wk(full.teams).sort("team"), wk(live.teams).sort("team")
        assert a.height == b.height > 0


@needs_db
def test_the_live_path_reproduces_the_walk_forward_predictions_of_that_week(world):
    wk, data = world["wk"], world["data"]
    data0 = world["data"]
    models = LP.fit_models(world["full_vol"], world["full_eff"], data0.player_log.filter(pl.col("gameday") < world["cut"]), data0.team_log.filter(pl.col("gameday") < world["cut"]))               # the replay fits on rows before the cutoff, as the harness does
    vt = vf.FeatureTables({q: wk(d) for q, d in world["live_vol"].players.items()}, wk(world["live_vol"].teams))
    et = vf.FeatureTables({q: wk(d) for q, d in world["live_eff"].players.items()}, wk(world["live_eff"].teams))
    elig_of = {(r["game_id"], r["player_id"]): r["elig"] for r in wk(data.player_log).iter_rows(named=True) if r["elig"]}
    P = LP.predict_players(models, vt, et, elig_of)
    ref = pl.read_parquet(PRED).filter((pl.col("method") == "vol_x_eff") & (pl.col("season") == S) & (pl.col("week") == W))
    rp = ref.filter(pl.col("kind") == "player").select(player_id="entity", market="market", ref="prediction")
    j = P.join(rp, on=["player_id", "market"], how="inner")
    assert j.height == P.height == rp.height > 500
    assert ((j["mean"].is_null()) == (j["ref"].is_null())).all()
    both = j.drop_nulls()
    assert np.abs((both["mean"] - both["ref"]).to_numpy()).max() < 1e-9
    tl = wk(data.team_log)
    games = tl.filter(pl.col("is_home")).select("game_id", home_team="team").join(tl.filter(~pl.col("is_home")).select("game_id", away_team="team"), on="game_id")
    G = LP.predict_games(models, vt, et, games)
    rg = ref.filter(pl.col("kind") == "game").select(game_id="entity", market="market", ref="prediction")
    jg = G.join(rg, on=["game_id", "market"], how="inner").drop_nulls()
    assert jg.height == G.height == rg.height and np.abs((jg["mean"] - jg["ref"]).to_numpy()).max() < 1e-9


@needs_db
def test_putting_the_week_s_outcomes_back_changes_no_feature_of_that_week(world):
    """The future-blindness property of the feature builders, stated on the live inputs: the unplayed week and the played week give the same rows."""
    wk = world["wk"]
    a = wk(world["live_vol"].players["rush_att"]).sort("player_id")
    b = wk(world["full_vol"].players["rush_att"]).sort("player_id")
    assert a.select([c for c in a.columns if c != "label"]).equals(b.select([c for c in a.columns if c != "label"]))
