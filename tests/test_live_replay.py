"""The whole live path, replayed on 2024 week 10 as if unplayed, against the committed walk-forward predictions of every wired layer (plan P5.7 item 3).

The week's outcomes (stats, snaps, plays, scores) are removed from a working copy of the raw database; the live run then builds targets from the schedule and the chart, features from
history, the 4a injury features from the frames that hold the target games, fits on rows before the week's cutoff and predicts. Required: identical to the committed predictions
to within 1e-9 for every prediction both have, and the live run covers at least 97% of the committed rows.
"""
import shutil
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

import config
from live import layers as LY
from live import store as ST
from live import weekly as W
from models import injuries as ij

BASE = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
INJ = config.PROCESSED_DIR / f"injury_ablation_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
needs_all = pytest.mark.skipif(not (config.RAW_DUCKDB_PATH.exists() and BASE.exists() and INJ.exists()), reason="needs the raw database and the committed predictions")
S, WK = 2024, 10


def _weights(tmp_path, layers):
    t = pl.read_csv(config.ROOT / "layer_weights.csv").filter(pl.col("layer").is_in(layers) | (pl.col("weight") == 0.0))
    p = tmp_path / "weights.csv"
    t.write_csv(p)
    return p


def _played():
    from eval import backtest as bt
    d = bt.load_backtest_data().player_log.filter((pl.col("season") == S) & (pl.col("week") == WK))
    return d.select("player_id", "game_id")


@pytest.fixture(scope="module")
def replay4a(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("replay")
    w = _weights(tmp, ["4a_injuries", "4b_simulation"])
    active = LY.read_active(w)
    sched = None
    from live import targets as TG
    import duckdb
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("select * from (select distinct on (game_id) * from schedules order by game_id, pulled_at desc) where season=2024").pl()
    con.close()
    cutoff = TG.target_games(sched, S, WK)["gameday"].min()
    models = W.train_models(active, train_cutoff=cutoff)
    out = W.run(datetime(2024, 11, 6, 12, 0, tzinfo=timezone.utc), store_dir=tmp / "store", models=models, version="replay", work_dir=tmp, layer_weights_csv=w,
                replay=dict(hide_outcomes_from=cutoff, status_clock=ij.main_run_as_of, fit_cutoff=cutoff))
    return out, active


def _compare(live: pl.DataFrame, ref: pl.DataFrame, tag: str, played: pl.DataFrame | None = None):
    """`played`: (player_id, game_id) of the game each player actually played that week. A player traded that week is listed by the stale chart for his old team too; the
    committed predictions only have the game he played, so only that live row is compared."""
    if played is not None:
        keep = live.filter(pl.col("player_id").is_null())
        live = pl.concat([live.filter(pl.col("player_id").is_not_null()).join(played, on=["player_id", "game_id"], how="semi"), keep])
    ref = ref.select(player_id=pl.when(pl.col("kind") == "player").then(pl.col("entity")), game_id=pl.when(pl.col("kind") == "game").then(pl.col("entity")), market="market", ref="prediction")
    lp = live.filter(pl.col("player_id").is_not_null()).join(ref.filter(pl.col("player_id").is_not_null()).drop("game_id"), on=["player_id", "market"], how="inner")
    lg = live.filter(pl.col("player_id").is_null()).drop("player_id").join(ref.filter(pl.col("game_id").is_not_null()).drop("player_id"), on=["game_id", "market"], how="inner")
    j = pl.concat([lp.select("market", "mean", "ref"), lg.select("market", "mean", "ref")]).drop_nulls()
    n_ref = ref.drop_nulls("ref").height
    assert j.height >= 0.97 * n_ref, (tag, j.height, n_ref)
    d = np.abs((j["mean"] - j["ref"]).to_numpy())
    assert d.max() < 1e-9, (tag, d.max(), j.filter((pl.col("mean") - pl.col("ref")).abs() > 1e-9).group_by("market").len().to_dicts())
    return j.height


@needs_all
def test_the_phase3_markets_match_the_committed_phase3_predictions(replay4a):
    out, active = replay4a
    p = out["predictions"]
    owned = {m for ms in active.values() for m in ms}
    ref = pl.read_parquet(BASE).filter((pl.col("method") == "vol_x_eff") & (pl.col("season") == S) & (pl.col("week") == WK) & ~pl.col("market").is_in(sorted(owned)))
    assert _compare(p.filter(pl.col("stack") == "phase3"), ref, "phase3", _played()) > 100


@needs_all
def test_the_4a_markets_match_the_committed_injury_layer_predictions(replay4a):
    out, active = replay4a
    p = out["predictions"]
    ref = pl.read_parquet(INJ).filter((pl.col("season") == S) & (pl.col("week") == WK) & pl.col("market").is_in(sorted(active["4a_injuries"])))
    live = p.filter(pl.col("stack") == "phase3+4a")
    assert set(live["market"]) == active["4a_injuries"]
    assert _compare(live, ref, "4a", _played()) > 300


@needs_all
def test_every_record_carries_the_stack_of_its_market_and_nothing_started_is_stored(replay4a):
    out, active = replay4a
    p = out["predictions"]
    owned = active["4a_injuries"] | active["4b_simulation"]
    assert set(zip(p["market"], p["stack"])) == ({(m, "phase3") for m in set(p["market"]) - owned} | {(m, "phase3+4a") for m in active["4a_injuries"]} | {("spread", "phase3+4b")})
    assert out["refused"] == 0 and out["written"] == p.height


GE = config.PROCESSED_DIR / "game_expectations.parquet"


@needs_all
def test_the_4b_spread_matches_the_simulated_spread_of_the_committed_game_expectations(replay4a):
    """4b.5's spread is the mean of simulate_game's first stream, Normal(margin_mean, margin_sd) seeded by the game id; the committed 4b.2 expectations give the reference."""
    from sim import simulate as sm
    out, active = replay4a
    assert active["4b_simulation"] == {"spread"}
    live = out["predictions"].filter((pl.col("market") == "spread"))
    assert set(live["stack"]) == {"phase3+4b"}
    ge = pl.read_parquet(GE).filter((pl.col("season") == S) & (pl.col("week") == WK)).select("game_id", "margin_mean", "margin_sd")
    ref = {r["game_id"]: float(sm.game_rng(r["game_id"]).normal(r["margin_mean"], r["margin_sd"], sm.BACKTEST.n_sim).mean()) for r in ge.iter_rows(named=True)}
    j = live.select("game_id", "mean").with_columns(ref=pl.col("game_id").replace_strict(ref, default=None)).drop_nulls()
    assert j.height == len(ref) == live.height
    assert np.abs((j["mean"] - j["ref"]).to_numpy()).max() < 1e-9
    dbg = out["debug"]["sim"]["expectations"].join(ge, on="game_id", suffix="_ref")
    assert np.abs((dbg["margin_mean"] - dbg["margin_mean_ref"]).to_numpy()).max() < 1e-9 and np.abs((dbg["margin_sd"] - dbg["margin_sd_ref"]).to_numpy()).max() < 1e-9
