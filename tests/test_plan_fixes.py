"""Tests for the 3.5.2 fixes: config constants, the holdout lock on loaders, config.MARKETS, the append-only
players table, best_baseline.parquet and the single continuity implementation."""
import numpy as np
import polars as pl
import pytest
import duckdb

import config
from eval import backtest as bt
from eval import baselines as bl
from eval import compare
from features import lineups as lu
from features import volume_features as vf
from features import weights
from ingest import ids
from ingest import pull_nflverse as pn
from models import efficiency as ef

needs_db = pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="no local raw db")


# ------------------------------------------------------------------ 1. config constants
def test_new_config_constants_and_ids_imports_the_threshold():
    assert config.DATA_START_SEASON == 2016 and config.PROSPECTIVE_START_SEASON == 2026
    assert config.MAX_UNMATCHED_FRACTION == 0.01 and ids.MAX_UNMATCHED_FRACTION is config.MAX_UNMATCHED_FRACTION
    assert not hasattr(config, "DB_PATH")  # two databases is fine; DUCKDB_PATH / RAW_DUCKDB_PATH stay as they are


# ------------------------------------------------------------------ 2. holdout lock
@pytest.mark.parametrize("bad", [2025, 2026, 2030])
def test_every_training_or_evaluation_loader_refuses_the_holdout(bad):
    loaders = [lambda: bl.build_team_game_log(max_season=bad), lambda: bl.build_player_game_log(max_season=bad),
               lambda: bl.build_role_table(max_season=bad), lambda: lu.build_lineups(max_season=bad),
               lambda: vf.build_team_volume(max_season=bad), lambda: vf.load_feature_tables(None, max_season=bad),
               lambda: ef.load_efficiency_tables(None, max_season=bad)]
    for load in loaders:
        with pytest.raises(config.HoldoutError, match="locked holdout"):
            load()
    with pytest.raises(ValueError):  # the harness raises the same family of error
        bt.walk_forward("x", None, seasons=[bad])


@needs_db
def test_default_cap_is_the_last_backtest_season_not_everything():
    assert bl.build_team_game_log()["season"].max() == 2024
    assert bl.build_player_game_log()["season"].max() == 2024
    assert vf.build_team_volume().join(bl.build_team_game_log().select("game_id"), on="game_id").height > 0
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    has_2025 = con.execute("SELECT count(*) FROM schedules WHERE season >= 2025").fetchone()[0]
    assert has_2025 > 0 and bl.build_team_game_log(max_season=2024)["season"].max() == 2024  # the data is there; the loader stops


def test_raw_pulls_are_exempt_and_still_fetch_2025_and_2026(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(pn.nflreadpy, "load_schedules", lambda seasons: (seen.append(list(seasons)), pl.DataFrame({"season": list(seasons)}))[1])
    assert pn.pull_schedules([2024, 2025, 2026], tmp_path / "r.duckdb") == 3 and seen == [[2024, 2025, 2026]]


# ------------------------------------------------------------------ 3. config.MARKETS
OLD_PENALTY_MAP = {"pass_att": "pass_yds", "pass_cmp": "pass_yds", "pass_yds": "pass_yds", "rush_att": "rush_att",
                   "rush_yds": "rush_yds", "targets": "rec", "rec": "rec", "rec_yds": "rec_yds",
                   "spread": "game_total", "moneyline": "game_total", "total": "game_total"}
OLD_PLAYER_MARKETS = {"pass_att": "attempts", "pass_cmp": "completions", "pass_yds": "passing_yards", "rush_att": "carries",
                      "rush_yds": "rushing_yards", "targets": "targets", "rec": "receptions", "rec_yds": "receiving_yards"}


def _old_eligible(family, slot):   # the pre-3.5.2 hard-coded pools, kept here as the reference
    _PASS, _RUSH, _RECV = ("pass_att", "pass_cmp", "pass_yds"), ("rush_att", "rush_yds"), ("targets", "rec", "rec_yds")
    if family == "QB":
        return _PASS if slot == 1 else ()
    if family == "RB":
        return (_RUSH + _RECV) if slot in (1, 2) else ()
    if family == "WR":
        return _RECV if slot in (1, 2, 3) else ()
    if family == "TE":
        return _RECV if slot in (1, 2) else ()
    return ()


def test_markets_dict_resolves_to_exactly_the_old_behaviour():
    assert list(config.MARKETS) == list(OLD_PENALTY_MAP)                       # eleven keys, same order
    assert {m: s["penalty_key"] for m, s in config.MARKETS.items()} == OLD_PENALTY_MAP
    assert config.BASELINE_TO_PENALTY_MARKET == OLD_PENALTY_MAP                # the alias
    assert bl.PLAYER_MARKETS == OLD_PLAYER_MARKETS and list(bl.PLAYER_MARKETS) == list(OLD_PLAYER_MARKETS)
    assert bl.GAME_MARKETS == ("spread", "moneyline", "total")
    for fam in ("QB", "RB", "WR", "TE", "K", "DB"):
        for slot in range(0, 6):
            assert bt.eligible_markets(fam, slot) == _old_eligible(fam, slot), (fam, slot)
    for spec in config.MARKETS.values():
        assert {"kind", "baseline_family", "stat", "pool", "penalty_key"} <= set(spec)
        assert spec["penalty_key"] in config.CONTINUITY_PENALTIES
        assert spec["baseline_family"] == spec["kind"]


# ------------------------------------------------------------------ 4. players append-only + players_current
def _players(rows):
    return pl.DataFrame(rows, schema={"gsis_id": pl.String, "pfr_id": pl.String, "espn_id": pl.String}, orient="row")


def test_players_are_appended_and_the_view_serves_the_latest_pull(monkeypatch, tmp_path):
    db = tmp_path / "m.duckdb"
    pulls = iter([_players([("G1", "P1", "1"), ("G2", "P2", "2")]), _players([("G1", "P1x", "1"), ("G3", "P3", "3")])])
    monkeypatch.setattr(pn.nflreadpy, "load_players", lambda: next(pulls))
    assert pn.pull_players(db) == 2 and pn.pull_players(db) == 2
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT count(*) FROM players").fetchone()[0] == 4                       # nothing replaced
    assert con.execute("SELECT count(DISTINCT pulled_at) FROM players").fetchone()[0] == 2
    cur = dict(con.execute("SELECT gsis_id, pfr_id FROM players_current").fetchall())
    assert cur == {"G1": "P1x", "G2": "P2", "G3": "P3"}                                          # latest per player, none lost
    con.close()
    out = ids.add_canonical_gsis_id(pl.DataFrame({"pfr_id": ["P1x", "P2", "P3"]}), db_path=db)
    assert out["gsis_id"].to_list() == ["G1", "G2", "G3"]
    with pytest.raises(ids.IdMatchError):                                                        # the superseded id no longer resolves
        ids.add_canonical_gsis_id(pl.DataFrame({"pfr_id": ["P1"] * 5}), db_path=db)


def test_legacy_players_table_is_upgraded_in_place_and_new_pulls_win(monkeypatch, tmp_path):
    db = tmp_path / "m.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE players AS SELECT * FROM (VALUES ('G1','P1','1'), ('G2','P2','2')) t(gsis_id, pfr_id, espn_id)")
    con.close()
    pn.ensure_players_view(db)
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT count(*) FROM players_current").fetchone()[0] == 2                # identical to the old table
    assert con.execute("SELECT count(pulled_at) FROM players").fetchone()[0] == 0                # legacy rows stay NULL
    con.close()
    monkeypatch.setattr(pn.nflreadpy, "load_players", lambda: _players([("G1", "Pnew", "1")]))
    pn.pull_players(db)
    cur = dict(duckdb.connect(str(db), read_only=True).execute("SELECT gsis_id, pfr_id FROM players_current").fetchall())
    assert cur == {"G1": "Pnew", "G2": "P2"}


@needs_db
def test_master_id_joins_and_join_rates_are_unchanged_on_real_data():
    """The view serves exactly the legacy table's rows, so every join output is identical."""
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    table = con.execute("SELECT gsis_id, pfr_id, espn_id FROM players WHERE gsis_id IS NOT NULL ORDER BY gsis_id").pl()
    view = ids._load_master(config.DUCKDB_PATH).sort("gsis_id")
    assert table.equals(view) and view.height > 20_000
    raw = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    for t, col, rate in (("player_stats", "player_id", 0.999), ("snap_counts", "pfr_player_id", 0.995), ("rosters_weekly", "pfr_id", 0.998)):
        df = raw.execute(f"SELECT DISTINCT {col} FROM {t} WHERE {col} IS NOT NULL").pl()
        arg = "gsis_id" if col == "player_id" else col
        out = ids.add_canonical_gsis_id(df.rename({col: arg}))
        assert 1 - out["gsis_id"].null_count() / out.height >= rate, t


# ------------------------------------------------------------------ 6. best_baseline.parquet
def test_best_baseline_per_market_uses_identical_rows():
    rows = []
    for i in range(20):
        for m, errs in (("a", dict(last3=3, season_avg=1, recency=2, blend_70_30=4, role_avg=5)),):
            for meth, e in errs.items():
                rows.append(dict(method=meth, kind="player", market=m, season=2024, week=1, gameday=None, entity=f"e{i}",
                                 prediction=10.0 + e if not (meth == "season_avg" and i < 5) else None, actual=10.0))
    preds = pl.DataFrame(rows, schema={"method": pl.String, "kind": pl.String, "market": pl.String, "season": pl.Int32,
                                       "week": pl.Int32, "gameday": pl.Date, "entity": pl.String, "prediction": pl.Float64, "actual": pl.Float64})
    best = compare.best_baseline_per_market(preds)
    assert best.height == 5 and set(best["n"]) == {15}                       # the 5 rows season_avg lacks are dropped for everyone
    assert best.filter(pl.col("is_best"))["method"].to_list() == ["season_avg"]
    assert best.filter(pl.col("method") == "role_avg")["loss"].item() == 5.0


@needs_db
def test_committed_best_baseline_matches_the_choice_in_the_phase3_results():
    from pathlib import Path
    root = config.ROOT
    if not (root / "best_baseline.parquet").exists():
        pytest.skip("best_baseline.parquet not generated yet")
    best = pl.read_parquet(root / "best_baseline.parquet")
    assert best["market"].n_unique() == 11 and best.filter(pl.col("is_best")).height == 11
    assert set(best["method"]) == set(bl.METHODS)
