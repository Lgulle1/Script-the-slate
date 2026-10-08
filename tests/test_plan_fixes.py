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


def _old_eligible(family, slot):   # the slot pools of slotpool_v1 (RB 1-2, WR 1-3, TE 1-2), kept here for reference only
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


def test_markets_dict_keeps_the_original_eleven_and_adds_the_two_qb_rushing_markets():
    qb = ["qb_rush_att", "qb_rush_yds"]
    assert len(config.MARKETS) == 13 and set(config.MARKETS) == set(OLD_PENALTY_MAP) | set(qb)
    assert {m: s["penalty_key"] for m, s in config.MARKETS.items() if m not in qb} == OLD_PENALTY_MAP   # the original eleven, unchanged
    assert {m: config.MARKETS[m]["penalty_key"] for m in qb} == {"qb_rush_att": "rush_att", "qb_rush_yds": "rush_yds"}  # assumption
    assert config.BASELINE_TO_PENALTY_MARKET == {m: s["penalty_key"] for m, s in config.MARKETS.items()}   # the alias
    assert {m: c for m, c in bl.PLAYER_MARKETS.items() if m not in qb} == OLD_PLAYER_MARKETS
    assert {m: bl.PLAYER_MARKETS[m] for m in qb} == {"qb_rush_att": "rush_att_ex_kneel", "qb_rush_yds": "rush_yds_ex_kneel"}
    assert bl.GAME_MARKETS == ("spread", "moneyline", "total")
    for m in ("pass_att", "pass_cmp", "pass_yds"):                              # QB1 is unchanged
        assert config.MARKETS[m]["pool"] == {"QB": (1,)}
    for m in ("rush_att", "rush_yds", "targets", "rec", "rec_yds"):          # the five rule-driven markets (3.5.2c)
        assert config.MARKETS[m]["pool"] == "ELIGIBLE_PLAYER_RULE"
    for m in ("qb_rush_att", "qb_rush_yds"):
        assert config.MARKETS[m]["pool"] == "QB_RUSH_RULE" and config.QB_RUSH_MIN_ATT == 4
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
    """players_current is exactly the latest pull per gsis_id of the players table (however many pulls it holds),
    so master-id joins read one consistent row per player."""
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    allrows = con.execute("SELECT gsis_id, pfr_id, espn_id, pulled_at FROM players WHERE gsis_id IS NOT NULL").pl()
    latest = (allrows.sort("pulled_at", descending=True, nulls_last=True).unique(subset="gsis_id", keep="first", maintain_order=True)
              .select("gsis_id", "pfr_id", "espn_id").sort("gsis_id"))
    view = ids._load_master(config.DUCKDB_PATH).sort("gsis_id")
    assert latest.equals(view) and view.height > 20_000 and view["gsis_id"].is_unique().all()
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
    path = config.result_path("best_baseline")
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet")
    best = pl.read_parquet(path)
    assert best["market"].n_unique() == 13 and best.filter(pl.col("is_best")).height == 13   # 11 original + 2 QB-rushing
    assert set(best["method"]) == set(bl.METHODS)


# ------------------------------------------------------------------ 3.5.2b: FEATURE_HISTORY_START and the 2016-2019 backfill
def test_feature_history_start_is_2020_and_in_every_loader_window():
    assert config.FEATURE_HISTORY_START == 2020 and config.DATA_START_SEASON == 2016
    sql = config.season_sql(None)
    assert "season >= 2020" in sql and "season <= 2024" in sql
    assert "season >= 2020" in config.season_sql(2022) and "season <= 2022" in config.season_sql(2022)
    with pytest.raises(config.HoldoutError):
        config.season_sql(2025)


@needs_db
def test_loaders_read_from_2020_on_even_though_older_seasons_are_stored():
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    older = con.execute("SELECT count(*) FROM schedules WHERE season < 2020").fetchone()[0]
    if not older:
        pytest.skip("2016-2019 not pulled in this database")
    assert older > 0
    tl = bl.build_team_game_log()
    pg = bl.build_player_game_log()
    ln = lu.build_lineups()
    tv = vf.build_team_volume()
    roles, _ = bl.build_role_table()
    assert tl["season"].min() == pg["season"].min() == ln["season"].min() == roles["season"].min() == 2020
    assert tv["game_id"].str.slice(0, 4).cast(pl.Int32).min() == 2020 and tl["season"].max() == 2024
    data = bt.load_backtest_data()
    assert data.team_log["season"].min() == data.player_log["season"].min() == 2020


def test_pull_seasons_start_at_the_data_start_season(monkeypatch):
    monkeypatch.setattr(pn.nflreadpy, "get_current_season", lambda: 2026)
    seasons = pn.pull_seasons()
    assert seasons[0] == config.DATA_START_SEASON == 2016 and seasons[-1] == 2026 and seasons == sorted(set(seasons))


def test_backfill_pulls_only_2016_2019_and_clips_ftn_and_pfr_and_skips_participation(monkeypatch, tmp_path):
    calls = {}

    def fake(name):
        def load(seasons, **kw):
            calls.setdefault(name, []).append(list(seasons))
            return pl.DataFrame({"season": list(seasons)})
        return load

    for fn in ("load_schedules", "load_pbp", "load_player_stats", "load_snap_counts", "load_rosters_weekly", "load_injuries",
               "load_depth_charts", "load_ftn_charting", "load_nextgen_stats", "load_pfr_advstats", "load_ff_opportunity",
               "load_officials", "load_participation"):
        monkeypatch.setattr(pn.nflreadpy, fn, fake(fn))
    out = pn.pull_pre_feature_history(tmp_path / "r.duckdb")
    assert not [v for v in out.values() if isinstance(v, str)]                       # nothing failed
    for name, seasons in calls.items():
        assert all(s < config.FEATURE_HISTORY_START and s >= config.DATA_START_SEASON for call in seasons for s in call), name
    assert calls["load_schedules"] == [[2016, 2017, 2018, 2019]]
    assert calls["load_pfr_advstats"] == [[2018, 2019]] * 3                          # rush / rec / pass, clipped to 2018+
    assert "load_ftn_charting" not in calls and out["ftn_charting"] == 0             # FTN starts in 2022
    assert "load_participation" not in calls                                          # already stored for 2016-2025
    con = duckdb.connect(str(tmp_path / "r.duckdb"))
    assert con.execute("SELECT count(*) FROM schedules WHERE pulled_at IS NOT NULL").fetchone()[0] == 4
