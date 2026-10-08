"""Phase 4 input tables (3.5.3): row counts from independent sources, timing, determinism, no holdout, share sums.

Built for 2024 only (the models still train on 2020-2023 history), twice, so determinism is checked on real data."""
import hashlib

import duckdb
import polars as pl
import pytest

import config
from eval import baselines as bl
from eval import eligibility as elig
from features import phase4_inputs as p4

pytestmark = pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="raw database not present")
SEASON = 2024


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    a = p4.build_all((SEASON,))
    b = p4.build_all((SEASON,))
    return a, b


def _digest(t: pl.DataFrame) -> str:
    return hashlib.sha256(t.hash_rows(seed=0).to_numpy().tobytes() + "|".join(t.columns).encode()).hexdigest()


def test_content_identical_on_second_run(built):
    a, b = built
    for name in p4.TABLES:
        assert a[name].equals(b[name]), name
        assert _digest(a[name]) == _digest(b[name])


def test_persisted_parquet_is_byte_identical(built, tmp_path):
    a, _ = built
    p4.persist(a, db_path=tmp_path / "a.duckdb", out_dir=tmp_path)
    first = {n: (tmp_path / f"{n}.parquet").read_bytes() for n in p4.TABLES}
    p4.persist(a, db_path=tmp_path / "b.duckdb", out_dir=tmp_path)
    assert first == {n: (tmp_path / f"{n}.parquet").read_bytes() for n in p4.TABLES}
    con = duckdb.connect(str(tmp_path / "a.duckdb"))
    assert con.execute("SELECT count(*) FROM team_game_rates").fetchone()[0] == a["team_game_rates"].height


def test_no_holdout_rows(built):
    a, _ = built
    for name, t in a.items():
        assert t["season"].max() < config.HOLDOUT_SEASON, name
        assert set(t["season"].unique()) == {SEASON}, name
    with pytest.raises(config.HoldoutError):
        p4.build_all((config.HOLDOUT_SEASON,))


def test_trained_through_week_is_before_week(built):
    w = built[0]["walkforward_predictions"]
    assert (w["trained_through_week"] < w["week"]).all()
    assert ((w["trained_through_season"] < w["season"]) | (w["trained_through_week"] < w["week"])).all()
    for name, t in built[0].items():     # every row records its cutoff, and the cutoff precedes the outcome
        assert t["cutoff_date"].null_count() == 0, name
        assert (t["cutoff_date"] <= t["outcome_known_from"]).all(), name


def test_team_game_rates_rows(built):
    t = built[0]["team_game_rates"]
    games = bl.build_team_game_log(max_season=SEASON).filter(pl.col("season") == SEASON)
    assert t.height == games.height
    assert t.select("game_id", "team").unique().height == t.height
    shares = t.select(pl.sum_horizontal([f"share_{s}" for s in p4.STATES]).alias("s"))["s"]
    assert (shares - 1).abs().max() < 1e-9
    assert (t["dropbacks"] <= t["plays"]).all()


def test_player_game_roles_rows_and_shares(built):
    r = built[0]["player_game_roles"]
    assert r.select("game_id", "team", "player_id").unique().height == r.height
    # independent source: every skill player-game in player_stats with a carry, target or pass attempt is present
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    n = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT ON (player_id, season, week) * FROM player_stats ORDER BY player_id, season, week, pulled_at DESC) "
        f"WHERE season_type = 'REG' AND season = {SEASON} AND position IN ('QB','RB','HB','FB','WR','TE') "
        "AND (coalesce(carries,0) > 0 OR coalesce(targets,0) > 0 OR coalesce(attempts,0) + coalesce(sacks_suffered,0) > 0)").fetchone()[0]
    con.close()
    assert r.height >= n
    assert set(r["expected_role"]) <= set(p4.ROLES) and set(r["played_role"].drop_nulls()) <= set(p4.ROLES)
    s = r.group_by("game_id", "team").agg([pl.col(c).sum() for c in ("carry_share", "target_share", "dropback_share")])
    for c in ("carry_share", "target_share", "dropback_share"):
        assert s[c].max() <= 1 + 1e-9, c


def test_prediction_table_row_counts(built):
    w, m = built[0]["walkforward_predictions"], built[0]["market_predictions"]
    log = bl.build_player_game_log(max_season=SEASON).filter(pl.col("season") == SEASON)
    marks = {mk: log.filter(pl.col("elig").map_elements(lambda e, mk=mk: mk in elig.markets_of(e), return_dtype=pl.Boolean)) for mk in bl.PLAYER_MARKETS}
    games = bl.build_team_game_log(max_season=SEASON).filter((pl.col("season") == SEASON) & pl.col("is_home"))
    ties = log.select("game_id").unique()  # placeholder to keep the frame import used
    # market_predictions: eligible player-games per market, plus every game for spread/total and non-tied games for moneyline
    for mk, f in marks.items():
        assert m.filter(pl.col("market") == mk).height == f.height, mk
    assert m.filter(pl.col("market") == "spread").height == games.height
    assert m.filter(pl.col("market") == "total").height == games.height
    assert m.filter(pl.col("market") == "moneyline").height == games.filter(pl.col("pf") != pl.col("pa")).height
    # walkforward_predictions: volume quantities one per eligible player-game; ratios only where the denominator is positive
    for q, mk in (("pass_att", "pass_att"), ("rush_att", "rush_att"), ("targets", "targets"), ("qb_rush_att", "qb_rush_att")):
        assert w.filter(pl.col("quantity") == q).height == marks[mk].height, q
    for q, mk, den in (("comp_rate", "pass_cmp", "attempts"), ("yds_per_cmp", "pass_yds", "completions"), ("ypc", "rush_yds", "carries"),
                       ("qb_ypc", "qb_rush_yds", "rush_att_ex_kneel"), ("catch_rate", "rec", "targets"), ("yds_per_rec", "rec_yds", "receptions")):
        assert w.filter(pl.col("quantity") == q).height == marks[mk].filter(pl.col(den) > 0).height, q
    assert w.filter(pl.col("quantity") == "team_plays").height == 2 * games.height
    assert w.filter(pl.col("quantity") == "pts_per_play").height == 2 * games.height
    done = w.filter(pl.col("expected").is_not_null())
    assert ((done["actual"] - done["expected"]) - done["residual"]).abs().max() < 1e-9
    assert w.filter(pl.col("expected").is_null())["residual"].null_count() == w.filter(pl.col("expected").is_null()).height


def test_best_baseline_uses_only_earlier_weeks(built):
    m = built[0]["market_predictions"]
    first = m.filter(pl.col("week") == m["week"].min())
    assert first["best_baseline"].null_count() == first.height   # nothing earlier in the table to judge by
