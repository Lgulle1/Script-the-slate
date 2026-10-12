"""Phase 5 prediction store (live/store.py) and schedule-lines log (live/lines.py): the gates P5.1, P5.3, P5.6 of docs/overnight_plan.md."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from live import lines as LN
from live import store as ST

T0 = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)                    # a Friday run
KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)                  # Sunday 1 pm ET
PAST = datetime(2026, 10, 15, 0, 15, tzinfo=timezone.utc)                  # Thursday night, already started


def _preds(rows=None):
    rows = rows or [("2026_07_AAA_BBB", "p1", "rush_att", KICK, 12.0), ("2026_07_AAA_BBB", None, "spread", KICK, 3.5)]
    return pl.DataFrame([dict(game_id=g, player_id=p, market=m, kickoff_time=k, model_version="mv1", stack="phase3", mean=v, input_hash="h1") for g, p, m, k, v in rows],
                        schema_overrides={"player_id": pl.String})


def test_a_prediction_for_a_started_game_is_refused_and_logged(tmp_path):
    s = ST.PredictionStore(tmp_path)
    r = s.write(_preds([("g_past", "p1", "rush_att", PAST, 10.0), ("g_ok", "p2", "rush_att", KICK, 9.0)]), now=T0, run_id="r1")
    assert (r.written, r.refused) == (1, 1)
    assert s.load()["game_id"].to_list() == ["g_ok"]
    log = pl.read_csv(tmp_path / "refusals.csv")
    assert log["game_id"].to_list() == ["g_past"] and "kickoff" in log["reason"][0]


def test_kickoff_equal_to_the_clock_is_refused(tmp_path):
    r = ST.PredictionStore(tmp_path).write(_preds([("g", "p1", "rush_att", T0, 10.0)]), now=T0, run_id="r1")
    assert (r.written, r.refused) == (0, 1)


def test_a_rerun_with_changed_inputs_is_version_2_and_leaves_version_1_byte_identical(tmp_path):
    s = ST.PredictionStore(tmp_path)
    s.write(_preds(), now=T0, run_id="r1")
    files = {p.name: p.read_bytes() for p in tmp_path.glob("*.parquet")}
    changed = _preds().with_columns(mean=pl.col("mean") + 1.0, input_hash=pl.lit("h2"))
    r = s.write(changed, now=T0 + timedelta(days=1), run_id="r2")
    assert r.written == 2
    all_ = s.load().sort("market", "version")
    assert all_["market"].to_list() == ["rush_att", "rush_att", "spread", "spread"] and all_["version"].to_list() == [1, 2, 1, 2]
    assert all(files[n] == (tmp_path / n).read_bytes() for n in files)                                  # version 1 files untouched


def test_an_identical_rerun_writes_nothing(tmp_path):
    s = ST.PredictionStore(tmp_path)
    s.write(_preds(), now=T0, run_id="r1")
    r = s.write(_preds(), now=T0 + timedelta(hours=18), run_id="r2")
    assert (r.written, r.unchanged) == (0, 2) and s.load().height == 2 and len(list(tmp_path.glob("*.parquet"))) == 1


def test_grading_uses_the_newest_version_stored_before_kickoff(tmp_path):
    s = ST.PredictionStore(tmp_path)
    s.write(_preds(), now=T0, run_id="r1")
    s.write(_preds().with_columns(mean=pl.col("mean") + 1.0, input_hash=pl.lit("h2")), now=KICK - timedelta(hours=4), run_id="r2")
    g = s.gradable().sort("market")
    assert g["version"].to_list() == [2, 2] and g["mean"].to_list() == [13.0, 4.5]
    late = s.load().with_columns(stored_at=pl.lit(KICK + timedelta(minutes=1)).cast(pl.Datetime("us", "UTC")), version=pl.lit(3))             # a version stored after kickoff
    assert ST.latest_before_kickoff(pl.concat([s.load(), late.select(s.load().columns)])).get_column("version").to_list() == [2, 2]


def test_unknown_stays_null_never_zero_and_a_missing_mean_is_not_a_prediction(tmp_path):
    s = ST.PredictionStore(tmp_path)
    p = _preds([("g", "p1", "rush_att", KICK, None), ("g", "p2", "rush_att", KICK, 5.0)])
    r = s.write(p, now=T0, run_id="r1")
    assert (r.written, r.refused) == (1, 1)
    row = s.load().row(0, named=True)
    assert row["median"] is None and row["q05"] is None and row["no_match"] is None                     # never filled with 0
    assert "no mean" in pl.read_csv(tmp_path / "refusals.csv")["reason"][0]


def test_no_column_comes_from_a_sportsbook_line(tmp_path):
    s = ST.PredictionStore(tmp_path)
    s.write(_preds(), now=T0, run_id="r1")
    cols = set(s.load().columns)
    assert cols == set(ST.RECORD_COLUMNS)
    assert not {c for c in cols if any(w in c.lower() for w in ("line", "odds", "vegas", "book", "juice", "price"))}
    with pytest.raises(ValueError, match="not in the record schema"):
        s.write(_preds().with_columns(spread_line=pl.lit(3.0)), now=T0, run_id="r2")                    # a line cannot be smuggled in
    assert set(LN.LINE_COLUMNS).isdisjoint(set(ST.RECORD_COLUMNS) - {"game_id"})


def test_the_prediction_path_never_imports_the_lines_log():
    import ast
    tree = ast.parse(Path(ST.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("lines" in m for m in imported), imported


def test_the_lines_log_is_append_only_and_separate(tmp_path):
    sched = pl.DataFrame({"game_id": ["g1", "g2"], "spread_line": [3.0, None], "total_line": [44.5, 41.0], "home_moneyline": [-150, None], "away_moneyline": [130, None]})
    path = tmp_path / "lines_log.csv"
    assert LN.log_lines(sched, captured_at=T0, path=path) == 2
    assert LN.log_lines(sched, captured_at=T0 + timedelta(hours=6), path=path) == 0                      # unchanged lines are not logged twice
    moved = sched.with_columns(spread_line=pl.when(pl.col("game_id") == "g1").then(3.5).otherwise(pl.col("spread_line")))
    assert LN.log_lines(moved, captured_at=T0 + timedelta(days=1), path=path) == 1
    t = pl.read_csv(path)
    assert t.filter(pl.col("game_id") == "g1")["spread_line"].to_list() == [3.0, 3.5] and t.filter(pl.col("game_id") == "g2")["spread_line"].null_count() == 1


def test_a_new_input_hash_or_model_version_with_the_same_mean_is_still_a_new_version(tmp_path):
    s = ST.PredictionStore(tmp_path)
    s.write(_preds(), now=T0, run_id="r1")
    assert s.write(_preds().with_columns(input_hash=pl.lit("h2")), now=T0 + timedelta(hours=1), run_id="r2").written == 2                    # the inputs moved, the mean did not
    assert s.write(_preds().with_columns(input_hash=pl.lit("h2"), model_version=pl.lit("mv2")), now=T0 + timedelta(hours=2), run_id="r3").written == 2
    assert s.load().filter(pl.col("market") == "spread").sort("version")["version"].to_list() == [1, 2, 3]


def test_a_change_of_stack_label_is_a_new_version_even_with_the_same_mean_and_inputs(tmp_path):
    s = ST.PredictionStore(tmp_path)
    s.write(_preds(), now=T0, run_id="r1")
    r = s.write(_preds().with_columns(stack=pl.lit("phase3+4a")), now=T0 + timedelta(hours=1), run_id="r2")
    assert r.written == 2 and s.load().sort("market", "version")["stack"].to_list() == ["phase3", "phase3+4a", "phase3", "phase3+4a"]
    with pytest.raises(ValueError, match="missing columns"):
        s.write(_preds().drop("stack"), now=T0, run_id="r3")                                         # a record without its stack label is not stored


def test_a_nan_or_infinite_mean_is_not_a_prediction(tmp_path):
    s = ST.PredictionStore(tmp_path)
    p = _preds([("g", "p1", "rush_att", KICK, float("nan")), ("g", "p2", "rush_att", KICK, float("inf")), ("g", "p3", "rush_att", KICK, 7.0)])
    r = s.write(p, now=T0, run_id="r1")
    assert (r.written, r.refused) == (1, 2) and s.load()["player_id"].to_list() == ["p3"]
    assert "not finite" in pl.read_csv(tmp_path / "refusals.csv")["reason"][0]
