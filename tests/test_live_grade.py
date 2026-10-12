"""Weekly grading (live/grade.py): plan P5.5 / P5.6."""
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from live import grade as GR
from live import store as ST

T0 = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)
KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)
TUE = datetime(2026, 10, 20, 14, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    s = ST.PredictionStore(tmp_path / "preds")
    rows = [("g1", "p1", "rush_att", 12.0), ("g1", "p2", "rush_att", 8.0), ("g1", None, "spread", 3.5), ("g1", None, "moneyline", 0.62)]
    s.write(pl.DataFrame([dict(game_id=g, player_id=p, market=m, kickoff_time=KICK, model_version="mv1", stack="phase3", mean=v, input_hash="h") for g, p, m, v in rows],
                         schema_overrides={"player_id": pl.String}), now=T0, run_id="r1")
    return s


def _actuals():
    return pl.DataFrame({"game_id": ["g1", "g1", "g1"], "player_id": ["p1", None, None], "market": ["rush_att", "spread", "moneyline"], "actual": [15.0, -2.0, 0.0]},
                        schema_overrides={"player_id": pl.String})


def test_errors_and_the_did_not_play_rows(tmp_path):
    s = _store(tmp_path)
    g = GR.grade(s.gradable(), _actuals(), tmp_path / "graded.parquet", graded_at=TUE).sort("market", "player_id")
    r1 = g.filter(pl.col("player_id") == "p1").row(0, named=True)
    assert (r1["actual"], r1["error"], r1["abs_error"], r1["did_not_play"]) == (15.0, -3.0, 3.0, False)
    p2 = g.filter(pl.col("player_id") == "p2").row(0, named=True)
    assert (p2["actual"], p2["did_not_play"], p2["error"]) == (0.0, True, 8.0)                           # error = prediction - actual; eligible, not in the stats: graded as 0 and flagged
    ml = g.filter(pl.col("market") == "moneyline").row(0, named=True)
    assert ml["actual"] == 0.0 and ml["brier"] == pytest.approx(0.62 ** 2) and ml["did_not_play"] is False
    assert g["track"].unique().to_list() == ["pure"] and g["quantile_band"].null_count() == g.height       # no quantiles stored yet: null, not 0


def test_grading_twice_is_identical_and_adds_nothing(tmp_path):
    s = _store(tmp_path)
    path = tmp_path / "graded.parquet"
    a = GR.grade(s.gradable(), _actuals(), path, graded_at=TUE)
    b1 = path.read_bytes()
    b = GR.grade(s.gradable(), _actuals(), path, graded_at=TUE + timedelta(days=1))
    assert path.read_bytes() == b1 and a.height == b.height == 4


def test_a_version_stored_after_kickoff_is_never_graded(tmp_path):
    s = _store(tmp_path)
    t = s.load()
    late = t.with_columns(version=pl.lit(2, dtype=pl.Int32), stored_at=pl.lit(KICK + timedelta(minutes=5)).cast(pl.Datetime("us", "UTC")), mean=pl.col("mean") + 100.0)
    g = GR.grade(ST.latest_before_kickoff(pl.concat([t, late])), _actuals(), tmp_path / "graded.parquet", graded_at=TUE)
    assert g["version"].unique().to_list() == [1] and g.filter(pl.col("market") == "spread")["mean"].to_list() == [3.5]


def test_a_game_with_no_actuals_yet_is_not_graded(tmp_path):
    s = _store(tmp_path)
    g = GR.grade(s.gradable(), _actuals().filter(pl.col("market") != "spread").filter(pl.col("market") != "moneyline"), tmp_path / "graded.parquet", graded_at=TUE)
    assert sorted(g["market"].unique().to_list()) == ["rush_att"] and g.height == 2                       # the two game markets wait for their result


def test_the_summary_reports_did_not_play_rows_separately(tmp_path):
    s = _store(tmp_path)
    g = GR.grade(s.gradable(), _actuals(), tmp_path / "graded.parquet", graded_at=TUE)
    t = GR.summary(g).filter(pl.col("market") == "rush_att").row(0, named=True)
    assert (t["n_played"], t["n_did_not_play"], t["mae_played"]) == (1, 1, 3.0) and t["mae_did_not_play"] == 8.0


def test_summaries_are_reported_per_stack_label_and_never_blended(tmp_path):
    s = ST.PredictionStore(tmp_path / "preds")
    rows = [("g1", "p1", "rush_att", 12.0, "phase3"), ("g2", "p3", "rush_att", 10.0, "phase3+4a")]
    s.write(pl.DataFrame([dict(game_id=g, player_id=p, market=m, kickoff_time=KICK, model_version="mv1", stack=st, mean=v, input_hash="h") for g, p, m, v, st in rows]),
            now=T0, run_id="r1")
    act = pl.DataFrame({"game_id": ["g1", "g2"], "player_id": ["p1", "p3"], "market": ["rush_att", "rush_att"], "actual": [15.0, 10.0]})
    g = GR.grade(s.gradable(), act, tmp_path / "graded.parquet", graded_at=TUE)
    t = GR.summary(g).sort("stack")
    assert t["stack"].to_list() == ["phase3", "phase3+4a"] and t["market"].to_list() == ["rush_att", "rush_att"] and t["mae_played"].to_list() == [3.0, 0.0]        # two rows, never one blended average
