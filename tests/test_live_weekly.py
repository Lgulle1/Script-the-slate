"""The weekly run end to end on the stored 2026-10-07 pull (live/weekly.py): plan P5.1-P5.3, P5.6."""
import json
from datetime import datetime, timedelta, timezone

import duckdb
import polars as pl
import pytest

import config
from live import store as ST
from live import weekly as W

needs_db = pytest.mark.skipif(not config.RAW_DUCKDB_PATH.exists(), reason="no local raw db")
NOW = datetime(2026, 10, 7, 13, 0, tzinfo=timezone.utc)                 # after the 12:28 UTC pull, before week 5's Thursday game


@pytest.fixture(scope="module")
def models():
    return W.train_models()


@needs_db
def test_a_run_before_the_week_stores_every_game_and_a_manifest(models, tmp_path):
    out = W.run(NOW, store_dir=tmp_path / "s", models=models, lines_path=tmp_path / "lines.csv", version="mv-test")
    assert out["status"] == "ok" and (out["season"], out["week"]) == (2026, 5) and out["refused"] == 0
    t = ST.PredictionStore(tmp_path / "s").load()
    assert t.height == out["written"] > 1000 and t["market"].n_unique() == 13 and t["mean"].null_count() == 0
    assert (t["stored_at"] < t["kickoff_time"]).all() and set(t["model_version"]) == {"mv-test"}
    assert t.filter(pl.col("player_id").is_null())["market"].is_in(["spread", "moneyline", "total"]).all()
    m = json.loads(next((tmp_path / "s").glob("manifest_*.json")).read_text())
    assert m["history"]["newest_game"] < m["cutoff"] and m["chart"]["teams"] == 2 * m["n_games"] and m["written"] == out["written"]
    assert pl.read_csv(tmp_path / "lines.csv").height == 15                                                 # the lines went to their own log


@needs_db
def test_an_identical_rerun_writes_nothing_and_a_started_game_is_refused_and_logged(models, tmp_path):
    s = tmp_path / "s"
    first = W.run(NOW, store_dir=s, models=models, version="mv-test", run_id="r1")
    again = W.run(NOW + timedelta(minutes=5), store_dir=s, models=models, version="mv-test", run_id="r2")
    assert again["written"] == 0 and again["unchanged"] > 1000
    late = W.run(datetime(2026, 10, 9, 13, 0, tzinfo=timezone.utc), store_dir=s, models=models, version="mv-test", run_id="r3")       # the Thursday game (9 Oct 00:15 UTC) is over
    assert late["refused"] > 50
    log = pl.read_csv(s / "refusals.csv")
    assert set(log["game_id"]) == {"2026_05_TB_DAL"} and log["reason"].str.contains("kickoff").all()
    t = ST.PredictionStore(s).load()
    assert t.filter(pl.col("game_id") == "2026_05_TB_DAL")["version"].max() == 1                            # nothing new for the started game
    assert first["written"] == t.height - late["written"]


@needs_db
def test_nothing_stamped_after_the_clock_reaches_a_run(models, tmp_path):
    """A run at 11:00 UTC on 7 Oct, before the 12:19 UTC schedule pull and the 12:28 chart pull, finds nothing: there is no as-of data yet."""
    out = W.run(datetime(2026, 10, 7, 11, 0, tzinfo=timezone.utc), store_dir=tmp_path / "s", models=models)
    assert out["status"] == "no upcoming game" and out["written"] == 0
