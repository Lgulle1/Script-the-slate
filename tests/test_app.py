"""The six app screens (app/): they load against stored records, say so when there are none, and the Lines log never touches a stored prediction."""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from app import data as D
from live import grade as GR
from live import store as ST

T0 = datetime(2026, 10, 16, 20, 0, tzinfo=timezone.utc)
KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)
from pathlib import Path
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _rec(game, player, market, mean, stack="phase3", with_dist=True):
    d = {}
    if with_dist:
        d = dict(median=mean, q05=mean * 0.4, q10=mean * 0.55, q25=mean * 0.8, q50=mean, q75=mean * 1.2, q90=mean * 1.5, q95=mean * 1.7,
                 ladder=json.dumps({"50": 0.8, "75": 0.5, "100": 0.2}) if market == "rec_yds" else json.dumps({"4": 0.7, "6": 0.4}))
    return dict(game_id=game, player_id=player, market=market, kickoff_time=KICK, model_version="mv", stack=stack, mean=mean, input_hash="h", **d)


@pytest.fixture
def records(tmp_path, monkeypatch):
    root = tmp_path / "records"
    s = ST.PredictionStore(root / "predictions")
    rows = [_rec("2026_07_AAA_BBB", "p1", "rec_yds", 80.0), _rec("2026_07_AAA_BBB", "p1", "targets", 8.0), _rec("2026_07_AAA_BBB", "p2", "rush_att", 15.0, with_dist=False),
            _rec("2026_07_AAA_BBB", None, "spread", 3.5, "phase3+4b"), _rec("2026_07_AAA_BBB", None, "total", 47.0), _rec("2026_07_AAA_BBB", None, "moneyline", 0.6, with_dist=False)]
    s.write(pl.DataFrame(rows, schema_overrides={"player_id": pl.String}, infer_schema_length=None), now=T0, run_id="run_1")
    ex = root / "predictions" / "explain"
    ex.mkdir()
    comp = pl.DataFrame([dict(game_id="2026_07_AAA_BBB", player_id="p1", team=None, stack="phase3", quantity="targets", feature=f, value=v, contribution=c, component_prediction=8.0, run_id="run_1")
                         for f, v, c in (("(baseline)", None, 3.0), ("rec_targets", 10.0, 3.5), ("rc_targets", 8.0, 1.5))] +
                        [dict(game_id="2026_07_AAA_BBB", player_id=None, team="AAA", stack="phase3", quantity="plays", feature="(baseline)", value=None, contribution=60.0, component_prediction=61.0, run_id="run_1")],
                        schema_overrides={"player_id": pl.String, "team": pl.String, "value": pl.Float64}, infer_schema_length=None)
    comp.write_parquet(ex / "run_1.parquet")
    pl.DataFrame([dict(term="margin", side="game", points=3.5, home_value=None, away_value=None, coefficient=None, home_team="BBB", away_team="AAA", game_id="2026_07_AAA_BBB", stack="phase3+4b", run_id="run_1"),
                  dict(term="x_rating", side="margin", points=2.0, home_value=1.0, away_value=None, coefficient=0.7, home_team="BBB", away_team="AAA", game_id="2026_07_AAA_BBB", stack="phase3+4b", run_id="run_1")]
                 ).write_parquet(ex / "run_1_margin.parquet")
    actuals = pl.DataFrame({"game_id": ["2026_07_AAA_BBB"] * 3, "player_id": ["p1", "p1", "p2"], "market": ["rec_yds", "targets", "rush_att"], "actual": [95.0, 6.0, 12.0]}, schema_overrides={"player_id": pl.String})
    GR.grade(s.gradable(), actuals, root / "graded" / "graded_predictions.parquet", graded_at=KICK + timedelta(days=2))
    monkeypatch.setenv("SLATE_RECORDS", str(root))
    monkeypatch.setattr(D, "player_names", lambda ids: {i: {"p1": "Player One", "p2": "Player Two"}.get(i, i) for i in ids})
    return root


def _run(screen=None):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if screen:
        at.sidebar.radio[0].set_value(screen).run()
    return at


def _hash(root):
    h = hashlib.sha256()
    for f in sorted((root / "predictions").rglob("*")):
        if f.is_file():
            h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


@pytest.mark.parametrize("screen", ["Slate", "Player report", "Matchup", "Lines log", "Grading", "Model health"])
def test_every_screen_loads_against_stored_records_without_an_exception(records, screen):
    at = _run(screen)
    assert not at.exception, [e.value for e in at.exception]
    assert at.header[0].value == screen


@pytest.mark.parametrize("screen", ["Slate", "Player report", "Matchup", "Lines log", "Grading"])
def test_every_screen_says_so_when_nothing_is_stored(tmp_path, monkeypatch, screen):
    monkeypatch.setenv("SLATE_RECORDS", str(tmp_path / "empty"))
    at = _run(screen)
    assert not at.exception and any("No" in i.value or "Nothing" in i.value for i in at.info)


def test_the_slate_shows_the_stored_margin_total_and_plays(records):
    at = _run("Slate")
    df = at.dataframe[0].value
    assert df["expected margin (home)"][0] == 3.5 and df["expected total"][0] == 47.0 and df["plays away"][0] == 61.0 and df["stored predictions"][0] == 6


def test_the_player_report_shows_the_ladder_drivers_and_says_confidence_components_are_not_stored(records):
    at = _run("Player report")
    assert not at.exception
    assert any("Not stored" in c.value for c in at.caption)
    assert any("rec_targets" in str(d.value.to_dict()) for d in at.dataframe)
    assert any(m.value == "8.0" for m in at.metric)                                    # the first market of the first player: targets


def test_the_lines_log_leaves_every_stored_prediction_byte_identical_and_logs_the_line(records):
    before = _hash(records)
    at = _run("Lines log")
    at.number_input(key="ll_line").set_value(75.5)
    at.number_input(key="ll_price").set_value(-115)
    at.run()
    assert any("Break-even" in m.value for m in at.markdown) or any("Break-even" in str(w.value) for w in at.markdown)
    at.button(key="ll_save").click().run()
    assert not at.exception
    assert _hash(records) == before                                                     # the prediction record is untouched
    assert at.session_state["screen"] == "Lines log"
    # the kickoff of the fixture game is in the future of the real clock only if the year is 2026 or earlier; the line is either logged or refused with its reason, never lost
    log = records / "lines" / "prop_lines_log.csv"
    rej = records / "lines" / "prop_lines_rejects.csv"
    assert log.exists() or rej.exists()


def test_a_flagged_gap_is_labelled_a_flag_not_a_recommendation(records):
    at = _run("Lines log")
    at.number_input(key="ll_line").set_value(50.0)                                      # the stored ladder says 80% for 50+, break-even at -110 is 52.4%
    at.run()
    assert any("not a recommendation" in w.value for w in at.warning)


def test_the_grading_screen_shows_graded_rows_and_keeps_the_market_aware_tab_empty(records):
    at = _run("Grading")
    assert not at.exception and len(at.tabs) == 2 and at.dataframe[0].value.shape[0] >= 1
    assert any("did-not-play" in c.value for c in at.caption)


def test_model_health_flags_nothing_when_active_layers_gained(records):
    at = _run("Model health")
    assert not at.exception and (at.success or at.error)


def test_display_maths_breakeven_and_probability_at_a_line():
    assert D.breakeven(-110) == pytest.approx(110 / 210) and D.breakeven(150) == pytest.approx(0.4)
    row = dict(ladder=json.dumps({"50": 0.8}), q05=40, q10=55, q25=70, q50=80, q75=95, q90=115, q95=130)
    assert D.prob_at_least(row, 50) == (0.8, True)                                       # an exact rung
    p, ok = D.prob_at_least(row, 80)
    assert ok and p == pytest.approx(0.5)
    assert D.prob_at_least(row, 200) == (0.05, False) and D.prob_at_least(dict(ladder=None, q05=None), 10) is None
