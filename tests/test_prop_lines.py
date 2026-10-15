"""The player-prop line log (live/prop_lines.py): rules of docs/overnight_plan.md, "Player-prop line logging"."""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from live import prop_lines as PL

ROOT = Path(__file__).resolve().parents[1]
KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)
T0 = KICK - timedelta(hours=20)

SCHED = pl.DataFrame([dict(game_id="2026_07_BUF_LA", home_team="LA", away_team="BUF", gameday=datetime(2026, 10, 18).date(), kickoff_time=KICK)],
                     schema_overrides={"kickoff_time": pl.Datetime("us", "UTC")})
PLAYERS = pl.DataFrame([
    dict(gsis_id="p_nacua", display_name="Puka Nacua", latest_team="LA", position="WR"),
    dict(gsis_id="p_allen", display_name="Josh Allen", latest_team="BUF", position="QB"),
    dict(gsis_id="p_allen_tb", display_name="Josh Allen", latest_team="TB", position="C"),
    dict(gsis_id="p_mh", display_name="Marvin Harrison", latest_team="LA", position="WR"),
    dict(gsis_id="p_twin1", display_name="Same Name", latest_team="LA", position="WR"),
    dict(gsis_id="p_twin2", display_name="Same Name", latest_team="BUF", position="WR")])


def _rows(*rows):
    base = dict(book="bookA", game_id="2026_07_BUF_LA", player_name="Puka Nacua", book_market="Receiving Yards", side="over", line=75.5, price=-115)
    return pl.DataFrame([{**base, **r} for r in rows], infer_schema_length=None)


def _log(tmp_path, rows, at=T0):
    return PL.log_prop_lines(rows, at, tmp_path / "prop_lines_log.csv", SCHED, PLAYERS)


def _read(tmp_path, name="prop_lines_log.csv"):
    return pl.read_csv(tmp_path / name, infer_schema_length=0)


def test_a_line_is_written_with_its_matched_player_and_our_market_name(tmp_path):
    r = _log(tmp_path, _rows({}))
    d = _read(tmp_path)
    assert (r.written, r.unchanged, r.refused) == (1, 0, 0)
    row = d.row(0, named=True)
    assert (row["player_id"], row["market"], row["side"], row["line"], row["price"], row["is_alt"], row["source"]) == ("p_nacua", "rec_yds", "over", "75.5", "-115", "false", "manual")
    assert list(d.columns) == PL.PROP_COLUMNS


def test_a_line_captured_at_or_after_kickoff_is_refused_and_logged_with_the_reason(tmp_path):
    for at in (KICK, KICK + timedelta(minutes=30)):
        r = _log(tmp_path, _rows({}), at=at)
        assert (r.written, r.refused) == (0, 1)
    rej = _read(tmp_path, "prop_lines_rejects.csv")
    assert rej.height == 2 and all("kickoff" in x for x in rej["reason"]) and not (tmp_path / "prop_lines_log.csv").exists()


def test_an_identical_recapture_is_not_written_again_but_a_changed_price_is(tmp_path):
    _log(tmp_path, _rows({}))
    r = _log(tmp_path, _rows({}), at=T0 + timedelta(hours=1))
    assert (r.written, r.unchanged) == (0, 1)
    r = _log(tmp_path, _rows({"price": -125}), at=T0 + timedelta(hours=2))
    assert (r.written, r.unchanged) == (1, 0)
    assert _read(tmp_path)["price"].to_list() == ["-115", "-125"]


def test_the_log_is_append_only_earlier_rows_never_change(tmp_path):
    _log(tmp_path, _rows({}))
    before = (tmp_path / "prop_lines_log.csv").read_text()
    _log(tmp_path, _rows({"line": 80.5}), at=T0 + timedelta(hours=1))
    assert (tmp_path / "prop_lines_log.csv").read_text().startswith(before)


def test_an_unknown_ambiguous_or_wrong_team_player_is_rejected_never_guessed(tmp_path):
    rows = _rows({"player_name": "Nobody Atall"}, {"player_name": "Same Name"}, {"player_name": "Josh Allen", "book_market": "Passing Yards", "line": 250.5})
    r = _log(tmp_path, rows)
    assert (r.written, r.refused) == (1, 2)                                   # Josh Allen of BUF matches; the TB namesake is not in this game
    reasons = _read(tmp_path, "prop_lines_rejects.csv")["reason"].to_list()
    assert any("not matched" in x for x in reasons) and any("more than one" in x for x in reasons)
    assert _read(tmp_path)["player_id"].to_list() == ["p_allen"]


def test_names_match_through_suffixes_punctuation_and_case(tmp_path):
    r = _log(tmp_path, _rows({"player_name": "MARVIN HARRISON JR."}))
    assert r.written == 1 and _read(tmp_path)["player_id"].to_list() == ["p_mh"]


def test_markets_are_mapped_by_name_alt_is_flagged_and_unknown_markets_are_rejected(tmp_path):
    rows = _rows({"book_market": "Alt Receptions", "side": "at_least", "line": 6}, {"book_market": "Anytime TD"}, {"book_market": "Receptions", "side": "O", "line": 6.5})
    r = _log(tmp_path, rows)
    d = _read(tmp_path)
    assert (r.written, r.refused) == (2, 1)
    assert d.filter(pl.col("market") == "rec").sort("line")["is_alt"].to_list() == ["true", "false"]
    assert "not mapped" in _read(tmp_path, "prop_lines_rejects.csv")["reason"][0]


def test_a_quarterbacks_rushing_prop_is_our_qb_rushing_market(tmp_path):
    _log(tmp_path, _rows({"player_name": "Josh Allen", "book_market": "Rushing Yards", "line": 28.5}, {"player_name": "Puka Nacua", "book_market": "Rushing Yards", "line": 5.5}))
    assert dict(zip(_read(tmp_path)["player_id"], _read(tmp_path)["market"])) == {"p_allen": "qb_rush_yds", "p_nacua": "rush_yds"}


def test_a_game_can_be_given_by_team_and_gameday_and_a_missing_price_stays_empty_not_zero(tmp_path):
    rows = _rows({"game_id": None, "team": "BUF", "gameday": "2026-10-18", "price": None})
    r = _log(tmp_path, rows)
    d = _read(tmp_path)
    assert r.written == 1 and d["game_id"].to_list() == ["2026_07_BUF_LA"] and d["price"].fill_null("").to_list() == [""]


def test_nothing_in_the_prediction_path_reads_the_prop_log():
    """The log is for a later after-the-fact comparison only: no model, feature, store, layer or run module may import it."""
    allowed = {"live/prop_lines.py"}
    users = []
    for d in ("live", "models", "features", "eval", "sim", "ingest"):
        for p in (ROOT / d).rglob("*.py"):
            if str(p.relative_to(ROOT)) not in allowed and re.search(r"prop_lines|prop_lines_log", p.read_text()):
                users.append(str(p.relative_to(ROOT)))
    for p in ROOT.glob("*.py"):
        if re.search(r"prop_lines", p.read_text()):
            users.append(p.name)
    assert users == [], users


def test_a_row_with_a_resolved_player_id_skips_name_matching(tmp_path):
    rows = _rows({"player_name": "typed any way", "player_id": "p_nacua"})
    r = _log(tmp_path, rows)
    assert r.written == 1 and _read(tmp_path)["player_id"].to_list() == ["p_nacua"]
    r2 = _log(tmp_path, _rows({"player_id": "no_such_id"}), at=T0 + timedelta(hours=1))
    assert r2.refused == 1


def test_a_row_carries_its_own_stamp_and_a_grace_accepts_a_just_after_kickoff_stamp_with_a_note(tmp_path):
    late = (KICK + timedelta(seconds=90)).isoformat()
    rows = _rows({"captured_at": (T0).isoformat(), "line": 70.5}, {"captured_at": late, "line": 75.5}, {"captured_at": (KICK + timedelta(seconds=900)).isoformat(), "line": 80.5})
    r = PL.log_prop_lines(rows, KICK + timedelta(hours=5), tmp_path / "prop_lines_log.csv", SCHED, PLAYERS, grace_seconds=300)
    assert (r.written, r.refused) == (2, 1)
    d = _read(tmp_path)
    assert d["captured_at"].to_list() == [T0.isoformat(), (KICK + timedelta(seconds=90)).isoformat()]
    assert d["note"].fill_null("").to_list() == ["", "stamp 90 s after kickoff"]
    assert _read(tmp_path, "prop_lines_rejects.csv").height == 1


def test_grace_default_refuses_and_the_grace_notes_the_late_stamp(tmp_path):
    late = (KICK + timedelta(seconds=90)).isoformat()
    PL.log_prop_lines(_rows({"captured_at": late}), T0, tmp_path / "a.csv", SCHED, PLAYERS)
    assert not (tmp_path / "a.csv").exists()
    r = PL.log_prop_lines(_rows({"captured_at": late}), T0, tmp_path / "b.csv", SCHED, PLAYERS, grace_seconds=300)
    assert r.written == 1 and "stamp 90 s after kickoff" in _read(tmp_path, "b.csv")["note"][0]


def test_a_row_without_its_stamp_in_a_stamped_batch_is_rejected_not_stamped_now(tmp_path):
    rows = _rows({"captured_at": T0.isoformat()}, {"captured_at": None, "line": 80.5})
    r = _log(tmp_path, rows)
    assert (r.written, r.refused) == (1, 1) and "no capture time" in _read(tmp_path, "prop_lines_rejects.csv")["reason"][0]


def test_a_roster_matches_the_name_inside_the_game_not_through_the_current_team(tmp_path):
    roster = pl.DataFrame([dict(game_id="2026_07_BUF_LA", gsis_id="p_old", display_name="Puka Nacua", team="LA", position="WR"),
                           dict(game_id="2026_07_BUF_LA", gsis_id="p_a", display_name="Twin Name", team="BUF", position="WR"), dict(game_id="2026_07_BUF_LA", gsis_id="p_b", display_name="Twin Name", team="LA", position="WR")])
    rows = _rows({"player_name": "Puka Nacua"}, {"player_name": "Twin Name"}, {"player_name": "Nobody Atall"})
    r = PL.log_prop_lines(rows, T0, tmp_path / "prop_lines_log.csv", SCHED, pl.DataFrame(schema=PLAYERS.schema), roster=roster)
    assert (r.written, r.refused) == (1, 2) and _read(tmp_path)["player_id"].to_list() == ["p_old"]
