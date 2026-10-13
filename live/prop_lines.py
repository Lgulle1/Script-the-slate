"""The player-prop line log (docs/overnight_plan.md, "Player-prop line logging"): sportsbook prop lines as the book showed them, in their OWN append-only table.

Read by nothing in the prediction path (tests/test_prop_lines.py enforces it). The only writer is `log_prop_lines`; the only first source is manual entry (a CSV).
Missing values are empty, never 0. A line captured at or after kickoff is refused; a player or market that cannot be matched is rejected with its reason, never guessed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PROP_COLUMNS = ["captured_at", "book", "game_id", "player_id", "player_name", "team", "market", "book_market", "side", "line", "price", "is_alt", "source", "note"]
REJECT_COLUMNS = ["captured_at", "book", "game_id", "player_name", "book_market", "side", "line", "reason"]
KEY = ["book", "game_id", "player_id", "market", "side", "line", "is_alt"]

# the book's market name (lower case, 'alt' marker removed) -> our market; QB rushing is told apart by position below
MARKET_NAMES = {
    "passing yds": "pass_yds", "passing yards": "pass_yds", "pass yds": "pass_yds", "player_pass_yds": "pass_yds",
    "passing attempts": "pass_att", "pass attempts": "pass_att", "pass att": "pass_att", "player_pass_attempts": "pass_att",
    "pass completions": "pass_cmp", "completions": "pass_cmp", "passing completions": "pass_cmp", "player_pass_completions": "pass_cmp",
    "rushing yds": "rush_yds", "rushing yards": "rush_yds", "rush yds": "rush_yds", "player_rush_yds": "rush_yds",
    "rushing attempts": "rush_att", "rush attempts": "rush_att", "carries": "rush_att", "rush att": "rush_att", "player_rush_attempts": "rush_att",
    "receptions": "rec", "player_receptions": "rec",
    "receiving yds": "rec_yds", "receiving yards": "rec_yds", "rec yds": "rec_yds", "player_reception_yds": "rec_yds",
    "targets": "targets", "receiving targets": "targets",
}
QB_MARKET = {"rush_yds": "qb_rush_yds", "rush_att": "qb_rush_att"}
SIDES = {"over": "over", "o": "over", "under": "under", "u": "under", "at_least": "at_least", "at least": "at_least", "+": "at_least", "atleast": "at_least"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


@dataclass
class Result:
    written: int
    unchanged: int
    refused: int


def normalize_name(name: str) -> str:
    """Lower case, no punctuation, no generational suffix, single spaces: 'Marvin Harrison Jr.' and 'marvin harrison' match."""
    parts = re.sub(r"[^a-z0-9 ]", "", str(name).lower().replace("-", " ")).split()
    return " ".join(p for p in parts if p not in SUFFIXES)


def map_market(book_market: str) -> tuple:
    """(our market or None, is_alt). 'Alt Passing Yds' -> ('pass_yds', True)."""
    raw = str(book_market).strip().lower()
    alt = bool(re.search(r"\balt(ernate)?\b", raw))
    raw = re.sub(r"\balt(ernate)?\b", "", raw)
    raw = re.sub(r"\s+", " ", re.sub(r"\b(player|total|o/u|over/under)\b", "", raw) if raw not in MARKET_NAMES else raw).strip(" -_")
    return MARKET_NAMES.get(raw), alt


def _num(x):
    if x is None or (isinstance(x, float) and x != x) or str(x).strip() == "":
        return None
    return float(str(x).replace("+", "")) if re.fullmatch(r"[+-]?\d+(\.\d+)?", str(x).strip()) else None


def _fmt(x) -> str:
    return "" if x is None else (str(int(x)) if float(x) == int(x) else str(float(x)))


def _last_logged(path: Path) -> dict:
    if not path.exists():
        return {}
    d = pl.read_csv(path, infer_schema_length=0)
    return {tuple(r[k] or "" for k in KEY): r["price"] or "" for r in d.iter_rows(named=True)}


def _append(path: Path, columns: list, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(columns)
        w.writerows(rows)


def log_prop_lines(rows: pl.DataFrame, captured_at: datetime, path, schedule: pl.DataFrame, players: pl.DataFrame, source: str = "manual", rejects_path=None) -> Result:
    """Append the lines of `rows` to the log at `path`.

    rows      book, player_name, book_market, side, line; optional price, note; and the game by `game_id`, or by `team` + `gameday` (resolved on `schedule`)
    schedule  game_id, home_team, away_team, gameday (date), kickoff_time (UTC datetime): live.targets.schedule_frame output
    players   gsis_id, display_name, latest_team, position (the player master table)
    """
    path = Path(path)
    rejects_path = Path(rejects_path) if rejects_path else path.with_name("prop_lines_rejects.csv")
    captured_at = captured_at.astimezone(timezone.utc)
    games = {r["game_id"]: r for r in schedule.iter_rows(named=True)}
    by_name: dict = {}
    for r in players.iter_rows(named=True):
        by_name.setdefault(normalize_name(r["display_name"]), []).append(r)
    last = _last_logged(path)
    out, rej = [], []
    unchanged = 0
    stamp = captured_at.isoformat()
    for r in rows.iter_rows(named=True):
        reject = lambda why: rej.append([stamp, r.get("book") or "", r.get("game_id") or "", r.get("player_name") or "", r.get("book_market") or "", r.get("side") or "", r.get("line") if r.get("line") is not None else "", why])
        gid = r.get("game_id")
        if not gid:
            hit = [g for g in games.values() if r.get("team") in (g["home_team"], g["away_team"]) and str(g["gameday"]) == str(r.get("gameday"))]
            if len(hit) != 1:
                reject("game not found from team + gameday" if not hit else "more than one game for team + gameday")
                continue
            gid = hit[0]["game_id"]
        g = games.get(gid)
        if g is None:
            reject("game_id not in the schedule")
            continue
        if captured_at >= g["kickoff_time"]:
            reject(f"captured at or after kickoff ({g['kickoff_time'].isoformat()})")
            continue
        cand = [p for p in by_name.get(normalize_name(r.get("player_name") or ""), []) if p["latest_team"] in (g["home_team"], g["away_team"])]
        if len(cand) != 1:
            reject("player not matched in the game's teams" if not cand else "more than one player matches")
            continue
        p = cand[0]
        market, is_alt = map_market(r.get("book_market") or "")
        if market is None:
            reject("book market not mapped")
            continue
        if p["position"] == "QB" and market in QB_MARKET:
            market = QB_MARKET[market]
        side = SIDES.get(str(r.get("side") or "").strip().lower())
        line = _num(r.get("line"))
        if side is None or line is None:
            reject("side or line not understood")
            continue
        price = _num(r.get("price"))
        row = [stamp, r.get("book") or "", gid, p["gsis_id"], r.get("player_name"), p["latest_team"], market, r.get("book_market"), side, _fmt(line),
               _fmt(price), str(bool(r.get("is_alt")) or is_alt).lower(), source, r.get("note") or ""]
        key = (row[1], gid, p["gsis_id"], market, side, _fmt(line), row[11])
        if last.get(key) == _fmt(price):
            unchanged += 1
            continue
        last[key] = _fmt(price)
        out.append(row)
    _append(path, PROP_COLUMNS, out)
    _append(rejects_path, REJECT_COLUMNS, rej)
    return Result(len(out), unchanged, len(rej))
