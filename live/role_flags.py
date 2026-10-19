"""Who is missing around a pick (yours, 2026-10-19: Allen's card was built with Hall's status blank).

teammate_flags(season, week) lists, per team, the QB / RB / WR / TE who regularly play (3-game average of at least 35% of offensive snaps) and are Out, Doubtful, or did not practice with
no game status yet. "Did not practice, no status" is treated as at least doubtful: he will not get the reps. The card shows these next to every pick on the team, so a pick whose
role depends on a missing teammate is never shown without it.

data_age(season, week) is how old the injury report in the database is, so a card is never built on a report from days ago without saying so.
"""
from __future__ import annotations

from datetime import datetime, timezone

import duckdb

import config

OFFENSE = ("QB", "RB", "WR", "TE")
MIN_SNAP_PCT = 35.0


def _con(raw_db=None):
    return duckdb.connect(str(raw_db or config.RAW_DUCKDB_PATH), read_only=True)


def data_age(season: int, week: int, raw_db=None) -> dict:
    con = _con(raw_db)
    pulled = con.execute("SELECT max(pulled_at) FROM injuries WHERE season = ? AND week = ?", [season, week]).fetchone()[0]
    depth = con.execute("SELECT max(pulled_at) FROM depth_charts WHERE season = ?", [season]).fetchone()[0]
    con.close()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return dict(injuries_pulled_utc=pulled.isoformat() if pulled else None, injuries_age_hours=(now - pulled).total_seconds() / 3600 if pulled else None,
                depth_pulled_utc=depth.isoformat() if depth else None, depth_age_hours=(now - depth).total_seconds() / 3600 if depth else None)


def teammate_flags(season: int, week: int, raw_db=None) -> dict:
    """{team: [dict(name, position, status, snap_pct)]} for regular offensive players flagged this week."""
    con = _con(raw_db)
    rows = con.execute("""
        WITH inj AS (
          SELECT DISTINCT ON (gsis_id) gsis_id, full_name, team, position, report_status, practice_status
          FROM injuries WHERE season = ? AND week = ? AND gsis_id IS NOT NULL ORDER BY gsis_id, pulled_at DESC),
        snaps AS (
          SELECT lower(player) AS nm, team, avg(offense_pct) AS pct FROM (
            SELECT DISTINCT ON (pfr_game_id, pfr_player_id) player, team, offense_pct FROM snap_counts
            WHERE season = ? AND week < ? AND game_type = 'REG' ORDER BY pfr_game_id, pfr_player_id, pulled_at DESC) GROUP BY 1, 2)
        SELECT i.full_name, i.team, i.position, i.report_status, i.practice_status, s.pct
        FROM inj i LEFT JOIN snaps s ON lower(i.full_name) = s.nm AND i.team = s.team
        WHERE i.position IN ('QB', 'RB', 'WR', 'TE')
          AND (i.report_status IN ('Out', 'Doubtful') OR (i.report_status IS NULL AND i.practice_status LIKE 'Did Not%'))
    """, [season, week, season, week]).fetchall()
    con.close()
    out: dict = {}
    for name, team, pos, status, practice, pct in rows:
        pct = float(pct) if pct is not None else 0.0
        # snap percentages are stored as fractions (0-1) in some pulls and percentages in others
        pct = pct * 100 if pct <= 1.0 else pct
        if pct < MIN_SNAP_PCT:
            continue
        out.setdefault(team, []).append(dict(name=name, position=pos, status=status or "did not practice, no game status", snap_pct=round(pct)))
    return out


def label(f: dict) -> str:
    return f"{f['name']} ({f['position']}): {f['status']}"
