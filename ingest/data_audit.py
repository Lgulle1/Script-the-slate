"""Data freshness audit. Re-run every week: `python -m ingest.data_audit`.

For every table in the raw DuckDB (plus the master `players` table) report:
  * row count per season (all pulls included, so repeated pulls inflate counts;
    n_pulls / latest_pulled_at show how many pulls there have been and how fresh),
  * max week present per season,
  * percent filled for a hand-picked list of important columns.
Writes data/data_audit.json.
"""
import json
from datetime import datetime, timezone

import duckdb

import config

AUDIT_PATH = config.DATA_DIR / "data_audit.json"

# table -> columns whose fill rate matters. Edit as features start depending on more.
IMPORTANT_COLUMNS = {
    "schedules": ["game_id", "gameday", "gametime", "home_team", "away_team", "roof", "stadium_id",
                  "home_score", "away_score", "home_qb_id", "away_qb_id", "referee"],
    "pbp": ["game_id", "play_id", "posteam", "defteam", "epa", "wp", "yardline_100", "down", "ydstogo",
            "play_type", "passer_player_id", "rusher_player_id", "receiver_player_id"],
    "player_stats": ["player_id", "team", "position", "passing_yards", "rushing_yards", "receiving_yards",
                     "targets", "fantasy_points_ppr"],
    "snap_counts": ["pfr_player_id", "game_id", "team", "offense_snaps", "offense_pct", "defense_snaps"],
    "rosters_weekly": ["gsis_id", "team", "position", "status", "depth_chart_position", "pfr_id", "espn_id"],
    "injuries": ["gsis_id", "team", "report_status", "report_primary_injury", "practice_status", "date_modified"],
    "depth_charts": ["gsis_id", "club_code", "depth_team", "pos_abb", "formation", "dt"],
    "ftn_charting": ["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_screen_pass", "n_blitzers"],
    "nextgen_stats_passing": ["player_gsis_id", "avg_time_to_throw", "completion_percentage_above_expectation"],
    "nextgen_stats_receiving": ["player_gsis_id", "avg_separation", "avg_cushion"],
    "nextgen_stats_rushing": ["player_gsis_id", "efficiency", "rush_yards_over_expected"],
    "pfr_advstats_pass": ["pfr_player_id", "game_id", "passing_drops", "passing_bad_throws"],
    "pfr_advstats_rec": ["pfr_player_id", "game_id", "receiving_drop", "receiving_broken_tackles"],
    "pfr_advstats_rush": ["pfr_player_id", "game_id", "rushing_yards_before_contact", "rushing_broken_tackles"],
    "ff_opportunity": ["game_id", "player_id", "total_fantasy_points_exp", "pass_completions_exp", "rush_yards_gained_exp"],
    "officials": ["game_id", "official_id", "official_name", "position"],
    "participation": ["nflverse_game_id", "play_id", "offense_players", "defense_players", "offense_personnel",
                      "defense_personnel"],
    "weather": ["game_id", "temp_f", "wind_mph", "precip_in", "source"],
    "players": ["gsis_id", "display_name", "position", "pfr_id", "espn_id", "birth_date"],
}

# Tables with no season column: derive it from a game-id column ("2023_01_...").
SEASON_FROM_GAME_ID = {"participation": "nflverse_game_id", "weather": "game_id"}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def audit_table(con, table: str, important: list[str]) -> dict:
    cols = {r[0] for r in con.execute(f"DESCRIBE {_q(table)}").fetchall()}
    n = con.execute(f"SELECT count(*) FROM {_q(table)}").fetchone()[0]
    out: dict = {"rows": n}

    if "pulled_at" in cols:
        n_pulls, latest = con.execute(
            f"SELECT count(DISTINCT pulled_at), max(pulled_at) FROM {_q(table)}"
        ).fetchone()
        out["n_pulls"] = n_pulls
        out["latest_pulled_at"] = latest.isoformat() if latest else None

    season_expr = None
    if "season" in cols:
        season_expr = "TRY_CAST(season AS INTEGER)"
    elif table in SEASON_FROM_GAME_ID and SEASON_FROM_GAME_ID[table] in cols:
        season_expr = f"TRY_CAST(left({_q(SEASON_FROM_GAME_ID[table])}, 4) AS INTEGER)"
    if season_expr:
        week = "max(TRY_CAST(week AS INTEGER))" if "week" in cols else "NULL"
        rows = con.execute(
            f"SELECT {season_expr} s, count(*), {week} FROM {_q(table)} GROUP BY s ORDER BY s"
        ).fetchall()
        out["by_season"] = {
            str(s): {"rows": c, **({"max_week": w} if w is not None else {})} for s, c, w in rows
        }

    present = [c for c in important if c in cols]
    absent = [c for c in important if c not in cols]
    if present and n:
        exprs = ", ".join(f"count({_q(c)}) * 100.0 / count(*)" for c in present)
        vals = con.execute(f"SELECT {exprs} FROM {_q(table)}").fetchone()
        out["pct_filled"] = {c: round(v, 2) for c, v in zip(present, vals)}
    if absent:
        out["columns_not_found"] = absent  # visible, so a renamed upstream column doesn't hide
    return out


def run_audit(raw_db=config.RAW_DUCKDB_PATH, main_db=config.DUCKDB_PATH, out_path=AUDIT_PATH) -> dict:
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "tables": {}}
    for db, only in ((raw_db, None), (main_db, {"players"})):
        if not db.exists():
            continue
        con = duckdb.connect(str(db), read_only=True)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' ORDER BY 1").fetchall()]  # views (weather_observed, players_current) just repeat a table
            for t in tables:
                if only is None or t in only:
                    report["tables"][t] = audit_table(con, t, IMPORTANT_COLUMNS.get(t, []))
        finally:
            con.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = run_audit()
    print(f"audited {len(rep['tables'])} tables -> {AUDIT_PATH}")
    for t, info in rep["tables"].items():
        if info.get("columns_not_found"):
            print(f"  {t}: columns not found: {info['columns_not_found']}")
