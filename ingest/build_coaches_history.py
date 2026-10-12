"""4d.0 coaches_history, option 3 (your decision of 2026-10-12; docs/overnight_plan.md "4d decisions").

One row per (season, team, role) for 2016-2026, roles head_coach / offensive_coordinator / play_caller:
  2016-2024  head_coach from nflverse schedules (home_coach / away_coach, per game): the coach of the most games that season; every coach of the
             season with his game count is in `note`, and the per-game table is coaches_history_games (an interim change is never lost).
             offensive_coordinator and play_caller: UNKNOWN.
  2025       the holdout season: never loaded, every role UNKNOWN.
  2026       the hand-built 2026 staff, data/coaches_seed.csv (head_coach, offensive_coordinator, offensive_play_caller -> play_caller).
UNKNOWN rows have name = null (never "" or 0) and status "UNKNOWN": a missing play caller carries no information downstream.
Teams are the franchise's current code (OAK -> LV, SD -> LAC, STL -> LA); team_code_that_season keeps the season's own code.

  python -m ingest.build_coaches_history         writes data/coaches_history.csv and the DuckDB tables coaches_history / coaches_history_games,
                                                 and prints the UNKNOWN count per season and role
"""
from __future__ import annotations

import duckdb
import polars as pl

import config

FIRST_SEASON, LAST_HISTORY_SEASON, LIVE_SEASON = 2016, 2024, 2026
SEASONS = tuple(range(FIRST_SEASON, LIVE_SEASON + 1))
ROLES = ("head_coach", "offensive_coordinator", "play_caller")
SEED_ROLE = {"head_coach": "head_coach", "offensive_coordinator": "offensive_coordinator", "offensive_play_caller": "play_caller"}
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
CSV_PATH = config.DATA_DIR / "coaches_history.csv"
SCHEMA = {"season": pl.Int64, "team": pl.String, "role": pl.String, "name": pl.String, "status": pl.String, "source": pl.String,
          "n_games": pl.Int64, "team_code_that_season": pl.String, "note": pl.String}


def load_head_coach_games(raw_db=config.RAW_DUCKDB_PATH, first: int = FIRST_SEASON, last: int = LAST_HISTORY_SEASON) -> pl.DataFrame:
    """Per regular-season team-game: (season, week, game_id, team, head_coach) from nflverse schedules, first..last; never the holdout."""
    if last >= config.HOLDOUT_SEASON:
        raise config.HoldoutError(f"head coaches through {last} would read the holdout season {config.HOLDOUT_SEASON}")
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        q = ("SELECT season, week, game_id, {t} AS team, {c} AS head_coach FROM "
             "(SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC) "
             "WHERE game_type = 'REG' AND season BETWEEN ? AND ?")
        d = pl.concat([con.execute(q.format(t=t, c=c), [first, last]).pl() for t, c in (("home_team", "home_coach"), ("away_team", "away_coach"))])
    finally:
        con.close()
    return d.sort("season", "week", "game_id", "team")


def history(games: pl.DataFrame, seed: pl.DataFrame, seasons=SEASONS, teams=None) -> pl.DataFrame:
    """The coaches_history table from the per-game head coaches (`games`) and the 2026 seed."""
    g = games.with_columns(franchise=pl.col("team").replace(FRANCHISE))
    teams = tuple(sorted(teams or set(g["franchise"].unique().to_list()) | set(config.TEAM_NAME_TO_ABBR.values())))
    hc = (g.group_by("season", "franchise", "head_coach").agg(n=pl.len(), first_week=pl.col("week").min(), code=pl.col("team").first())
          .sort("season", "franchise", "n", "first_week", descending=[False, False, True, False]))
    hc_rows = {}
    for (season, team), grp in hc.group_by("season", "franchise", maintain_order=True):
        top = grp.row(0, named=True)
        others = [f"{r['head_coach']} ({r['n']})" for r in grp.iter_rows(named=True)]
        hc_rows[(season, team)] = (top["head_coach"], top["n"], top["code"], "; ".join(others) if len(others) > 1 else "")
    s = seed.with_columns(team=pl.col("team").replace_strict(config.TEAM_NAME_TO_ABBR)).filter(pl.col("role").is_in(list(SEED_ROLE)))
    seed_rows = {(r["team"], SEED_ROLE[r["role"]]): r for r in s.iter_rows(named=True)}
    rows = []
    for season in seasons:
        for team in teams:
            for role in ROLES:
                r = dict(season=season, team=team, role=role, name=None, status="UNKNOWN", source=None, n_games=None,
                         team_code_that_season=None, note="")
                if season >= config.HOLDOUT_SEASON and season < LIVE_SEASON:
                    r["note"] = "holdout season: not loaded"
                elif season == LIVE_SEASON:
                    sr = seed_rows.get((team, role))
                    if sr is not None and sr["name"]:
                        r.update(name=sr["name"], status="known", source="coaches_seed.csv", team_code_that_season=team,
                                 note=f"confidence {sr.get('confidence')}; {sr.get('source_url') or ''}".strip())
                elif role == "head_coach" and (season, team) in hc_rows:
                    name, n, code, note = hc_rows[(season, team)]
                    r.update(name=name, status="known", source="nflverse schedules", n_games=n, team_code_that_season=code,
                             note=(f"head coaches this season: {note}" if note else ""))
                elif role != "head_coach":
                    r["note"] = "not in any approved source (option 3)"
                rows.append(r)
    return pl.DataFrame(rows, schema=SCHEMA)


def lookup(h: pl.DataFrame, season: int, team: str, role: str):
    """The coach in a role for a team-season, or None when UNKNOWN (no information, never a placeholder)."""
    r = h.filter((pl.col("season") == season) & (pl.col("team") == team) & (pl.col("role") == role))
    return r["name"][0] if r.height and r["status"][0] == "known" else None


def build(raw_db=config.RAW_DUCKDB_PATH, db_path=config.DUCKDB_PATH, csv_path=CSV_PATH, seed_csv=config.DATA_DIR / "coaches_seed.csv") -> pl.DataFrame:
    games = load_head_coach_games(raw_db)
    h = history(games, pl.read_csv(seed_csv, infer_schema_length=0))
    h.write_csv(csv_path)
    con = duckdb.connect(str(db_path))
    try:
        con.register("h", h.to_arrow())
        con.register("g", games.with_columns(franchise=pl.col("team").replace(FRANCHISE)).to_arrow())
        con.execute("CREATE OR REPLACE TABLE coaches_history AS SELECT * FROM h")
        con.execute("CREATE OR REPLACE TABLE coaches_history_games AS SELECT * FROM g")
    finally:
        con.close()
    return h


if __name__ == "__main__":
    h = build()
    with pl.Config(tbl_rows=40):
        print(h.group_by("season", "role").agg(unknown=(pl.col("status") == "UNKNOWN").sum(), rows=pl.len())
              .pivot(on="role", index="season", values="unknown").sort("season"))
    print(f"{h.height} rows written to {CSV_PATH.name} and coaches_history; holdout {config.HOLDOUT_SEASON} not loaded")
