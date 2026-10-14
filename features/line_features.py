"""Book game lines as features: the INFORMED comparison model (docs/overnight_plan.md, "Improvement plan from weeks 1-5", workstream C).

This is the only module that feeds a sportsbook line into a model, and it is used only by the separate "informed" stack and its ablation; the pure Phase 3 / 4a / 4b path never imports it
(tests/test_line_features.py enforces that). Columns, per (game, team), from the schedule table's total_line and spread_line (positive spread_line = home favoured):
  ln_total          the game's total
  ln_team_spread    the team's own spread (positive = the team is favoured)
  ln_team_pts       the team's implied points, (total + team spread) / 2
  ln_opp_pts        the opponent's implied points, (total - team spread) / 2
A missing line gives null columns, never zero. Seasons are capped at the last development season; the holdout raises.
"""
from __future__ import annotations

import duckdb
import polars as pl

import config

LN_COLUMNS = ["ln_total", "ln_team_spread", "ln_team_pts", "ln_opp_pts"]


def load_schedule_lines(raw_db=config.RAW_DUCKDB_PATH, first: int = 2020, last: int = max(config.BACKTEST_SEASONS)) -> pl.DataFrame:
    if last >= config.HOLDOUT_SEASON:
        raise config.HoldoutError(f"lines through {last} would read the holdout season {config.HOLDOUT_SEASON}")
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        return con.execute("SELECT game_id, home_team, away_team, spread_line, total_line FROM (SELECT DISTINCT ON (game_id) * FROM schedules WHERE game_type = 'REG' ORDER BY game_id, pulled_at DESC) "
                           "WHERE season BETWEEN ? AND ? ORDER BY game_id", [first, last]).pl()
    finally:
        con.close()


def features(sched: pl.DataFrame) -> pl.DataFrame:
    """sched: game_id, home_team, away_team, spread_line, total_line -> one row per (game_id, team)."""
    if sched["game_id"].is_duplicated().any():
        raise ValueError("more than one schedule row for a game")
    s = sched.with_columns(spread_line=pl.col("spread_line").cast(pl.Float64), total_line=pl.col("total_line").cast(pl.Float64))
    sides = [s.select("game_id", team="home_team", spread=pl.col("spread_line"), total=pl.col("total_line")),
             s.select("game_id", team="away_team", spread=-pl.col("spread_line"), total=pl.col("total_line"))]
    f = pl.concat(sides).with_columns(ln_total=pl.col("total"), ln_team_spread=pl.col("spread"), ln_team_pts=(pl.col("total") + pl.col("spread")) / 2, ln_opp_pts=(pl.col("total") - pl.col("spread")) / 2)
    return f.select("game_id", "team", *LN_COLUMNS).sort("game_id", "team")


def attach(tables, feats: pl.DataFrame):
    """FeatureTables with the four columns joined on (game_id, team), for players and teams."""
    from features.volume_features import FeatureTables
    if feats.select("game_id", "team").is_duplicated().any():
        raise ValueError("more than one line row for a (game, team)")
    missing = [c for c in LN_COLUMNS if c not in feats.columns]
    if missing:
        raise KeyError(f"line features are missing columns {missing}")
    f = feats.select("game_id", "team", *LN_COLUMNS)
    players = {q: df.join(f, on=["game_id", "team"], how="left", maintain_order="left") for q, df in tables.players.items()}
    return FeatureTables(players, tables.teams.join(f, on=["game_id", "team"], how="left", maintain_order="left"))
