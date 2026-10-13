"""Read-only data access for the app (plan: "Phase 7 app and Phase 6.1 / 6.2"): stored records in, plain frames out. Nothing here recomputes a model or writes a prediction."""
from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

import config
from live import store as ST

QCOLS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
QPROB = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
GAME_MARKETS = ("spread", "moneyline", "total")


def records_dir() -> Path:
    return Path(os.environ.get("SLATE_RECORDS", config.ROOT / "records"))


def predictions() -> pl.DataFrame:
    """Every stored record (all versions)."""
    return ST.PredictionStore(records_dir() / "predictions").load()


def latest(preds: pl.DataFrame) -> pl.DataFrame:
    """The newest stored version of each (game, player, market)."""
    if preds.is_empty():
        return preds
    return preds.sort("version").group_by(ST.KEY, maintain_order=True).last().sort(ST.KEY)


def explain_tables() -> tuple:
    """(components, margin terms) from the newest stored run that has them (a game appears in many runs: the latest run per game wins)."""
    d = records_dir() / "predictions" / "explain"
    comp = [pl.read_parquet(f) for f in sorted(d.glob("run_*.parquet")) if not f.name.endswith("_margin.parquet")] if d.exists() else []
    marg = [pl.read_parquet(f) for f in sorted(d.glob("run_*_margin.parquet"))] if d.exists() else []
    newest = lambda t: (t.sort("run_id").group_by([c for c in t.columns if c in ("game_id", "player_id", "team", "stack", "quantity", "feature", "term", "side")], maintain_order=True).last()
                        if t.height else t)
    return (newest(pl.concat(comp, how="diagonal_relaxed")) if comp else pl.DataFrame(), newest(pl.concat(marg, how="diagonal_relaxed")) if marg else pl.DataFrame())


def graded() -> pl.DataFrame:
    f = records_dir() / "graded" / "graded_predictions.parquet"
    return pl.read_parquet(f) if f.exists() else pl.DataFrame()


def graded_summary() -> pl.DataFrame:
    f = records_dir() / "graded" / "graded_summary.csv"
    return pl.read_csv(f) if f.exists() else pl.DataFrame()


def prop_log() -> pl.DataFrame:
    f = records_dir() / "lines" / "prop_lines_log.csv"
    return pl.read_csv(f, infer_schema_length=0) if f.exists() else pl.DataFrame()


def factor_table() -> pl.DataFrame:
    from eval import factor_performance as FP
    return FP.build()


def game_parts(game_id: str) -> dict:
    """season, week, away, home from an id like 2026_05_BUF_LA."""
    s, w, a, h = game_id.split("_", 3)
    return dict(season=int(s), week=int(w), away=a, home=h)


def player_names(ids: list) -> dict:
    """gsis id -> display name from the player master table; the id itself when the table is not available."""
    names = {i: i for i in ids}
    try:
        import duckdb
        con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
        rows = con.execute("SELECT DISTINCT gsis_id, display_name FROM players WHERE gsis_id IN (SELECT UNNEST(?))", [[i for i in ids if i]]).fetchall()
        con.close()
        names.update({g: n for g, n in rows})
    except Exception:
        pass
    return names


def week_games(preds: pl.DataFrame, season: int, week: int) -> list:
    g = sorted({x for x in preds["game_id"].unique().to_list() if game_parts(x)["season"] == season and game_parts(x)["week"] == week})
    return g


# ----- display-only maths for the Lines log screen
def breakeven(price: float) -> float:
    """Break-even win probability of American odds."""
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def prob_at_least(row: dict, line: float):
    """(probability the real number is >= line, exact?) from a stored record: the ladder rung when `line` is one, else the stored quantiles interpolated linearly
    (a line outside q05..q95 is clamped and flagged inexact). None without a stored distribution."""
    if row.get("ladder"):
        lad = json.loads(row["ladder"])
        if str(int(line)) in lad and float(line) == int(line):
            return float(lad[str(int(line))]), True
    qs = [row.get(q) for q in QCOLS]
    if any(q is None for q in qs):
        return None
    xs, ps = list(qs), list(QPROB)
    if line <= xs[0]:
        return 0.95, False
    if line >= xs[-1]:
        return 0.05, False
    for i in range(len(xs) - 1):
        if xs[i] <= line <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                f = ps[i + 1]
            else:
                f = ps[i] + (ps[i + 1] - ps[i]) * (line - xs[i]) / (xs[i + 1] - xs[i])
            return 1.0 - f, True
    return None
