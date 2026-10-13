"""What happened: actual results for grading, and the best simple baseline on the same rows (plan Phase 5.2).

Outcome-side reading only (live.history.read_played_logs: the one reader that may open 2025+ data); never a model input.
"""
from __future__ import annotations

import polars as pl

import config
from eval import baselines as bl
from live import history as H

# The best simple baseline per market by 2020-2024 mean absolute error (Phase 2 methods; recorded in docs/overnight_plan.md).
BEST_BASELINE = {"pass_att": "blend_70_30", "pass_cmp": "blend_70_30", "pass_yds": "blend_70_30", "qb_rush_att": "blend_70_30", "qb_rush_yds": "blend_70_30",
                 "rush_att": "recency", "rush_yds": "recency", "targets": "recency", "rec": "recency", "rec_yds": "recency"}


def played(season: int, raw_db=config.RAW_DUCKDB_PATH):
    """(actuals long frame game_id, player_id, market, actual; player log; team log) for every game of `season` the raw database holds a result for."""
    plog, tlog = H.read_played_logs(season, raw_db)
    plog, tlog = plog.filter(pl.col("season") == season), tlog.filter(pl.col("season") == season)
    rows = [plog.select("game_id", "player_id", market=pl.lit(m), actual=pl.col(col).cast(pl.Float64)) for m, col in bl.PLAYER_MARKETS.items()]
    pa = pl.concat(rows)
    home = tlog.filter(pl.col("is_home")).select("game_id", hp="pf", ap="pa")
    none = pl.lit(None, dtype=pl.String)
    g = pl.concat([home.select("game_id", market=pl.lit("spread"), actual=(pl.col("hp") - pl.col("ap")).cast(pl.Float64), player_id=none),
                   home.select("game_id", market=pl.lit("total"), actual=(pl.col("hp") + pl.col("ap")).cast(pl.Float64), player_id=none),
                   home.filter(pl.col("hp") != pl.col("ap")).select("game_id", market=pl.lit("moneyline"), actual=(pl.col("hp") > pl.col("ap")).cast(pl.Float64), player_id=none)])
    return pl.concat([pa, g.select(pa.columns)]), plog, tlog


def baseline_mae(graded: pl.DataFrame, plog: pl.DataFrame, history: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per market: the best simple baseline's mean absolute error on the played rows of `graded` (did-not-play rows have no baseline target and are left out of both sides).

    `history` is the player log the baseline may read (default `plog`; each baseline only reads games before the target's game date)."""
    history = plog if history is None else history
    by_game = {(r["player_id"], r["game_id"]): r for r in plog.iter_rows(named=True)}
    out = []
    rows = graded.filter(pl.col("player_id").is_not_null() & ~pl.col("did_not_play") & pl.col("market").is_in(list(BEST_BASELINE)))
    for r in rows.iter_rows(named=True):
        t = by_game.get((r["player_id"], r["game_id"]))
        if t is None:
            continue
        target = bl.PlayerTarget(player_id=t["player_id"], gameday=t["gameday"], season=t["season"], team_game_num=t["team_game_num"], opponent=t["opponent"],
                                 family=t["family"], slot=t["slot"], team=t["team"], week=t["week"])
        pred = bl.predict_player(BEST_BASELINE[r["market"]], history, target)[r["market"]]
        if pred is not None:
            out.append(dict(stack=r["stack"], market=r["market"], model_abs=abs(r["mean"] - r["actual"]), baseline_abs=abs(pred - r["actual"])))
    if not out:
        return pl.DataFrame(schema={"stack": pl.String, "market": pl.String, "n_compared": pl.UInt32, "model_mae": pl.Float64, "baseline_mae": pl.Float64})
    return (pl.DataFrame(out).group_by("stack", "market").agg(n_compared=pl.len(), model_mae=pl.col("model_abs").mean(), baseline_mae=pl.col("baseline_abs").mean()).sort("stack", "market"))
