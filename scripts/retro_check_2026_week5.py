"""A retrospective check, NOT a track record: the live pipeline run as of 2026-10-07 13:00 UTC (the data then available, full stack) for the week-5 games of 2026, graded against the
results pulled afterwards. The predictions were made after the fact (nothing was stored before kickoff), so this cannot count toward the prospective record or any gate; it only says whether
the live path, which has never seen a 2026 outcome in any fit, behaves sensibly on real unseen games.

  python scripts/retro_check_2026_week5.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

import config
from eval import baselines as bl
from live import history as H
from live import weekly as W

OUT = config.ROOT / "retro_check_2026_week5.parquet"


def actuals(season: int, week: int) -> pl.DataFrame:
    """(game_id, player_id, market, actual) for eligible-or-not players who played, and game markets, from the data pulled after the games."""
    with config.input_history(season):
        plog = bl.build_player_game_log(config.RAW_DUCKDB_PATH, max_season=season).filter((pl.col("season") == season) & (pl.col("week") == week))
        tlog = bl.build_team_game_log(config.RAW_DUCKDB_PATH, max_season=season).filter((pl.col("season") == season) & (pl.col("week") == week))
    rows = []
    for m, col in bl.PLAYER_MARKETS.items():
        rows.append(plog.select("game_id", "player_id", market=pl.lit(m), actual=pl.col(col).cast(pl.Float64)))
    pa = pl.concat(rows)
    home = tlog.filter(pl.col("is_home")).select("game_id", hp="pf", ap="pa")
    g = pl.concat([home.select("game_id", market=pl.lit("spread"), actual=(pl.col("hp") - pl.col("ap")).cast(pl.Float64), player_id=pl.lit(None, dtype=pl.String)),
                   home.select("game_id", market=pl.lit("total"), actual=(pl.col("hp") + pl.col("ap")).cast(pl.Float64), player_id=pl.lit(None, dtype=pl.String)),
                   home.filter(pl.col("hp") != pl.col("ap")).select("game_id", market=pl.lit("moneyline"), actual=(pl.col("hp") > pl.col("ap")).cast(pl.Float64), player_id=pl.lit(None, dtype=pl.String))])
    return pl.concat([pa, g.select(pa.columns)])


def baseline_trailing(preds: pl.DataFrame, season: int, cutoff) -> pl.DataFrame:
    """A transparent yardstick: each player's mean of his last 4 games played (any season) for the market's stat; games: the league's mean margin 0 / mean total."""
    with config.input_history(season):
        plog = bl.build_player_game_log(config.RAW_DUCKDB_PATH, max_season=season).filter(pl.col("gameday") < cutoff)
    out = []
    for m, col in bl.PLAYER_MARKETS.items():
        t = (plog.sort("player_id", "gameday").group_by("player_id", maintain_order=True).agg(pl.col(col).cast(pl.Float64).tail(4).mean().alias("trail4")))
        out.append(t.select("player_id", market=pl.lit(m), trail4="trail4"))
    return preds.join(pl.concat(out), on=["player_id", "market"], how="left")


def main():
    now = datetime(2026, 10, 7, 13, 0, tzinfo=timezone.utc)
    res = W.run(now, store_dir=Path("/tmp/retro_store"), version="retro", work_dir=Path("/tmp/retro_work"))
    preds, season, week = res["predictions"], res["season"], res["week"]
    act = actuals(season, week)
    cutoff = preds["kickoff_time"].min().date()
    j = preds.join(act, on=["game_id", "player_id", "market"], how="left", nulls_equal=True)
    played_games = set(act["game_id"].unique().to_list())
    j = j.filter(pl.col("game_id").is_in(played_games))
    is_game = pl.col("market").is_in(["spread", "total", "moneyline"])
    j = j.with_columns(did_not_play=pl.col("actual").is_null() & ~is_game, actual=pl.when(pl.col("actual").is_null() & ~is_game).then(0.0).otherwise(pl.col("actual")))
    j = baseline_trailing(j, season, cutoff).with_columns(err=(pl.col("mean") - pl.col("actual")).abs(), err_trail4=(pl.col("trail4") - pl.col("actual")).abs())
    j.write_parquet(OUT)
    played = j.filter(~pl.col("did_not_play") & pl.col("actual").is_not_null())
    t = played.group_by("stack", "market").agg(n=pl.len(), mae=pl.col("err").mean(), mae_trail4=pl.col("err_trail4").mean(), bias=(pl.col("mean") - pl.col("actual")).mean()).sort("market")
    dn = j.filter(pl.col("did_not_play")).group_by("market").agg(n_dnp=pl.len(), mean_pred=pl.col("mean").mean()).sort("market")
    with pl.Config(tbl_rows=30, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
        print(f"week {week} {season}: {len(played_games)} games with results, {j.height} graded predictions")
        print(t)
        print("eligible players who did not play (excluded above):")
        print(dn)
    return j


if __name__ == "__main__":
    main()
