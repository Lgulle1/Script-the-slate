"""What each player was expected to do, and what he did, every game 2020-2026 (the residuals every comp search averages).

expected = the comp-free projection stack made BEFORE the game:
  2020-2024  data/processed/proj_variants/injury_wf.parquet  (Phase 3 + 4a walk-forward, the live stack's recipe)
  2025       records/holdout_2025/predictions                 (stored before each week)
  2026       records/retro_2026/predictions (weeks 1-5) and records/predictions (live runs)
actual = the stat the market settles on (config.MARKETS[...]['stat']) from the played logs.
resid = actual - expected; rel = resid / max(expected, FLOOR[market]) (a percentage delta, the "Gibbs ran 48% over his expectation" number).
"""
from __future__ import annotations

import glob

import polars as pl

import config
from live import history as H

PLAYER_MARKETS = ("pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds", "targets", "rec", "rec_yds", "qb_rush_att", "qb_rush_yds")
FLOOR = dict(pass_att=10.0, pass_cmp=6.0, pass_yds=60.0, rush_att=3.0, rush_yds=12.0, targets=2.0, rec=1.5, rec_yds=15.0, qb_rush_att=2.0, qb_rush_yds=8.0)
OUT = config.PROCESSED_DIR / "comps2_expectations.parquet"


def _stored(pattern: str) -> pl.DataFrame:
    d = pl.concat([pl.read_parquet(f).select("game_id", "player_id", "market", "mean", "stored_at") for f in glob.glob(str(config.ROOT / pattern))], how="diagonal_relaxed")
    return d.filter(pl.col("player_id").is_not_null()).sort("stored_at").unique(["game_id", "player_id", "market"], keep="last").select("game_id", "player_id", "market", expected="mean")


def build() -> pl.DataFrame:
    log = H.read_played_logs(max(config.BACKTEST_SEASONS) + 2)[0]
    log = log.select("player_id", "season", "week", "game_id", "team", "opponent", "family", "slot", "gameday",
                     *{config.MARKETS[m]["stat"] for m in PLAYER_MARKETS})
    wf = pl.read_parquet(config.PROCESSED_DIR / "proj_variants" / "injury_wf.parquet").filter(pl.col("kind") == "player")
    wf = wf.select("market", "season", "week", player_id="entity", expected="prediction").join(
        log.select("player_id", "season", "week", "game_id"), on=["player_id", "season", "week"], how="inner").drop("season", "week")
    live = pl.concat([_stored("records/holdout_2025/predictions/*.parquet"), _stored("records/retro_2026/predictions/*.parquet"),
                      _stored("records/predictions/run_*.parquet")])
    live = live.filter(~pl.col("game_id").is_in(wf["game_id"].unique().implode()))
    e = pl.concat([wf.select("game_id", "player_id", "market", "expected"), live]).filter(pl.col("market").is_in(PLAYER_MARKETS))
    e = e.unique(["game_id", "player_id", "market"], keep="last")
    long = log.unpivot(index=["player_id", "season", "week", "game_id", "team", "opponent", "family", "slot", "gameday"],
                       on=sorted({config.MARKETS[m]["stat"] for m in PLAYER_MARKETS}), variable_name="stat", value_name="actual")
    stat_market = pl.DataFrame(dict(market=list(PLAYER_MARKETS), stat=[config.MARKETS[m]["stat"] for m in PLAYER_MARKETS]))
    long = long.join(stat_market, on="stat").drop("stat")
    d = e.join(long, on=["game_id", "player_id", "market"], how="inner").drop_nulls(["expected", "actual"])
    fl = pl.col("market").replace_strict(FLOOR, return_dtype=pl.Float64)
    d = d.with_columns(resid=pl.col("actual") - pl.col("expected"), rel=(pl.col("actual") - pl.col("expected")) / pl.max_horizontal(pl.col("expected"), fl))
    return d.sort("season", "week", "game_id", "market", "player_id")


if __name__ == "__main__":
    d = build()
    d.write_parquet(OUT)
    print(d.group_by("season").agg(n=pl.len(), mean_rel=pl.col("rel").mean(), sd_rel=pl.col("rel").std()).sort("season"))
