"""One eligibility rule for who is graded (and trained on) in the rush and receiving markets.

The rule lives in config.ELIGIBLE_PLAYER_RULE; this module applies it. For a player-game it needs only
information known BEFORE kickoff:

  * chart flags from the pre-game weekly depth chart (eval/baselines.build_chart_flags):
        chart_rb   a non-fullback back ranked RB1 or RB2 (ranked by depth_team within the team)
        chart_fb   a fullback ranked FB1            chart_wr   a WR at depth 1-3          chart_te   a TE at depth 1
  * usage over the previous `window_games` games he PLAYED: avg_carries_prev / avg_targets_prev
    (nothing from the game itself; NaN with no history, which makes the usage clause false)

and produces `elig`, the comma-joined list of markets he is scored for (e.g. "rush_att,rush_yds,targets,rec,rec_yds").
Rushing is RB-group only (position family RB, which folds in fullbacks): no QB, WR or TE is ever in rush_att / rush_yds.
The QB markets: pass_att / pass_cmp / pass_yds keep the plain QB1 pool; qb_rush_att / qb_rush_yds are scored for QB1 when he
averaged at least config.QB_RUSH_MIN_ATT kneel-excluded rush attempts over his previous games.
"""
from __future__ import annotations

import polars as pl

import config

CHART_COLUMNS = ("chart_rb", "chart_fb", "chart_wr", "chart_te")
_CHART_GROUP = {"RB": "chart_rb", "FB": "chart_fb", "WR": "chart_wr", "TE": "chart_te"}


def usage_columns() -> tuple[str, str, str]:
    return "avg_carries_prev", "avg_targets_prev", "avg_qb_rush_prev"


def add_usage_averages(df: pl.DataFrame) -> pl.DataFrame:
    """Previous-N-games-played average carries and targets (strictly earlier games, same player).

    Needs df sorted by player_id, gameday. Uses up to `window_games` earlier games, at least one.
    """
    n = config.ELIGIBLE_PLAYER_RULE["usage"]["window_games"]
    c, t, q = usage_columns()
    return df.with_columns(
        pl.col("carries").cast(pl.Float64).shift(1).rolling_mean(window_size=n, min_samples=1).over("player_id").alias(c),
        pl.col("targets").cast(pl.Float64).shift(1).rolling_mean(window_size=n, min_samples=1).over("player_id").alias(t),
        pl.col("rush_att_ex_kneel").cast(pl.Float64).shift(1).rolling_mean(window_size=n, min_samples=1).over("player_id").alias(q),
    )


def eligibility_expr(rule: dict | None = None) -> dict[str, pl.Expr]:
    """market -> boolean expression over a player-game frame carrying family, slot, chart_* and avg_*_prev columns."""
    rule = rule or config.ELIGIBLE_PLAYER_RULE
    use = rule["usage"]
    c, t, q = usage_columns()
    usage_rush = (pl.col(c) >= use["min_carries"]).fill_null(False)
    usage_recv = (pl.col(t) >= use["min_targets"]).fill_null(False)
    chart_rush = pl.lit(False)
    chart_recv = pl.lit(False)
    for group, which in rule["chart_markets"].items():
        col = pl.col(_CHART_GROUP[group]).fill_null(False)
        if which in ("both", "rushing"):
            chart_rush = chart_rush | col
        if which in ("both", "receiving"):
            chart_recv = chart_recv | col
    if rule["scope"] == "per_market":
        rush, recv = chart_rush | usage_rush, chart_recv | usage_recv
    elif rule["scope"] == "all_markets":
        rush = recv = chart_recv | chart_rush | usage_rush | usage_recv
    else:
        raise ValueError(f"unknown eligibility scope {rule['scope']!r}")
    rush = rush & pl.col("family").is_in(list(rule["rush_families"]))     # rushing: the RB group only
    out = {}
    for m, spec in config.MARKETS.items():
        if spec["kind"] != "player":
            continue
        if spec["pool"] == "ELIGIBLE_PLAYER_RULE":
            out[m] = rush if m in rule["rush_markets"] else recv
        elif spec["pool"] == "QB_RUSH_RULE":   # QB1 who has been running: >= QB_RUSH_MIN_ATT kneel-excluded attempts a game
            out[m] = ((pl.col("family") == "QB") & (pl.col("slot") == 1) & (pl.col(q) >= config.QB_RUSH_MIN_ATT)).fill_null(False)
        else:  # plain depth-chart pool (QB1)
            cond = pl.lit(False)
            for fam, slots in spec["pool"].items():
                cond = cond | ((pl.col("family") == fam) & pl.col("slot").is_in(list(slots)))
            out[m] = cond
    return out


def add_eligibility(df: pl.DataFrame, rule: dict | None = None) -> pl.DataFrame:
    """Adds `elig`: comma-joined markets each player-game is scored for (empty string when none)."""
    exprs = eligibility_expr(rule)
    parts = [pl.when(e).then(pl.lit(m)).otherwise(None) for m, e in exprs.items()]
    return df.with_columns(pl.concat_list(parts).list.drop_nulls().list.join(",").alias("elig"))


def markets_of(elig: str) -> tuple:
    """The markets in an `elig` string, in config.MARKETS order."""
    return tuple(elig.split(",")) if elig else ()
