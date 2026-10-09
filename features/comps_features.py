"""Comparable features for the volume and efficiency models (build plan 4c.5).

The per-target, per-market table of models.comps.comp_shifts (data/processed/comp_shifts_<window>.parquet) becomes extra feature columns, prefixed
`cmp_`, in the Phase 3 feature tables (features.volume_features.FeatureTables):
  * a volume model takes the columns of its own market (pass_att, rush_att, targets, qb_rush_att), the team-plays model those of the game markets;
  * an efficiency model takes those of its market (pass_cmp, pass_yds, rush_yds, rec, rec_yds, qb_rush_yds), points per play those of the game markets.
Per search: shift_vol, shift_eff, n_eff, nomatch, best_sim (4c.4), n_eff_eff, nomatch_eff (the efficiency side), and share_obs, share_der, share_est,
completeness (4c.5) -- the observed, derived and estimated shares of the feature weight behind the matches, kept as separate columns, never lumped.
Booleans become 0 / 1. A row without comparables (a player-game that is not a scored target) keeps them missing, never 0.

Walk-forward by construction: every value of a week-W target comes from games before W. The game markets' columns are the same for spread, total and
moneyline (same units, continuity row and expectations), so the team tables read them from `total`.
"""
from __future__ import annotations

import polars as pl

import config

SEARCHES = ("S1", "S2", "S3", "S4", "S5")
SHIFT_COLUMNS = ("shift_vol", "shift_eff", "n_eff", "nomatch", "best_sim", "n_eff_eff", "nomatch_eff")         # 4c.4
SHARE_COLUMNS = ("share_obs", "share_der", "share_est", "completeness")                                       # 4c.5
PREFIX = "cmp_"
TEAM_MARKET = "total"


def columns(searches=SEARCHES) -> list:
    """The comparable columns (unprefixed) a model takes for the given searches (the 4c.6 per-search ablation passes one search, as a tuple)."""
    if isinstance(searches, str):
        raise TypeError(f"searches must be a sequence of search names, not the string {searches!r}")
    searches = tuple(searches)
    if not searches or not set(searches) <= set(SEARCHES) or len(set(searches)) != len(searches):
        raise ValueError(f"searches must be distinct names among {SEARCHES}, got {searches}")
    return [f"{c}_{s}" for s in searches for c in SHIFT_COLUMNS + SHARE_COLUMNS]


def load(window: str = "recency_weighted", path=None) -> pl.DataFrame:
    return pl.read_parquet(path or config.PROCESSED_DIR / f"comp_shifts_{window.replace(':', '_')}.parquet")


def _market_frame(feats: pl.DataFrame, market: str, keys: list, searches) -> pl.DataFrame:
    cols = columns(searches)
    missing = [c for c in cols if c not in feats.columns]
    if missing:
        raise KeyError(f"the comparable table lacks {len(missing)} requested columns (a table written before 4c.5?): {missing[:4]}")
    side = pl.col("player_id").is_not_null() if "player_id" in keys else pl.col("player_id").is_null()
    out = feats.filter((pl.col("market") == market) & side).select(*keys, *cols)
    assert out.select(keys).is_unique().all(), f"comparable features are not unique per {keys} for {market}"
    return out.select(*keys, *[pl.col(c).cast(pl.Float64).alias(PREFIX + c) for c in cols])        # numeric for LightGBM: booleans 0 / 1, an all-null column stays null


def attach(tables, feats: pl.DataFrame, specs: dict, searches=SEARCHES):
    """FeatureTables with the comparable columns of each table's market joined on: players on (player_id, game_id), teams on (game_id, team).
    `specs` is the quantity -> spec dict of the tables (volume_features.PLAYER_SPECS or efficiency.EFFICIENCY_SPECS), whose `market` names the
    market whose columns a quantity takes."""
    from features.volume_features import FeatureTables
    players = {}
    for q, df in tables.players.items():
        f = _market_frame(feats, specs[q]["market"], ["player_id", "game_id"], searches)
        players[q] = df.join(f, on=["player_id", "game_id"], how="left", maintain_order="left")
    t = _market_frame(feats, TEAM_MARKET, ["game_id", "team"], searches)
    return FeatureTables(players, tables.teams.join(t, on=["game_id", "team"], how="left", maintain_order="left"))


def input_shares(feats: pl.DataFrame) -> pl.DataFrame:
    """For the confidence score's input-completeness component (build plan 4c.5.3; the score itself comes in Phase 5): per target and market, the
    observed / derived / estimated shares and the completeness penalty of every search, as separate columns."""
    keys = ["season", "week", "game_id", "team", "player_id", "market"]
    return feats.select(*keys, *[f"{c}_{s}" for s in SEARCHES for c in SHARE_COLUMNS]).sort(keys, nulls_last=True)
