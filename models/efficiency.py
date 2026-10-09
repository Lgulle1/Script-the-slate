"""Efficiency models: one LightGBM regression per efficiency quantity.

  comp_pct      completions / attempts       (QB1; the efficiency side of pass_cmp)
  yds_per_cmp   passing yards / completions  (QB1; with comp_pct, the efficiency side of pass_yds)
  ypc           rushing yards / carries      (RB1/RB2 and FB; rush_yds)
  catch_pct     receptions / targets         (WR/TE/RB; rec)
  yds_per_rec   receiving yards / receptions (WR/TE/RB; with catch_pct, rec_yds)
  pts_per_play  team points / team plays     (game markets: spread, moneyline, total)

A market's final prediction is volume x efficiency (models/volume.py x this module); combining and
grading happen in run_volume_efficiency_backtest.py (3.3), not here.

Same philosophy as the volume models: features from features/volume_features.py (own recency- and
continuity-weighted pooled rates and their weight mass, team rates, opponent-allowed rates, role
context, no market lines), built once with each row using only games before its week's cutoff; same
starting hyperparameters (150 trees, 8 leaves, min 30 rows/leaf, L1); trained through the phase-2
walk-forward harness on games before each week, so 2025 is never loaded; no comparables, injuries or
simulation. A row's label is its ratio and its training weight is its denominator (a 2-carry game
should not count like a 25-carry game); rows with a zero denominator are not trained on.

Note: an L1 objective predicts a weighted conditional MEDIAN of the ratio, which sits a little below
the mean for right-skewed rates (yards per carry especially). That is a property of the specified
objective, not of the features.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from eval import backtest as bt
from features import volume_features as vf
from models import volume as vm

_PASS_STATS = ["completions", "attempts", "passing_yards"]
_REC_STATS = ["receptions", "targets", "receiving_yards"]
EFFICIENCY_SPECS = {
    "comp_pct": dict(market="pass_cmp", ratio=("completions", "attempts"), stats=_PASS_STATS,
                     pooled=[("comp_pct", "completions", "attempts"), ("ypa", "passing_yards", "attempts")]),
    "yds_per_cmp": dict(market="pass_yds", ratio=("passing_yards", "completions"), stats=_PASS_STATS,
                        pooled=[("yds_per_cmp", "passing_yards", "completions"), ("comp_pct", "completions", "attempts")]),
    "ypc": dict(market="rush_yds", ratio=("rushing_yards", "carries"), stats=["rushing_yards", "carries"],
                pooled=[("ypc", "rushing_yards", "carries")]),
    "catch_pct": dict(market="rec", ratio=("receptions", "targets"), stats=_REC_STATS,
                      pooled=[("catch_pct", "receptions", "targets"), ("yds_per_tgt", "receiving_yards", "targets")]),
    "yds_per_rec": dict(market="rec_yds", ratio=("receiving_yards", "receptions"), stats=_REC_STATS,
                        pooled=[("yds_per_rec", "receiving_yards", "receptions"), ("catch_pct", "receptions", "targets")]),
    # QB yards per (kneel-excluded) carry: the efficiency side of qb_rush_yds
    "qb_ypc": dict(market="qb_rush_yds", ratio=("rush_yds_ex_kneel", "rush_att_ex_kneel"), stats=["rush_yds_ex_kneel", "rush_att_ex_kneel"],
                   pooled=[("qb_ypc", "rush_yds_ex_kneel", "rush_att_ex_kneel")]),
}
TEAM_FEATURE_STATS = vf.TEAM_EFF_STATS + ["pass_rate"]
PTS_PER_PLAY = "pts_per_play"
# clip predictions into each ratio's natural range
_BOUNDS = {"comp_pct": (0.0, 1.0), "catch_pct": (0.0, 1.0)}


def build_efficiency_tables(player_log, team_log, team_volume, lineups) -> vf.FeatureTables:
    """Efficiency feature tables from already-loaded frames (no database access)."""
    week_cutoff = vf.week_cutoffs(team_log)
    clock = vf._Clock(team_log["season"].unique().to_list())
    side = vf.build_team_side_features(team_log, team_volume, week_cutoff, clock, stats=TEAM_FEATURE_STATS)
    players = {q: vf.build_player_features(q, spec, player_log, team_log, team_volume, lineups, week_cutoff, clock, side)
               for q, spec in EFFICIENCY_SPECS.items()}
    return vf.FeatureTables(players, vf.build_team_game_features(team_log, team_volume, side, PTS_PER_PLAY))


def load_efficiency_tables(data, raw_db=None, max_season=None, injury=None, comps=None, comp_searches=None) -> vf.FeatureTables:
    """Efficiency feature tables; `injury` adds the 4a columns, `comps` the comparable columns of each ratio's market (4c.5). None = Phase 3."""
    import config
    raw_db = raw_db or config.RAW_DUCKDB_PATH
    cap = config.cap_season(max_season)  # raises config.HoldoutError for 2025+
    tv = vf.build_team_volume(raw_db, cap)
    ln = vf.lu.build_lineups(raw_db, cap)
    bt.assert_no_holdout(tv.join(data.team_log.select("game_id", "season").unique(), on="game_id", how="inner"))
    tables = vf._with_injury(build_efficiency_tables(data.player_log, data.team_log, tv, ln), injury, data)   # injury=None -> Phase 3
    return vf._with_comps(tables, comps, EFFICIENCY_SPECS, comp_searches)


def efficiency_predictors(tables: vf.FeatureTables, params=vm.PARAMS, min_train=vm.MIN_TRAIN_ROWS):
    """(player_predictor, game_predictor) for eval.backtest.walk_forward.

    As in the volume models, training rows are the feature rows whose (player, gameday) / (team, gameday)
    appear in `history`, so the harness's cutoff decides what a model may learn from.
    """
    def player(history, targets, cutoff):
        keys = history.player_log.select("player_id", "gameday")
        want = pl.DataFrame({"player_id": [t.player_id for t in targets], "gameday": [t.gameday for t in targets],
                             "_i": range(len(targets))})
        out = [{q: None for q in EFFICIENCY_SPECS} for _ in targets]
        for q in EFFICIENCY_SPECS:
            df = tables.players[q]
            train = df.join(keys, on=["player_id", "gameday"], how="inner").filter(pl.col("den") > 0)
            test = df.join(want, on=["player_id", "gameday"], how="inner")
            if train.height < min_train or test.height == 0:
                continue
            assert train["gameday"].max() < cutoff, "training row at/after the cutoff"
            model, cols = vm._fit(train, params)
            lo, hi = _BOUNDS.get(q, (0.0, np.inf))
            for i, p in zip(test["_i"].to_list(), np.clip(model.predict(test.select(cols).to_pandas()), lo, hi)):
                out[i][q] = float(p)
        return out

    def game(history, targets, cutoff):
        df = tables.teams
        train = (df.join(history.team_log.select("team", "gameday"), on=["team", "gameday"], how="inner")
                 .filter(pl.col("den") > 0))
        out = [{"ppp_home": None, "ppp_away": None} for _ in targets]
        if train.height < min_train:
            return out
        assert train["gameday"].max() < cutoff, "training row at/after the cutoff"
        model, cols = vm._fit(train, params)
        for side, team_of in (("ppp_home", lambda t: t.home_team), ("ppp_away", lambda t: t.away_team)):
            want = pl.DataFrame({"team": [team_of(t) for t in targets], "gameday": [t.gameday for t in targets],
                                 "_i": range(len(targets))})
            test = df.join(want, on=["team", "gameday"], how="inner")
            for i, p in zip(test["_i"].to_list(), np.maximum(model.predict(test.select(cols).to_pandas()), 0.0)):
                out[i][side] = float(p)
        return out

    return player, game


# ---- actuals the harness needs to score efficiency quantities
def efficiency_player_actuals(row: dict) -> dict:
    """Extra actuals for one player-game row: each ratio the player is eligible for, when its denominator > 0."""
    elig = bt.eligible_markets(row)
    out = {}
    for q, spec in EFFICIENCY_SPECS.items():
        num, den = spec["ratio"]
        if spec["market"] in elig and row[den] and row[den] > 0:
            out[q] = row[num] / row[den]
    return out


def ppp_extras(team_log: pl.DataFrame, team_volume: pl.DataFrame) -> pl.DataFrame:
    """Per-game actual points per play for home and away (harness extra actuals)."""
    tg = team_log.join(team_volume.select("game_id", "team", "plays"), on=["game_id", "team"], how="left")
    tg = tg.with_columns(ppp=pl.col("pf") / pl.col("plays"))
    home = tg.filter(pl.col("is_home")).select("game_id", ppp_home=pl.col("ppp"))
    away = tg.filter(~pl.col("is_home")).select("game_id", ppp_away=pl.col("ppp"))
    return home.join(away, on="game_id")
