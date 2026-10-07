"""Volume models: one LightGBM regression per volume quantity.

  pass_att   QB1 pass attempts               rush_att  RB1/RB2 (and FB) carries
  targets    WR/TE/RB targets                team_plays  a team's offensive plays (game-market volume)

Features come from features/volume_features.py (recency- and continuity-weighted own history, team
rates, opponent-allowed rates, role context); they are built once, each row using only games before
its own week's cutoff. Training goes through the phase-2 walk-forward harness: for every week W the
models are refit on rows from games before W's first kickoff and predict week W, so the locked 2025
holdout is never loaded. Hyperparameters start from the earlier RB test (150 trees, 8 leaves,
min 30 rows/leaf, L1 objective) -- a starting point, not a spec. No comparables, injuries,
simulation, or market lines.

Note: an L1 objective predicts a conditional MEDIAN while the phase-2 baselines predict means; for
right-skewed volume this can differ by a fraction of a unit.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from features.volume_features import PLAYER_SPECS, FeatureTables

PARAMS = dict(objective="l1", n_estimators=150, num_leaves=8, min_child_samples=30, learning_rate=0.05,
              colsample_bytree=0.8, random_state=0, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
MIN_TRAIN_ROWS = 300  # no model (predictions are None) until a quantity has this many earlier rows

_KEYS = ("player_id", "game_id", "gameday", "season", "week", "team", "opponent", "label", "den")
TEAM_PLAYS = "team_plays"


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _KEYS]


def _fit(train: pl.DataFrame, params=PARAMS):
    cols = feature_columns(train)
    model = lgb.LGBMRegressor(**params)
    weight = train["den"].to_numpy() if "den" in train.columns else None  # efficiency rows weigh by their denominator
    model.fit(train.select(cols).to_pandas(), train["label"].to_numpy(), sample_weight=weight)
    return model, cols


def _predict(model, cols, df: pl.DataFrame) -> np.ndarray:
    return np.maximum(model.predict(df.select(cols).to_pandas()), 0.0)


def volume_predictors(tables: FeatureTables, params=PARAMS, min_train=MIN_TRAIN_ROWS):
    """(player_predictor, game_predictor) for eval.backtest.walk_forward.

    Training rows are the feature rows whose (player, gameday) / (team, gameday) appear in
    `history`, so the harness's cutoff -- not this module -- decides what a model may learn from.
    """
    def player(history, targets, cutoff):
        keys = history.player_log.select("player_id", "gameday")
        want = pl.DataFrame({"player_id": [t.player_id for t in targets], "gameday": [t.gameday for t in targets],
                             "_i": range(len(targets))})
        out = [{q: None for q in PLAYER_SPECS} for _ in targets]  # None = no model yet / no feature row
        for q in PLAYER_SPECS:
            df = tables.players[q]
            train = df.join(keys, on=["player_id", "gameday"], how="inner")
            test = df.join(want, on=["player_id", "gameday"], how="inner")
            if train.height < min_train or test.height == 0:
                continue
            assert train["gameday"].max() < cutoff, "training row at/after the cutoff"
            model, cols = _fit(train, params)
            for i, p in zip(test["_i"].to_list(), _predict(model, cols, test)):
                out[i][q] = float(p)
        return out

    def game(history, targets, cutoff):
        df = tables.teams
        train = df.join(history.team_log.select("team", "gameday"), on=["team", "gameday"], how="inner")
        out = [{"plays_home": None, "plays_away": None} for _ in targets]
        if train.height < min_train:
            return out
        assert train["gameday"].max() < cutoff, "training row at/after the cutoff"
        model, cols = _fit(train, params)
        for side, team_of in (("plays_home", lambda t: t.home_team), ("plays_away", lambda t: t.away_team)):
            want = pl.DataFrame({"team": [team_of(t) for t in targets], "gameday": [t.gameday for t in targets],
                                 "_i": range(len(targets))})
            test = df.join(want, on=["team", "gameday"], how="inner")
            for i, p in zip(test["_i"].to_list(), _predict(model, cols, test)):
                out[i][side] = float(p)
        return out

    return player, game


def plays_extras(team_log: pl.DataFrame, team_volume: pl.DataFrame) -> pl.DataFrame:
    """Per-game actual plays for home and away (the harness's extra actuals for team_plays)."""
    tg = team_log.join(team_volume.select("game_id", "team", "plays"), on=["game_id", "team"], how="left")
    home = tg.filter(pl.col("is_home")).select("game_id", plays_home=pl.col("plays"))
    away = tg.filter(~pl.col("is_home")).select("game_id", plays_away=pl.col("plays"))
    return home.join(away, on="game_id")
