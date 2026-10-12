"""Phase 3 prediction for targets that have not been played: fit once on the training feature tables, predict the target feature rows.

`fit_models(vol_tables, eff_tables, cutoff=None)` fits one LightGBM per volume / efficiency quantity on the rows of the training tables (when `cutoff` is
given, only rows dated before it: the replay used by the tests to reproduce the walk-forward harness; live, the tables hold 2020-2024 games only and
`cutoff` is None, so models stay fit on 2020-2024 rows, plan P5.4). `predict(models, vol_target, eff_target)` returns, per (game_id, player_id) and per
(game_id, team), the combined market predictions of models/combine.py, restricted to the markets each player is eligible for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from eval import eligibility as elig
from features.volume_features import PLAYER_SPECS, FeatureTables
from models import combine, efficiency
from models import volume as vm


@dataclass
class LiveModels:
    vol: dict          # quantity -> (model, columns)
    eff: dict
    vol_team: tuple | None
    eff_team: tuple | None
    n_train: dict


def _train(df: pl.DataFrame, cutoff, positive_den: bool):
    if cutoff is not None:
        df = df.filter(pl.col("gameday") < cutoff)
    if positive_den:
        df = df.filter(pl.col("den") > 0)
    return df


def fit_models(vol: FeatureTables, eff: FeatureTables, cutoff=None, min_train=vm.MIN_TRAIN_ROWS) -> LiveModels:
    n, vm_, em_ = {}, {}, {}
    for q in PLAYER_SPECS:
        tr = _train(vol.players[q], cutoff, False)
        n[q] = tr.height
        if tr.height >= min_train:
            vm_[q] = vm._fit(tr)
    for q in efficiency.EFFICIENCY_SPECS:
        tr = _train(eff.players[q], cutoff, True)
        n[q] = tr.height
        if tr.height >= min_train:
            em_[q] = vm._fit(tr)
    tv, te = _train(vol.teams, cutoff, False), _train(eff.teams, cutoff, True)
    n["plays"], n["pts_per_play"] = tv.height, te.height
    return LiveModels(vm_, em_, vm._fit(tv) if tv.height >= min_train else None, vm._fit(te) if te.height >= min_train else None, n)


def _pred(model_cols, df: pl.DataFrame, bounds=(0.0, np.inf)) -> np.ndarray:
    model, cols = model_cols
    return np.clip(model.predict(df.select(cols).to_pandas()), *bounds)


def predict_players(models: LiveModels, vol: FeatureTables, eff: FeatureTables, elig_of: dict) -> pl.DataFrame:
    """(game_id, player_id, market, mean) for every target player-market in `elig_of` ((game_id, player_id) -> comma-joined markets)."""
    vp, ep = {}, {}
    for q, mc in models.vol.items():
        df = vol.players[q]
        if df.height:
            for g, p, v in zip(df["game_id"], df["player_id"], _pred(mc, df)):
                vp.setdefault((g, p), {})[q] = float(v)
    for q, mc in models.eff.items():
        df = eff.players[q]
        if df.height:
            lo, hi = efficiency._BOUNDS.get(q, (0.0, np.inf))
            for g, p, v in zip(df["game_id"], df["player_id"], _pred(mc, df, (lo, hi))):
                ep.setdefault((g, p), {})[q] = float(v)
    rows = []
    for (g, p), markets in elig_of.items():
        out = combine.combine_player(vp.get((g, p), {}), ep.get((g, p), {}))
        for m in elig.markets_of(markets):
            rows.append((g, p, m, out.get(m)))
    return pl.DataFrame(rows, schema={"game_id": pl.String, "player_id": pl.String, "market": pl.String, "mean": pl.Float64}, orient="row")


def predict_games(models: LiveModels, vol: FeatureTables, eff: FeatureTables, games: pl.DataFrame) -> pl.DataFrame:
    """(game_id, market, mean) for spread, moneyline, total; `games` has game_id, home_team, away_team."""
    plays, ppp = {}, {}
    if models.vol_team and vol.teams.height:
        for g, t, v in zip(vol.teams["game_id"], vol.teams["team"], _pred(models.vol_team, vol.teams)):
            plays[(g, t)] = float(v)
    if models.eff_team and eff.teams.height:
        for g, t, v in zip(eff.teams["game_id"], eff.teams["team"], _pred(models.eff_team, eff.teams)):
            ppp[(g, t)] = float(v)
    rows = []
    for r in games.iter_rows(named=True):
        g, h, a = r["game_id"], r["home_team"], r["away_team"]
        out = combine.combine_game({"plays_home": plays.get((g, h)), "plays_away": plays.get((g, a))},
                                   {"ppp_home": ppp.get((g, h)), "ppp_away": ppp.get((g, a))})
        rows += [(g, m, out.get(m)) for m in combine.GAME_MARKETS]
    return pl.DataFrame(rows, schema={"game_id": pl.String, "market": pl.String, "mean": pl.Float64}, orient="row")
