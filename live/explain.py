"""Per-prediction attribution: why a prediction is the number it is.

Every player market is a product of LightGBM components (models/combine.py: rec_yds = targets x catch_pct x yds_per_rec). For each component the model's raw output
is split EXACTLY into a baseline (the model's average output) plus one contribution per input feature (LightGBM `pred_contrib`, TreeSHAP), so
    sum of a row group's contributions == model.predict(row)            (to < 1e-9, tested)
and component_prediction = that sum clipped to the component's natural range, the number combine.py multiplies. A game's margin from the 4b.2 model is a plain linear
sum, so it decomposes by term (margin_decomposition). Nothing here changes a prediction; it only reads the fitted models.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from models import combine, efficiency
from models import game_state as gs

BASELINE = "(baseline)"

MARKET_COMPONENTS = {
    "pass_att": ("pass_att",),
    "pass_cmp": ("pass_att", "comp_pct"),
    "pass_yds": ("pass_att", "comp_pct", "yds_per_cmp"),
    "rush_att": ("rush_att",),
    "rush_yds": ("rush_att", "ypc"),
    "qb_rush_att": ("qb_rush_att",),
    "qb_rush_yds": ("qb_rush_att", "qb_ypc"),
    "targets": ("targets",),
    "rec": ("targets", "catch_pct"),
    "rec_yds": ("targets", "catch_pct", "yds_per_rec"),
}
GAME_COMPONENTS = {"plays": "plays", "pts_per_play": "pts_per_play"}

LONG_SCHEMA = {"quantity": pl.String, "feature": pl.String, "value": pl.Float64, "contribution": pl.Float64, "component_prediction": pl.Float64}


def contributions(model_cols, df: pl.DataFrame, quantity: str, keys: list, bounds=(0.0, np.inf)) -> pl.DataFrame:
    """Long table: for each row of `df` one baseline row and one row per model feature (keys, quantity, feature, value, contribution, component_prediction)."""
    model, cols = model_cols
    if not df.height:
        return pl.DataFrame(schema={**{k: df.schema[k] for k in keys}, **LONG_SCHEMA})
    X = df.select(cols).to_pandas()
    contrib = np.asarray(model.booster_.predict(X, pred_contrib=True), dtype=float)          # (n, F + 1): per feature, then the baseline
    pred = np.clip(contrib.sum(axis=1), *bounds)
    vals = df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    vals = np.where(np.isfinite(vals), vals, np.nan)
    kdf = df.select(keys)
    n = df.height
    parts = []
    for j, c in enumerate(cols):
        parts.append(kdf.with_columns(quantity=pl.lit(quantity), feature=pl.lit(c), value=pl.Series(vals[:, j]).fill_nan(None),
                                      contribution=pl.Series(contrib[:, j]), component_prediction=pl.Series(pred)))
    parts.append(kdf.with_columns(quantity=pl.lit(quantity), feature=pl.lit(BASELINE), value=pl.lit(None, dtype=pl.Float64),
                                  contribution=pl.Series(contrib[:, -1]), component_prediction=pl.Series(pred)))
    out = pl.concat(parts)
    assert out.height == n * (len(cols) + 1)
    return out


def top_drivers(long: pl.DataFrame, n: int = 5) -> pl.DataFrame:
    """The n features with the largest absolute contribution per prediction (the baseline is where every prediction starts, not a driver)."""
    keys = [c for c in long.columns if c not in LONG_SCHEMA]
    return (long.filter(pl.col("feature") != BASELINE)
            .with_columns(abs_contribution=pl.col("contribution").abs())
            .sort([*keys, "abs_contribution"], descending=[False] * len(keys) + [True], maintain_order=True)
            .group_by(keys, maintain_order=True).head(n).drop("abs_contribution"))


def explain_players(models, vol, eff, elig_of: dict) -> pl.DataFrame:
    """Contributions of every component of every eligible target player (keys game_id, player_id). `models` is a live.predict.LiveModels."""
    parts = []
    for q, mc in models.vol.items():
        df = vol.players[q]
        parts.append(contributions(mc, df, q, ["game_id", "player_id"]))
    for q, mc in models.eff.items():
        df = eff.players[q]
        parts.append(contributions(mc, df, q, ["game_id", "player_id"], efficiency._BOUNDS.get(q, (0.0, np.inf))))
    if not parts:
        return pl.DataFrame()
    keep = pl.DataFrame([dict(game_id=g, player_id=p) for g, p in elig_of], schema={"game_id": pl.String, "player_id": pl.String})
    return pl.concat(parts).join(keep, on=["game_id", "player_id"], how="inner").with_columns(team=pl.lit(None, dtype=pl.String))


def explain_teams(models, vol, eff) -> pl.DataFrame:
    """Contributions of the two team components (plays, points per play) for every team in the target games (keys game_id, team)."""
    parts = []
    if models.vol_team and vol.teams.height:
        parts.append(contributions(models.vol_team, vol.teams, "plays", ["game_id", "team"]))
    if models.eff_team and eff.teams.height:
        parts.append(contributions(models.eff_team, eff.teams, "pts_per_play", ["game_id", "team"]))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts).with_columns(player_id=pl.lit(None, dtype=pl.String))


def explain_stack(models, vol, eff, elig_of: dict, stack: str) -> pl.DataFrame:
    """One stack's attribution table: game_id, player_id, team, stack, quantity, feature, value, contribution, component_prediction."""
    frames = [f for f in (explain_players(models, vol, eff, elig_of), explain_teams(models, vol, eff)) if f.height]
    cols = ["game_id", "player_id", "team", "stack", "quantity", "feature", "value", "contribution", "component_prediction"]
    if not frames:
        return pl.DataFrame(schema={**{c: pl.String for c in ("game_id", "player_id", "team", "stack", "quantity", "feature")}, "value": pl.Float64, "contribution": pl.Float64,
                                    "component_prediction": pl.Float64})
    return pl.concat(frames, how="diagonal_relaxed").with_columns(stack=pl.lit(stack)).select(cols)


def margin_decomposition(home: dict, away: dict, coef: dict) -> pl.DataFrame:
    """The 4b.2 expected margin by term. side is 'home' / 'away' (that side's expected-points terms), 'margin' (home - away; league average and the constant cancel
    or are shared) or 'game' for the 'margin' and 'total' summary rows. points of the non-summary terms with side == 'margin' sum to the margin."""
    rows = []
    terms = [("league_pts", lambda s: s["league_pts"], None), ("c", lambda s: coef["c"], None)] + [(f, (lambda s, f=f: coef[f] * s[f]), f) for f in gs.FEATURES]
    for name, fn, feat in terms:
        h, a = float(fn(home)), float(fn(away))
        rows += [dict(term=name, side="home", points=h, home_value=home.get(feat) if feat else None, away_value=away.get(feat) if feat else None, coefficient=coef.get(feat) if feat else None),
                 dict(term=name, side="away", points=a, home_value=None, away_value=None, coefficient=None),
                 dict(term=name, side="margin", points=h - a, home_value=None, away_value=None, coefficient=None)]
    margin, total = gs.game_prediction(home, away, coef)
    rows += [dict(term="margin", side="game", points=margin, home_value=None, away_value=None, coefficient=None),
             dict(term="total", side="game", points=total, home_value=None, away_value=None, coefficient=None)]
    schema = {"term": pl.String, "side": pl.String, "points": pl.Float64, "home_value": pl.Float64, "away_value": pl.Float64, "coefficient": pl.Float64}
    return pl.DataFrame(rows, schema=schema).with_columns(home_team=pl.lit(home.get("team")), away_team=pl.lit(away.get("team")))
