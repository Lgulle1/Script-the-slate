"""Combine volume and efficiency predictions into each market's final prediction.

  pass_att / rush_att / targets   = the volume prediction itself
  pass_cmp  = pass_att x comp_pct               pass_yds = pass_att x comp_pct x yds_per_cmp
  rush_yds  = rush_att x ypc
  rec       = targets x catch_pct               rec_yds  = targets x catch_pct x yds_per_rec
  games     = home pts = plays_home x ppp_home, away pts likewise;
              total = home + away, spread = home - away (same sign as nflverse spread_line),
              moneyline = P(home win) = Phi(spread / config.GAME_MARGIN_SD), as in the baselines.

A market whose volume or efficiency component is missing is None, never a partial product.
"""
from __future__ import annotations

import math

import config

PLAYER_MARKETS = tuple(m for m, spec in config.MARKETS.items() if spec["kind"] == "player")
GAME_MARKETS = tuple(m for m, spec in config.MARKETS.items() if spec["kind"] == "game")


def _mul(*xs):
    return None if any(x is None for x in xs) else math.prod(xs)


def combine_player(vol: dict, eff: dict) -> dict:
    return {
        "pass_att": vol.get("pass_att"),
        "pass_cmp": _mul(vol.get("pass_att"), eff.get("comp_pct")),
        "pass_yds": _mul(vol.get("pass_att"), eff.get("comp_pct"), eff.get("yds_per_cmp")),
        "rush_att": vol.get("rush_att"),
        "rush_yds": _mul(vol.get("rush_att"), eff.get("ypc")),
        "targets": vol.get("targets"),
        "rec": _mul(vol.get("targets"), eff.get("catch_pct")),
        "rec_yds": _mul(vol.get("targets"), eff.get("catch_pct"), eff.get("yds_per_rec")),
    }


def combine_game(vol: dict, eff: dict) -> dict:
    home = _mul(vol.get("plays_home"), eff.get("ppp_home"))
    away = _mul(vol.get("plays_away"), eff.get("ppp_away"))
    if home is None or away is None:
        return {"spread": None, "moneyline": None, "total": None}
    margin = home - away
    return {"spread": margin, "total": home + away,
            "moneyline": 0.5 * (1 + math.erf(margin / config.GAME_MARGIN_SD / math.sqrt(2)))}


def combined_predictors(vol_pair, eff_pair):
    """Compose (player, game) predictors from the volume and efficiency models into one harness predictor."""
    vp, vg = vol_pair
    ep, eg = eff_pair

    def player(history, targets, cutoff):
        return [combine_player(v, e) for v, e in zip(vp(history, targets, cutoff), ep(history, targets, cutoff))]

    def game(history, targets, cutoff):
        return [combine_game(v, e) for v, e in zip(vg(history, targets, cutoff), eg(history, targets, cutoff))]

    return player, game
