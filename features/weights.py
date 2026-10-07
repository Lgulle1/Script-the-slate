"""Pure weighting functions: recency, continuity, quality.

No training happens here; other modules import these. All constants live in config.py.

Continuity factors (this file is the single implementation; the feature code calls it)
------------------------------------------------------------------------------------
A past game is down-weighted by the product of the penalties (config.CONTINUITY_PENALTIES, row chosen by
config.MARKETS[market]["penalty_key"]) for every factor that CHANGED between that game and the game being
predicted. Each factor is detected from the pregame depth chart (features/lineups.py builds the group ids):

  QB           the depth-1 quarterback differs
  RB_group     the running backs at depth 1-2 differ (fullbacks are listed as RB)
  WR_TE_group  WR1 through WR3 plus TE1 differ (a fixed 3 WR + 1 TE, NOT the four highest-ranked WR/TE of
               either position; decided to keep it this way -- do not change without re-running Phase 3)
  OL           the five starters (T/G/C at depth 1) differ
  role         the player's own depth-chart slot bucket (1, 2, 3, 0 = unlisted) or position family differs
  HC, OC       flagged ONLY when the player changed teams, until a coaches_history table exists
  Any team change flags every factor. A missing depth chart on either side is never a change.
"""
import numpy as np

import config


def games_elapsed(past_season: int, past_game: int, target_season: int, target_game: int,
                  games_in_season=None) -> float:
    """Team games between a past game and the game being predicted.

    `past_game` / `target_game` are 1-based team-game numbers within their regular
    season (byes don't count). The immediately preceding game is 1 game back. Each
    offseason crossed adds config.OFFSEASON_GAP_GAMES; fully skipped seasons also add
    their games. `games_in_season` maps season -> game count (default 16 before 2021,
    17 from 2021).
    """
    if (target_season, target_game) <= (past_season, past_game):
        raise ValueError("past game must come before the target game")
    games_in_season = games_in_season or {}

    def n(season):
        return games_in_season.get(season, 17 if season >= 2021 else 16)

    if past_season == target_season:
        return float(target_game - past_game)
    elapsed = (n(past_season) - past_game) + config.OFFSEASON_GAP_GAMES + target_game
    for season in range(past_season + 1, target_season):
        elapsed += n(season) + config.OFFSEASON_GAP_GAMES
    return float(elapsed)


def recency_weight(games_ago: float, half_life: float = config.RECENCY_HALF_LIFE_GAMES) -> float:
    """Exponential decay: 1.0 at 0 games back, 0.5 at one half-life (6 team games)."""
    if games_ago < 0:
        raise ValueError(f"games_ago must be >= 0, got {games_ago}")
    return 0.5 ** (games_ago / half_life)


def continuity_flags(lin_past, lin_target, slot_past, slot_target, family_past, family_target, team_changed) -> dict:
    """Which continuity factors changed between each past game and the target game.

    `lin_past[c]` / `lin_target[c]` are lineup-group ids for c in QB / RB / WRTE / OL (-1 = chart missing);
    the other arguments compare the player's slot bucket, position family and team. Past-game arguments may
    be numpy arrays (one entry per past game); the result maps factor -> boolean (array), in the fixed order
    HC, OC, role, QB, OL, RB_group, WR_TE_group that continuity_weight multiplies in.
    """
    flags = {"HC": team_changed, "OC": team_changed,
             "role": (slot_past != slot_target) | (family_past != family_target) | team_changed}
    for factor, comp in (("QB", "QB"), ("OL", "OL"), ("RB_group", "RB"), ("WR_TE_group", "WRTE")):
        t = lin_target[comp]
        flags[factor] = ((lin_past[comp] != t) & (lin_past[comp] != -1) & (t != -1)) | team_changed
    return flags


def continuity_weight(changes: dict, market: str, penalties: dict | None = None):
    """Product of per-market penalties for each factor that changed.

    `changes` maps factor name -> bool or a boolean numpy array (True = that factor changed between the past
    game and the game being predicted); the result is a float or an array of the same shape. Unchanged factors
    contribute 1.0. Factors are multiplied in the order given. Raises if the market has no penalty table or a
    factor is unknown/missing from it, rather than silently treating a missing penalty as "no penalty".
    """
    penalties = config.CONTINUITY_PENALTIES if penalties is None else penalties
    if market not in penalties:
        raise KeyError(f"no continuity penalties configured for market {market!r}; "
                       "fill config.CONTINUITY_PENALTIES from the guide")
    table = penalties[market]
    unknown = set(changes) - set(config.CONTINUITY_FACTORS)
    if unknown:
        raise KeyError(f"unknown continuity factors: {sorted(unknown)}")
    shapes = [np.shape(c) for c in changes.values() if np.ndim(c)]
    weight = np.ones(shapes[0]) if shapes else 1.0  # array input -> array output, even if nothing changed
    for factor, changed in changes.items():
        if not np.any(changed):
            continue
        if factor not in table:
            raise KeyError(f"market {market!r} has no penalty for factor {factor!r}")
        weight = weight * (np.where(changed, table[factor], 1.0) if np.ndim(changed) else table[factor])
    return weight


def quality_weight(quality: str) -> float:
    """observed 1.0, derived 0.9, estimated 0.65.

    Not applied in Phase 2 or 3 (every input there is observed, so the weight is 1.0); first applied in 4c.
    """
    try:
        return config.QUALITY_WEIGHTS[quality]
    except KeyError:
        raise KeyError(f"unknown data quality {quality!r}; expected one of {sorted(config.QUALITY_WEIGHTS)}") from None
