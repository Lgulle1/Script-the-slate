"""Pure weighting functions: recency, continuity, quality.

No training happens here; other modules import these. All constants live in config.py.
"""
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


def continuity_weight(changes: dict, market: str, penalties: dict | None = None) -> float:
    """Product of per-market penalties for each factor that changed.

    `changes` maps factor name -> bool (True = that factor changed between the past
    game and the game being predicted). Unchanged factors contribute 1.0. Raises if
    the market has no penalty table or a factor is unknown/missing from it, rather
    than silently treating a missing penalty as "no penalty".
    """
    penalties = config.CONTINUITY_PENALTIES if penalties is None else penalties
    if market not in penalties:
        raise KeyError(f"no continuity penalties configured for market {market!r}; "
                       "fill config.CONTINUITY_PENALTIES from the guide")
    table = penalties[market]
    unknown = set(changes) - set(config.CONTINUITY_FACTORS)
    if unknown:
        raise KeyError(f"unknown continuity factors: {sorted(unknown)}")
    weight = 1.0
    for factor, changed in changes.items():
        if not changed:
            continue
        if factor not in table:
            raise KeyError(f"market {market!r} has no penalty for factor {factor!r}")
        weight *= table[factor]
    return weight


def quality_weight(quality: str) -> float:
    """observed 1.0, derived 0.9, estimated 0.65."""
    try:
        return config.QUALITY_WEIGHTS[quality]
    except KeyError:
        raise KeyError(f"unknown data quality {quality!r}; expected one of {sorted(config.QUALITY_WEIGHTS)}") from None
