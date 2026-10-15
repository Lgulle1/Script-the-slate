"""The locked betting strategies (records/cards/locked_setup.json), as code shared by the track record and the weekly page.

P (bet, locked 2026-10-15): no straight bets; 2-leg boosted parlays in rank order from the top 8 unders (#1+#2, #3+#4, ...): up to 4 a week, one per boost
             available, top pairs first; no boost, no parlay. Chosen on hidden-season tests (tournaments 3 and 4, blind 2021-2023).
A (retired, tracked): singles = the top 2 unders; parlay = #1 + #2 (2 legs, different games; none that week if they share a game)
B (tracked): singles = the top 8 unders; parlays = 2-leg in rank order from the top 8 (#1+#2, #3+#4, ...); its parlays are exactly P
Ranking: the picker's chance the under hits, highest first, among qualifying picks (FanDuel market, edge >= 4 points, player with 3+ games for his team).
Singles lists allow up to 3 picks per game and any number per player; parlay lists allow up to 3 per game and 1 per player (each the better rule by return per unit
staked over the 38 replayed weeks). Stakes: 1 unit per single, 1 unit per parlay; parlays carry the 30% profit boost.
"""
from __future__ import annotations

BOOST = 0.30
STRATEGIES = {
    "P": dict(name="Locked: 4 boosted parlays from the top 8", singles=0, parlay_from=8, role="bet"),
    "A": dict(name="Old A: top 2 + 1 parlay", singles=2, parlay_from=2, role="tracked"),
    "B": dict(name="Old B: top 8 + 4 parlays", singles=8, parlay_from=8, role="tracked"),
}
SINGLES_CAPS = dict(per_game=3, per_player=None)
PARLAY_CAPS = dict(per_game=3, per_player=1)


def top(rows: list, n: int, per_game=None, per_player=None, chance="picker_under_chance", player="player_id") -> list:
    """The first n rows by chance, skipping any that would break the per-game / per-player limit (only picks already taken count)."""
    out, g, p = [], {}, {}
    for r in sorted(rows, key=lambda r: -r[chance]):
        if per_player and p.get(r[player], 0) >= per_player:
            continue
        if per_game and g.get(r["game_id"], 0) >= per_game:
            continue
        out.append(r)
        g[r["game_id"]] = g.get(r["game_id"], 0) + 1
        p[r[player]] = p.get(r[player], 0) + 1
        if len(out) == n:
            break
    return out


def pairs(legs: list) -> list:
    """2-leg parlays in rank order: each unused leg with the next unused leg from a different game."""
    used, out = set(), []
    for i, a in enumerate(legs):
        if i in used:
            continue
        for j in range(i + 1, len(legs)):
            if j not in used and legs[j]["game_id"] != a["game_id"]:
                used |= {i, j}
                out.append((a, legs[j]))
                break
    return out


def card(rows: list, key: str, player="player_id") -> dict:
    """Singles and parlays of strategy `key` for one week's qualifying picks."""
    s = STRATEGIES[key]
    singles = top(rows, s["singles"], SINGLES_CAPS["per_game"], SINGLES_CAPS["per_player"], player=player) if s["singles"] else []
    plegs = top(rows, s["parlay_from"], PARLAY_CAPS["per_game"], PARLAY_CAPS["per_player"], player=player)
    return dict(singles=singles, parlays=pairs(plegs))


def single_profit(r) -> float | None:
    if r.get("won") is None:
        return None
    return r["dec_u"] - 1 if r["won"] == 1 else -1.0


def parlay_profit(a, b) -> float | None:
    if a.get("won") is None or b.get("won") is None:
        return None
    dec = a["dec_u"] * b["dec_u"]
    return (dec - 1) * (1 + BOOST) if (a["won"] == 1 and b["won"] == 1) else -1.0
