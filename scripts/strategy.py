"""The locked betting strategies (records/cards/locked_setup.json), as code shared by the track record and the weekly page.

S (bet since 2026-10-16, "Mark 3"): the same parlay rule as P, on the schedule picker's picks (scripts/schedule_picker.py), picker averaged over 5 seeds.
             Seed-averaged hidden seasons: +54% / +47% (two seed groups) vs Mark 2's +28% / +28%; P(better) 97% / 92% (scripts/seed_ensembles.py).
SS (tracked): Mark 3's top 8 as straight bets. C3 (tracked): Mark 3's picks as 3-leg parlays from the top 9.
P (Mark 2: bet 2026-10-15 to 2026-10-16, now tracked): no straight bets; 2-leg boosted parlays in rank order from the top 8 unders (#1+#2, #3+#4, ...): up to 4 a week, one per boost
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
    "S": dict(name="Mark 3 (bet): boosted 2-leg parlays from the schedule picker's top 8", singles=0, parlay_from=8, role="bet", pool="schedule"),
    "SS": dict(name="Mark 3 straight bets: the schedule picker's top 8 (tracked)", singles=8, parlay_from=0, role="tracked", pool="schedule"),
    "C3": dict(name="Mark 3 variant: 3-leg parlays from the top 9 (tracked)", singles=0, parlay_from=9, legs=3, role="candidate", pool="schedule"),
    "P": dict(name="Mark 2: boosted 2-leg parlays from the top 8 (tracked)", singles=0, parlay_from=8, role="tracked"),
    "A": dict(name="Mark 1 A: top 2 + 1 parlay", singles=2, parlay_from=2, role="tracked"),
    "B": dict(name="Mark 1 B: top 8 + 4 parlays", singles=8, parlay_from=8, role="tracked"),
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


def groups(legs: list, k: int = 2) -> list:
    """k-leg parlays in rank order: each unused leg with the next unused legs from games not already in the parlay."""
    if k == 2:
        return pairs(legs)
    used, out = set(), []
    for i, a in enumerate(legs):
        if i in used:
            continue
        grp, games = [i], {a["game_id"]}
        for j in range(i + 1, len(legs)):
            if j not in used and legs[j]["game_id"] not in games:
                grp.append(j)
                games.add(legs[j]["game_id"])
                if len(grp) == k:
                    break
        if len(grp) == k:
            used |= set(grp)
            out.append(tuple(legs[t] for t in grp))
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
    plegs = top(rows, s["parlay_from"], PARLAY_CAPS["per_game"], PARLAY_CAPS["per_player"], player=player) if s["parlay_from"] else []
    return dict(singles=singles, parlays=groups(plegs, s.get("legs", 2)))


def single_profit(r) -> float | None:
    if r.get("won") is None:
        return None
    return r["dec_u"] - 1 if r["won"] == 1 else -1.0


def parlay_profit(*legs) -> float | None:
    if any(l.get("won") is None for l in legs):
        return None
    dec = 1.0
    for l in legs:
        dec *= l["dec_u"]
    return (dec - 1) * (1 + BOOST) if all(l["won"] == 1 for l in legs) else -1.0
