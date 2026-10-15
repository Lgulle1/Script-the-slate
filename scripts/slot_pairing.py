"""Slot pairing test (yours, 2026-10-15): pair the locked parlays' legs within the same kickoff slot first, across slots only for the leftovers?

Same legs as the locked strategy (the picker's top 8 unders, 4+ edge, 3+ games with the team, up to 3 per game, each player once), same hidden-season
predictions as tournament 3 (each season predicted by a picker trained on the other five), 30% boost. Only the pairing changes:
  rank     #1+#2, #3+#4, ... (the locked rule; any slots, never one game)
  slot     within each kickoff slot (Thursday, Sunday morning, Sunday 1 PM, Sunday 4 PM, Sunday night, Monday, other) pair in rank order (never one game);
           the leftovers are then paired across slots in rank order
  random   200 random pairings of the same 8 legs (never one game): how much pairing alone moves the result

  python scripts/slot_pairing.py
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T3 = _load("tournament3")
T2, TN, ST = T3.T2, T3.TN, T3.ST


def slot_of(weekday, gametime):
    h = int(str(gametime).split(":")[0])
    if weekday == "Sunday":
        return "Sun AM" if h < 12 else "Sun 1PM" if h < 15 else "Sun 4PM" if h < 19 else "Sun night"
    return weekday


def pair_rank(legs):
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


def pair_slot(legs):
    out, left = [], []
    for s in dict.fromkeys(r["slot"] for r in legs):
        grp = [r for r in legs if r["slot"] == s]
        prs = pair_rank(grp)
        out += prs
        paired = {id(x) for p in prs for x in p}
        left += [r for r in grp if id(r) not in paired]
    left.sort(key=lambda r: -r["chance"])
    return out + pair_rank(left)


def pair_random(legs, rng):
    for _ in range(50):
        x = legs[:]
        rng.shuffle(x)
        prs = pair_rank(x)
        if len(prs) == len(pair_rank(legs)):
            return prs
    return pair_rank(legs)


def profit(a, b):
    return (a["dec"] * b["dec"] - 1) * (1 + ST.BOOST) if a["won"] == 1 and b["won"] == 1 else -1.0


def main():
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    makers, layered = T2.pickers()["P1 six averaged + layers (locked)"]
    p_over = T3.fold_predictions(d, A, makers, layered)["ALL"]           # every season predicted by a picker trained on the other five
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute("SELECT DISTINCT ON (game_id) game_id, weekday, gametime FROM schedules ORDER BY game_id, pulled_at DESC").pl()
    con.close()
    slot = {r["game_id"]: slot_of(r["weekday"], r["gametime"]) for r in sc.iter_rows(named=True)}
    x = d.select("season", "week", "game_id", "player_id", "market", "history").with_columns(
        chance=pl.Series(1 - p_over), edge=pl.Series((1 - p_over) - A.be_u), dec=pl.Series(A.dec_u), won=pl.Series(1 - A.over_won))
    x = x.filter(pl.col("market").is_in(T3.FOUR) & (pl.col("edge") >= 0.04) & (pl.col("history") >= 3) & pl.col("won").is_not_nan() & pl.col("chance").is_not_nan())
    rng = random.Random(7)
    res = {k: {} for k in ("rank", "slot", "random")}
    same_slot_share = []
    for (s, w), g in sorted(x.group_by("season", "week"), key=lambda t: t[0]):
        rows = [dict(game_id=r["game_id"], player=r["player_id"], chance=r["chance"], dec=r["dec"], won=r["won"], slot=slot.get(r["game_id"], "?"))
                for r in g.sort("chance", descending=True).iter_rows(named=True)]
        legs = ST.top(rows, 8, ST.PARLAY_CAPS["per_game"], ST.PARLAY_CAPS["per_player"], chance="chance", player="player")
        for k, prs in (("rank", pair_rank(legs)), ("slot", pair_slot(legs))):
            u = [profit(a, b) for a, b in prs]
            res[k].setdefault(s, []).extend(u)
            if k == "slot":
                same_slot_share += [a["slot"] == b["slot"] for a, b in prs]
        res["random"].setdefault(s, []).append([[profit(a, b) for a, b in pair_random(legs, rng)] for _ in range(200)])
    out = {}
    for k in ("rank", "slot"):
        out[k] = {str(s): dict(units=float(sum(v)), bets=len(v), roi=float(sum(v) / len(v))) for s, v in res[k].items()}
        U, B = sum(sum(v) for v in res[k].values()), sum(len(v) for v in res[k].values())
        out[k]["all"] = dict(units=U, bets=B, roi=U / B)
    # random pairings: distribution of the all-season return over 200 draws
    draws = []
    for i in range(200):
        U = B = 0
        for s, weeks in res["random"].items():
            for wk in weeks:
                U += sum(wk[i]); B += len(wk[i])
        draws.append(U / B)
    out["random"] = dict(median=float(np.median(draws)), p05=float(np.percentile(draws, 5)), p95=float(np.percentile(draws, 95)))
    out["slot_pairs_same_slot_share"] = float(np.mean(same_slot_share))
    (config.ROOT / "records" / "tuning" / "slot_pairing.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
