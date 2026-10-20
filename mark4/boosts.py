"""Re-score every version with the boosts the project manager actually has (2026-10-20): ONE 30% profit boost per window, four windows a week.

Windows (kickoff, US Eastern):  Thursday night | Sunday 1-4pm | Sunday night | Monday night.
  - A window with two or more games: one boosted 2-leg parlay, legs from two DIFFERENT games of that window (top two by chance).
  - A window with one game: one boosted straight bet on that game's best leg. (A same-game parlay is priced by FanDuel on its own, not as the
    product of its legs, so it cannot be scored from these lines.)
  Games outside the windows (Saturday, Thanksgiving day, London mornings) get no boost.
Strategies (on top of the boosts or alone): boosts only; boosts + the top N unboosted straight bets of the week (N 3 / 6); top N straights only.
Each with an edge bar (2 / 4 / 6%), unders only or either side, 4 or 7 markets.

Nested forward selection (the honest way to pick a winner on history): before season S, the version AND the strategy with the best recency-weighted
return over seasons < S are chosen; that pair is bet in S. So the luck of picking the best of many versions is inside the reported result.

  python -m mark4.boosts          # writes records/lab/boost_windows.json and boost_windows.csv
"""
from __future__ import annotations

import json

import duckdb
import numpy as np
import polars as pl

import config
from mark4 import forward as F

VERSIONS = {"Mark 3 det": "Mark 3 (frozen inputs)", "Mark 4.0 det": "Mark 4.0", "Mark 4d det": "Mark 4d", "Mark 4e det": "Mark 4e", "Mark 4f det": "Mark 4f",
            "Mark 4g det": "Mark 4g", "Mark 4h det": "Mark 4h"}
STRATS = [dict(kind=k, n=n, edge=e, sides=sd, markets=mk) for k, n in (("boosts", 0), ("boosts+sing", 3), ("boosts+sing", 6), ("singles", 6), ("singles", 10))
          for e in (0.02, 0.04, 0.06) for sd in ("under", "both") for mk in ("four", "all")]
MIN_PAST_STAKE, MIN_PAST_BETS = 30.0, 60


def windows(d: pl.DataFrame) -> np.ndarray:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, weekday, gametime FROM schedules WHERE game_type = 'REG' ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    w = (pl.when((pl.col("weekday") == "Thursday") & (pl.col("gametime") >= "19:00")).then(pl.lit("thu"))
         .when((pl.col("weekday") == "Sunday") & (pl.col("gametime") >= "12:00") & (pl.col("gametime") < "17:00")).then(pl.lit("sun"))
         .when((pl.col("weekday") == "Sunday") & (pl.col("gametime") >= "19:00")).then(pl.lit("snf"))
         .when((pl.col("weekday") == "Monday") & (pl.col("gametime") >= "19:00")).then(pl.lit("mnf")).otherwise(pl.lit("")))
    return d.select("game_id").join(sc.with_columns(win=w).select("game_id", "win"), on="game_id", how="left")["win"].fill_null("").to_numpy()


def strat_name(s):
    k = {"boosts": "4 boosts only", "boosts+sing": f"4 boosts + top {s['n']} straights", "singles": f"top {s['n']} straights, no boosts"}[s["kind"]]
    return f"{k} | edge {s['edge']:.0%} | {s['sides']} | {'4 markets' if s['markets'] == 'four' else '7 markets'}"


def run(W, kind, n, win_of):
    """{(season, week): (units, staked, bets)}; rows carry 'win' (window) and are sorted by chance."""
    out = {}
    for sw, rows in W.items():
        u = s = b = 0.0
        used = set()
        if kind in ("boosts", "boosts+sing"):
            for wname in ("thu", "sun", "snf", "mnf"):
                legs = [r for r in rows if r["win"] == wname]
                if not legs:
                    continue
                games = {r["game_id"] for r in legs}
                if len(games) >= 2:
                    first = legs[0]
                    second = next((r for r in legs[1:] if r["game_id"] != first["game_id"]), None)
                    if second is None:
                        continue
                    pr = (first["dec"] * second["dec"] - 1) * (1 + F.ST.BOOST) if (first["won"] == 1 and second["won"] == 1) else -1.0
                    u += pr; s += 1; b += 1
                    used |= {(first["game_id"], first["player_id"], first["market"]), (second["game_id"], second["player_id"], second["market"])}
                else:
                    r = legs[0]
                    u += (r["dec"] - 1) * (1 + F.ST.BOOST) if r["won"] == 1 else -1.0
                    s += 1; b += 1
                    used.add((r["game_id"], r["player_id"], r["market"]))
        if kind in ("boosts+sing", "singles"):
            rest = [r for r in rows if (r["game_id"], r["player_id"], r["market"]) not in used]
            for r in F.ST.top(rest, n, F.ST.SINGLES_CAPS["per_game"], None, chance="chance", player="player_id"):
                u += (r["dec"] - 1) if r["won"] == 1 else -1.0
                s += 1; b += 1
        if s:
            out[sw] = (u, s, b)
    return out


def evaluate(d, A, p, win):
    res = {}
    for s in STRATS:
        x = F.candidates(d, A, p, s["sides"], F.IB.FOUR if s["markets"] == "four" else None)
        x = x.join(pl.DataFrame(dict(game_id=d["game_id"].to_list(), win=list(win))).unique("game_id"), on="game_id", how="left").with_columns(pl.col("win").fill_null(""))
        W = {}
        for r in x.filter(pl.col("edge") >= s["edge"]).sort("chance", descending=True).iter_rows(named=True):
            W.setdefault((r["season"], r["week"]), []).append(r)
        per = run(W, s["kind"], s["n"], None)
        by = {}
        for (sea, wk), (u, st, b) in per.items():
            v = by.setdefault(int(sea), [0.0, 0.0, 0])
            v[0] += u; v[1] += st; v[2] += b
        res[strat_name(s)] = dict(seasons=by, weeks=per)
    return res


def past_roi(seasons, S):
    past = {k: v for k, v in seasons.items() if k < S}
    st = sum(F.season_rec_w(k, S) * v[1] for k, v in past.items())
    if st < MIN_PAST_STAKE or sum(v[1] for v in past.values()) < MIN_PAST_BETS:
        return None
    return sum(F.season_rec_w(k, S) * v[0] for k, v in past.items()) / st


def main():
    d, A, *_ = F.frame()
    win = windows(d)
    print("lines by window:", {w: int((win == w).sum()) for w in ("thu", "sun", "snf", "mnf", "")}, flush=True)
    allres = {}
    for k, label in VERSIONS.items():
        pf = F.OUT / f"forward_p_{k.replace(' ', '_')}_0.npy"
        if not pf.exists():
            continue
        allres[label] = evaluate(d, A, np.load(pf), win)
        print("scored", label, flush=True)
    rows = []
    # each version on its own, strategy chosen forward
    for label, res in allres.items():
        for S in F.SEASONS[1:]:
            best = max(((nm, past_roi(r["seasons"], S)) for nm, r in res.items()), key=lambda t: -1e9 if t[1] is None else t[1])
            v = res[best[0]]["seasons"].get(S, [0.0, 0.0, 0])
            rows.append(dict(version=label, selection="strategy chosen forward", season=S, strategy=best[0], units=v[0], staked=v[1], ret=v[0] / v[1] if v[1] else None))
        for fixed in ("4 boosts only | edge 4% | under | 4 markets", "4 boosts + top 3 straights | edge 4% | under | 4 markets"):
            for S in F.SEASONS[1:]:
                v = res[fixed]["seasons"].get(S, [0.0, 0.0, 0])
                rows.append(dict(version=label, selection=fixed, season=S, strategy=fixed, units=v[0], staked=v[1], ret=v[0] / v[1] if v[1] else None))
    # nested: version and strategy chosen together, forward
    for S in F.SEASONS[1:]:
        best = None
        for label, res in allres.items():
            for nm, r in res.items():
                pr = past_roi(r["seasons"], S)
                if pr is not None and (best is None or pr > best[2]):
                    best = (label, nm, pr)
        v = allres[best[0]][best[1]]["seasons"].get(S, [0.0, 0.0, 0])
        rows.append(dict(version="Nested: version + strategy chosen forward", selection=f"{best[0]}", season=S, strategy=best[1], units=v[0], staked=v[1],
                         ret=v[0] / v[1] if v[1] else None))
    t = pl.DataFrame(rows)
    t.write_csv(F.OUT / "boost_windows.csv")
    summ = (t.with_columns(w=0.5 ** (2026 - pl.col("season"))).group_by("version", "selection")
            .agg(recency_ret=(pl.col("w") * pl.col("units")).sum() / (pl.col("w") * pl.col("staked")).sum(), units=pl.col("units").sum(), staked=pl.col("staked").sum())
            .sort("version", "selection"))
    summ.write_csv(F.OUT / "boost_windows_summary.csv")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_width_chars(200)
    print(summ)
    print(t.filter(pl.col("version").str.starts_with("Nested")))


if __name__ == "__main__":
    main()
