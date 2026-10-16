"""Untested ideas, batch 1 (yours, 2026-10-16), all judged on the hidden seasons 2021-2026 exactly like tournament 3: each season's picks come from a picker
trained on the other five seasons only, then the locked rule (top 8 unders, 4+ edge, 3+ games with the team, up to 3 per game, each player once, 2-leg parlays
in rank order, 30% boost) or a variant of it is applied. Baseline = the locked strategy as it stands.

New picker inputs (picker retrained with them):
  schedule   rest days (team, opponent, difference), short week, division game, primetime, grass/turf
  weather    game temperature and wind (outdoor games; nflverse game-day values, so slightly better than a forecast would be)
  history    the player's own record against his lines in this market before this game: lines faced, share that went under, average (actual - line) / line
  movement   change in the under price from FanDuel's opening price (only where the file had an opening price at the same line: 10-35% of lines)
  all        all four together
Betting-rule variants (same picks as the baseline):
  3-leg      3-leg boosted parlays from the top 9 (3 a week) / top 12 (4 a week)
  more       2-leg parlays from the top 10 / top 12 (5 / 6 a week)
  floor      only legs the picker rates 60%+ (up to 8), so fewer parlays in weak weeks
  juice cap  drop legs priced worse than -150 / -135 (the boost is a share of profit; heavy favourites pay little)
  value rank rank legs by expected payout (chance x price) instead of chance
  no early   skip weeks 1-3 (thin current-season data)
  sizing     stake each parlay by its estimated value (Kelly-style, scaled to the same total stake)

  python scripts/ideas_batch.py      # writes records/tuning/ideas_batch1.json
"""
import importlib.util
import json
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
T2, TN, CF, ST = T3.T2, T3.TN, T3.CF, T3.ST
SEASONS = T3.SEASONS
FOUR = T3.FOUR


def schedule_feats(d):
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute("SELECT DISTINCT ON (game_id) game_id, weekday, gametime, away_rest, home_rest, div_game, surface, temp, wind, roof "
                     "FROM schedules ORDER BY game_id, pulled_at DESC").pl()
    con.close()
    x = d.select("game_id", "is_home").join(sc, on="game_id", how="left")
    h = x["is_home"].fill_null(0).to_numpy() == 1
    rest = np.where(h, x["home_rest"].to_numpy(), x["away_rest"].to_numpy()).astype(float)
    orest = np.where(h, x["away_rest"].to_numpy(), x["home_rest"].to_numpy()).astype(float)
    hour = np.array([int(str(t).split(":")[0]) if t else 13 for t in x["gametime"].to_list()])
    prime = ((hour >= 19) | x["weekday"].is_in(["Monday", "Thursday"]).to_numpy()).astype(float)
    grass = x["surface"].str.contains("grass").fill_null(False).to_numpy().astype(float)
    outdoor = ~x["roof"].is_in(["dome", "closed"]).fill_null(False).to_numpy()
    temp = np.where(outdoor, x["temp"].cast(pl.Float64).to_numpy(), np.nan)
    wind = np.where(outdoor, x["wind"].cast(pl.Float64).to_numpy(), np.nan)
    sched = np.column_stack([rest, orest, rest - orest, (rest <= 5).astype(float), x["div_game"].cast(pl.Float64).to_numpy(), prime, grass])
    return sched, np.column_stack([temp, wind])


def history_feats(d, A):
    """The player's record against his lines in this market strictly before this game (any season)."""
    x = d.select("player_id", "market", "game_id", "gameday", "line", "actual").with_row_index("i").with_columns(
        day=pl.col("gameday").cast(pl.String).str.slice(0, 10).str.to_date(), under=pl.Series(1 - A.over_won), rel=(pl.col("actual") - pl.col("line")) / pl.col("line"))
    g = (x.filter(pl.col("under").is_not_nan()).group_by("player_id", "market", "game_id", "day")
         .agg(u=pl.col("under").mean(), r=pl.col("rel").mean()).sort("player_id", "market", "day"))
    g = g.with_columns(n=pl.int_range(pl.len()).over("player_id", "market"),
                       su=pl.col("u").cum_sum().over("player_id", "market") - pl.col("u"),
                       sr=pl.col("r").cum_sum().over("player_id", "market") - pl.col("r"))
    g = g.with_columns(rate=pl.when(pl.col("n") > 0).then(pl.col("su") / pl.col("n")), avg_rel=pl.when(pl.col("n") > 0).then(pl.col("sr") / pl.col("n")))
    # rows of ungraded games (none in the hidden seasons) get the last known record via the join on game; graded rows get their own pre-game record
    y = x.join(g.select("player_id", "market", "game_id", "n", "rate", "avg_rel"), on=["player_id", "market", "game_id"], how="left").sort("i")
    return y.select(pl.col("n").cast(pl.Float64), "rate", "avg_rel").to_numpy()


def movement_feats(d):
    s = pl.read_parquet(TS_SEL()).filter(pl.col("side") == "under").select("game_id", "player_id", "market", "line", "price", "open_price")
    be = lambda c: pl.when(pl.col(c) > 0).then(100.0 / (pl.col(c) + 100.0)).otherwise(-pl.col(c) / (-pl.col(c) + 100.0))
    s = s.with_columns(move=be("price") - be("open_price")).unique(["game_id", "player_id", "market", "line"])
    y = d.select("game_id", "player_id", "market", "line").with_row_index("i").join(s.select("game_id", "player_id", "market", "line", "move"), on=["game_id", "player_id", "market", "line"], how="left").sort("i")
    return y["move"].to_numpy()[:, None]


def TS_SEL():
    return TN.TM.T.SEL


def hidden_preds(d, A, X, makers):
    season = d["season"].to_numpy()
    graded = ~np.isnan(A.over_won.astype(float))
    p = np.full(d.height, np.nan)
    for S in SEASONS:
        tr = np.isin(season, [t for t in SEASONS if t != S]) & graded
        te = season == S
        p[te] = np.mean([m().fit(X[tr], A.over_won[tr]).predict_proba(X[te])[:, 1] for m in makers], axis=0)
    return p


def week_rows(d, A, p_over):
    x = d.select("season", "week", "game_id", "player_id", "market", "history").with_columns(
        chance=pl.Series(1 - p_over), edge=pl.Series((1 - p_over) - A.be_u), dec=pl.Series(A.dec_u), won=pl.Series(1 - A.over_won))
    x = x.filter(pl.col("market").is_in(FOUR) & (pl.col("edge") >= 0.04) & (pl.col("history") >= CF.MIN_HISTORY) & pl.col("won").is_not_nan() & pl.col("chance").is_not_nan())
    W = {}
    for r in x.sort("chance", descending=True).iter_rows(named=True):
        W.setdefault((r["season"], r["week"]), []).append(r)
    return W


def groups(legs, k):
    """k-leg parlays in rank order, never two legs from one game."""
    used, out = set(), []
    for i, a in enumerate(legs):
        if i in used:
            continue
        grp, games = [i], {a["game_id"]}
        for j in range(i + 1, len(legs)):
            if j not in used and j not in grp and legs[j]["game_id"] not in games:
                grp.append(j)
                games.add(legs[j]["game_id"])
                if len(grp) == k:
                    break
        if len(grp) == k:
            used |= set(grp)
            out.append([legs[t] for t in grp])
    return out


def settle(par):
    dec = np.prod([l["dec"] for l in par])
    return (dec - 1) * (1 + ST.BOOST) if all(l["won"] == 1 for l in par) else -1.0


def run_rule(W, n=8, k=2, floor=None, max_juice=None, rank="chance", skip_early=False, sizing=False):
    per = {}
    for (s, w), rows in W.items():
        if skip_early and w <= 3:
            continue
        rr = rows
        if floor is not None:
            rr = [r for r in rr if r["chance"] >= floor]
        if max_juice is not None:
            rr = [r for r in rr if r["dec"] >= 1 + 100 / max_juice]
        if rank == "value":
            rr = sorted(rr, key=lambda r: -r["chance"] * r["dec"])
        legs = ST.top(rr, n, ST.PARLAY_CAPS["per_game"], ST.PARLAY_CAPS["per_player"], chance="chance" if rank == "chance" else "_v", player="player_id") if rank == "chance" else _top_keep(rr, n)
        pars = groups(legs, k)
        if not pars:
            continue
        res = [settle(p) for p in pars]
        if sizing:
            ev = np.array([max(np.prod([l["chance"] for l in p]) * np.prod([l["dec"] for l in p]) * (1 + ST.BOOST) - ST.BOOST - 1, 0.01) for p in pars])
            stake = ev / ev.mean()
            per.setdefault(s, [0.0, 0.0])
            per[s][0] += float(np.dot(stake, res)); per[s][1] += float(stake.sum())
        else:
            per.setdefault(s, [0.0, 0.0])
            per[s][0] += sum(res); per[s][1] += len(res)
    return per


def _top_keep(rows, n):
    """ST.top in the given order (rows already sorted)."""
    out, g, p = [], {}, {}
    for r in rows:
        if p.get(r["player_id"], 0) >= ST.PARLAY_CAPS["per_player"] or g.get(r["game_id"], 0) >= ST.PARLAY_CAPS["per_game"]:
            continue
        out.append(r)
        g[r["game_id"]] = g.get(r["game_id"], 0) + 1
        p[r["player_id"]] = p.get(r["player_id"], 0) + 1
        if len(out) == n:
            break
    return out


def summary(per):
    seas = {str(s): dict(units=round(u, 2), stake=round(b, 2), roi=round(u / b, 4) if b else None) for s, (u, b) in sorted(per.items())}
    U, B = sum(u for u, b in per.values()), sum(b for u, b in per.values())
    return dict(seasons=seas, units=round(U, 2), stake=round(B, 2), roi=round(U / B, 4) if B else None)


def singles(W, n=8):
    U = B = 0
    for rows in W.values():
        for r in ST.top(rows, n, ST.SINGLES_CAPS["per_game"], None, chance="chance", player="player_id"):
            U += (r["dec"] - 1) if r["won"] == 1 else -1.0
            B += 1
    return round(U / B, 4)


def main():
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X = TN.feats(d, layered=True)
    sched, wx = schedule_feats(d)
    hist = history_feats(d, A)
    mv = movement_feats(d)
    makers = CF.ensemble_makers()
    out = {"baseline": None, "features": {}, "rules": {}}
    variants = {"baseline": X, "schedule": np.hstack([X, sched]), "weather": np.hstack([X, wx]), "history": np.hstack([X, hist]),
                "movement": np.hstack([X, mv]), "all": np.hstack([X, sched, wx, hist, mv])}
    base_W = None
    for name, Xv in variants.items():
        p = hidden_preds(d, A, Xv, makers)
        W = week_rows(d, A, p)
        r = summary(run_rule(W))
        r["singles_top8_roi"] = singles(W)
        if name == "baseline":
            out["baseline"], base_W = r, W
        else:
            out["features"][name] = r
        print(f"{name:<10} parlays {r['roi']:+.1%} on {r['stake']:.0f}  singles top 8 {r['singles_top8_roi']:+.1%}  " +
              " ".join(f"{s}:{v['roi']:+.0%}" for s, v in r["seasons"].items()), flush=True)
    rules = {"3-leg from top 9 (3 a week)": dict(n=9, k=3), "3-leg from top 12 (4 a week)": dict(n=12, k=3),
             "2-leg from top 10 (5 a week)": dict(n=10), "2-leg from top 12 (6 a week)": dict(n=12),
             "floor 60%": dict(floor=0.60), "floor 58%": dict(floor=0.58), "juice cap -150": dict(max_juice=150), "juice cap -135": dict(max_juice=135),
             "value rank": dict(rank="value"), "skip weeks 1-3": dict(skip_early=True), "sizing by value": dict(sizing=True)}
    for name, kw in rules.items():
        r = summary(run_rule(base_W, **kw))
        out["rules"][name] = r
        print(f"{name:<30} {r['roi']:+.1%} on {r['stake']:.0f}  " + " ".join(f"{s}:{v['roi']:+.0%}" for s, v in r["seasons"].items()), flush=True)
    b = out["baseline"]["seasons"]
    for grp in ("features", "rules"):
        for name, r in out[grp].items():
            r["seasons_beating_baseline"] = sum(1 for s, v in r["seasons"].items() if s in b and v["roi"] is not None and v["roi"] > b[s]["roi"])
    (config.ROOT / "records" / "tuning" / "ideas_batch1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
