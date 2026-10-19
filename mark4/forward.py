"""Mark 4 forward test (2026-10-19): every season played forward, the way it would have been bet live.

For each season S (2022-2026) and each block of weeks (1-6, 7-12, 13-18), the picker is trained ONLY on lines graded before the block starts (earlier
seasons and the earlier weeks of S), then picks the block. With recency weighting a training line counts 0.5 per HALF_LIFE seasons back from the block.
The betting strategy is chosen forward too: before season S, the strategy with the best recency-weighted return over the forward results of the seasons
before S is the one bet in S. Nothing from S or later is ever seen when betting S. Mark 3 is run through the same protocol for a fair comparison.

Inputs a picker can use:
  base        Mark 3's inputs (the locked picker inputs + schedule)
  situations  the blind-spot situation columns (wind, temp, QB change, new coach, rest, spread, total, division, primetime, vacated carries/targets,
              own shares, injury status, played hurt last game, games since last, line vs last-4 average, last-3 lines over count, price move, defense vs line)
  comps       the comps v3 chains vs our projection and vs the line (dfn, int, sit, mean, agreement, chains found)
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import config

HERE = config.ROOT / "scripts"
HALF_LIFE = 1.0
BLOCKS = (1, 7, 13)
SEASONS = (2022, 2023, 2024, 2025, 2026)
MIN_PAST_STAKE, MIN_PAST_BETS = 30.0, 60     # a strategy is only chosen on a real past sample (2025 once picked one on 9 parlays)
OUT = config.ROOT / "records" / "lab"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


M3T = _load("mark3_tests")
IB, TN, T2, CF, ST = M3T.IB, M3T.TN, M3T.T2, M3T.CF, M3T.ST

SIT_COLS = ("wind", "temp", "dome", "qb_change", "new_coach", "rest", "opp_rest", "team_spread", "total_line", "div_game", "prime", "vac_car", "vac_tgt",
            "own_car", "own_tgt", "status", "hurt_last", "games_since_last", "line_vs_form", "last3_overs", "move_over", "dfn_line")
COMP_COLS = ("dfn_eff", "int_eff", "sit_eff", "chains_mean", "agree", "n_chains")


def frame():
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(T2.history_counts(d)))
    X3 = np.hstack([TN.feats(d, layered=True), IB.schedule_feats(d)[0]])
    from mark4 import blindspots as BS
    b = BS.frame().filter(pl.col("side") == "over").unique(["game_id", "player_id", "market"])
    b = b.with_columns(dome=pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64), prime=((pl.col("weekday") != "Sunday") | (pl.col("gametime") >= "19:00")).cast(pl.Float64),
                       qb_change=pl.col("qb_change").cast(pl.Float64), new_coach=pl.col("new_coach").cast(pl.Float64), line_vs_form=pl.col("line") / pl.col("prev4"))
    if "dfn_line" not in b.columns:
        b = b.with_columns(dfn_line=pl.lit(None, pl.Float64))
    k = ["game_id", "player_id", "market"]
    S = d.select(k).join(b.select(*k, *SIT_COLS), on=k, how="left").select(SIT_COLS).cast(pl.Float64, strict=False).to_numpy()
    from mark4 import book_lag as BL
    ev = BL.event_nth()
    E = d.select(k).join(ev, on=k, how="left").select(list(BL.EVENT_KEYS.values())).cast(pl.Float64).fill_null(0).to_numpy()
    comps = []
    for outcome in ("relc", "rline"):
        p = config.PROCESSED_DIR / f"comps3_chains_all_{outcome}.parquet"
        c = pl.read_parquet(p).select(*k, *[pl.col(x).alias(f"{x}_{outcome}") for x in COMP_COLS])
        comps.append(d.select(k).join(c, on=k, how="left").drop(k).to_numpy().astype(float))
    C = np.hstack(comps)
    S, C = (np.where(np.isfinite(z), z, np.nan) for z in (S, C))      # a line over a last-4 average of 0 is inf
    sp = config.PROCESSED_DIR / "mark4_spot_score.parquet"
    P = d.select(k).join(pl.read_parquet(sp), on=k, how="left").drop(k).to_numpy().astype(float) if sp.exists() else np.full((d.height, 1), np.nan)
    return d, A, X3, S, C, E, P


def det_makers():
    """The same six model shapes as Mark 3's picker (depth 2/3/4 x leaf 50/150) but with NO random hold-out: a fixed number of rounds, so the same data
    always gives the same picks and a difference between two versions is never seed luck."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return [lambda d=d, l=l: HistGradientBoostingClassifier(max_depth=d, learning_rate=0.05, max_iter=120, min_samples_leaf=l, l2_regularization=1.0,
                                                            early_stopping=False) for d in (2, 3, 4) for l in (50, 150)]


def forward_preds(d, A, X, seeds, recency: bool, half_life=HALF_LIFE, seasons=SEASONS, det=False):
    season, week = d["season"].to_numpy(), d["week"].to_numpy()
    graded = ~np.isnan(A.over_won.astype(float))
    t_row = season + (week - 1) / 18.0
    p = np.full(d.height, np.nan)
    makers = det_makers() if det else [m for s in seeds for m in M3T.makers(s)]
    for S in seasons:
        for b0 in BLOCKS:
            te = (season == S) & (week >= b0) & (week < b0 + 6)
            if not te.any():
                continue
            tr = graded & ((season < S) | ((season == S) & (week < b0)))
            t0 = S + (b0 - 1) / 18.0
            w = 0.5 ** ((t0 - t_row[tr]) / half_life) if recency else np.ones(tr.sum())
            keep = ~np.all(np.isnan(X[tr]), axis=0)        # a column not built yet in the training years (comps before 2022) is left out
            preds = []
            for mk in makers:
                m = mk()
                m.fit(X[tr][:, keep], A.over_won[tr].astype(int), sample_weight=w)
                preds.append(m.predict_proba(X[te][:, keep])[:, 1])
            p[te] = np.mean(preds, axis=0)
            print(f"   block {S} week {b0}+ done", flush=True)
    return p


# ------------------------------------------------------------------ strategies

def candidates(d, A, p, sides="under", markets=None, min_history=None):
    """Every line with the side chosen (best edge, or under only), its chance, edge, decimal price and result."""
    mh = CF.MIN_HISTORY if min_history is None else min_history
    eo, eu = p - A.be_o, (1 - p) - A.be_u
    over = (eo > eu) if sides == "both" else np.zeros(len(p), bool)
    x = d.select("season", "week", "game_id", "player_id", "market", "history").with_columns(
        side=pl.Series(np.where(over, "over", "under")), chance=pl.Series(np.where(over, p, 1 - p)), edge=pl.Series(np.where(over, eo, eu)),
        dec=pl.Series(np.where(over, A.dec_o, A.dec_u)), won=pl.Series(np.where(over, A.over_won, 1 - A.over_won).astype(float)))
    x = x.filter(pl.col("chance").is_not_nan() & pl.col("won").is_not_nan() & (pl.col("history") >= mh))
    if markets:
        x = x.filter(pl.col("market").is_in(list(markets)))
    return x


def weekly(x: pl.DataFrame, min_edge: float):
    W = {}
    for r in x.filter(pl.col("edge") >= min_edge).sort("chance", descending=True).iter_rows(named=True):
        W.setdefault((r["season"], r["week"]), []).append(r)
    return W


def run_strategy(W, kind, n):
    """{(season, week): (units, staked, bets)}.
    singles:      top n straight bets, 1 unit each
    boost1:       one boosted 2-leg parlay from the top 2 (one 30% profit boost a week) + nothing else
    boost1+sing:  the boosted parlay from the top 2 + straight bets on the next n
    mark3:        up to 4 boosted 2-leg parlays from the top 8 (Mark 3's rule)"""
    out = {}
    for sw, rows in W.items():
        u = s = b = 0.0
        if kind == "mark3":
            legs = ST.top(rows, 8, ST.PARLAY_CAPS["per_game"], ST.PARLAY_CAPS["per_player"], chance="chance", player="player_id")
            for par in IB.groups(legs, 2):
                u += IB.settle(par); s += 1; b += 1
        else:
            if kind in ("boost1", "boost1+sing"):
                legs = ST.top(rows, 2, ST.PARLAY_CAPS["per_game"], ST.PARLAY_CAPS["per_player"], chance="chance", player="player_id")
                if len(legs) == 2:
                    u += IB.settle(legs); s += 1; b += 1
                used = {(l["game_id"], l["player_id"], l["market"]) for l in legs}
                rows = [r for r in rows if (r["game_id"], r["player_id"], r["market"]) not in used]
            if kind in ("singles", "boost1+sing"):
                for r in ST.top(rows, n, ST.SINGLES_CAPS["per_game"], None, chance="chance", player="player_id"):
                    u += (r["dec"] - 1) if r["won"] == 1 else -1.0; s += 1; b += 1
        if s:
            out[sw] = (u, s, b)
    return out


STRATS = [dict(kind=k, n=n, edge=e, sides=sd, markets=mk)
          for k, n in (("mark3", 8), ("boost1", 0), ("singles", 3), ("singles", 5), ("singles", 8), ("boost1+sing", 3), ("boost1+sing", 6))
          for e in (0.02, 0.04, 0.06, 0.08) for sd in ("under", "both") for mk in ("four", "all")]


def strat_name(s):
    k = {"mark3": "4 boosted parlays (Mark 3 rule)", "boost1": "1 boosted parlay", "singles": f"top {s['n']} straights",
         "boost1+sing": f"1 boosted parlay + top {s['n']} straights"}[s["kind"]]
    return f"{k} | edge {s['edge']:.0%} | {s['sides']} | {'4 markets' if s['markets'] == 'four' else '7 markets'}"


def power_metrics(d, A, p, target=2026):
    """High-power checks on every line (thousands, not ~200 parlays):
      logloss_gain  recency-weighted log loss of the book's no-vig over chance minus the model's (positive = the model knows more than the price)
      top20_under / top20_both  return of the top 20 straight bets a week (edge >= 0), unders only / either side."""
    season = d["season"].to_numpy()
    ok = ~np.isnan(p) & ~np.isnan(A.over_won.astype(float)) & ~np.isnan(A.fair_o.astype(float))
    y = A.over_won[ok].astype(float)
    pm, pb = np.clip(p[ok], 1e-4, 1 - 1e-4), np.clip(A.fair_o[ok].astype(float), 1e-4, 1 - 1e-4)
    w = 0.5 ** ((target - season[ok]) / HALF_LIFE)
    ll = lambda q: -(y * np.log(q) + (1 - y) * np.log(1 - q))
    out = dict(logloss_gain=float((w * (ll(pb) - ll(pm))).sum() / w.sum()),
               logloss_gain_by_season={int(s): float((ll(pb) - ll(pm))[season[ok] == s].mean()) for s in np.unique(season[ok])})
    for sides in ("under", "both"):
        per = run_strategy(weekly(candidates(d, A, p, sides), 0.0), "singles", 20)
        st = sum(season_rec_w(s_, target) * v[1] for (s_, _), v in per.items())
        un = sum(season_rec_w(s_, target) * v[0] for (s_, _), v in per.items())
        out[f"top20_{sides}"] = un / st if st else None
        out[f"top20_{sides}_bets"] = int(sum(v[1] for v in per.values()))
    return out


def season_rec_w(season, target):
    return 0.5 ** ((target - season) / HALF_LIFE)


def evaluate(d, A, p, label, fixed=None):
    """Every strategy's forward results per season, and the forward-chosen strategy's results (strategy for S picked on seasons < S, recency-weighted)."""
    four = IB.FOUR
    res = {}
    for s in STRATS:
        x = candidates(d, A, p, s["sides"], four if s["markets"] == "four" else None)
        per = run_strategy(weekly(x, s["edge"]), s["kind"], s["n"])
        by = {}
        for (sea, wk), (u, st, b) in per.items():
            v = by.setdefault(int(sea), [0.0, 0.0, 0])
            v[0] += u; v[1] += st; v[2] += b
        res[strat_name(s)] = dict(spec=s, seasons=by, weeks=per)
    chosen = {}
    for S in SEASONS[1:]:
        best, bs = None, -1e9
        for nm, r in res.items():
            if fixed and nm != fixed:
                continue
            past = {k: v for k, v in r["seasons"].items() if k < S}
            stake = sum(season_rec_w(k, S) * v[1] for k, v in past.items())
            if stake < MIN_PAST_STAKE or sum(v[1] for v in past.values()) < MIN_PAST_BETS:
                continue
            roi = sum(season_rec_w(k, S) * v[0] for k, v in past.items()) / stake
            if roi > bs:
                best, bs = nm, roi
        if best:
            v = res[best]["seasons"].get(S, [0.0, 0.0, 0])
            chosen[S] = dict(strategy=best, units=v[0], staked=v[1], roi=v[0] / v[1] if v[1] else None, picked_on_past_roi=bs)
    return res, chosen


def summarize(chosen, target=2026):
    st = sum(season_rec_w(s, target) * c["staked"] for s, c in chosen.items())
    un = sum(season_rec_w(s, target) * c["units"] for s, c in chosen.items())
    return dict(recency_roi=un / st if st else None, units=sum(c["units"] for c in chosen.values()), staked=sum(c["staked"] for c in chosen.values()))


SETUPS = {
    "Mark 3 (forward)": dict(cols=("base",), recency=False),
    "Mark 3 inputs + recency": dict(cols=("base",), recency=True),
    "Mark 4a: + situations, recency": dict(cols=("base", "sit"), recency=True),
    "Mark 4b: + situations + comps, recency": dict(cols=("base", "sit", "comps"), recency=True),
    "Mark 4c: + situations + role-event games + comps, recency": dict(cols=("base", "sit", "events", "comps"), recency=True),
    "Mark 4d: + blind-spot score, recency": dict(cols=("base", "spot"), recency=True),
    "Mark 3 det": dict(cols=("base",), recency=False, det=True),
    "Mark 4.0 det": dict(cols=("base",), recency=True, det=True),
    "Mark 4d det": dict(cols=("base", "spot"), recency=True, det=True),
}


def main(names=None, seeds=(1, 2, 3)):
    names = names or None
    t0 = time.time()
    d, A, X3, S, C, E, P = frame()
    blocks = dict(base=X3, sit=S, comps=C, events=E, spot=P)
    print(f"frame {d.height} rows, base {X3.shape[1]}, situations {S.shape[1]}, comps {C.shape[1]} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for name, cfg in SETUPS.items():
        if names and name not in names:
            continue
        X = np.hstack([blocks[c] for c in cfg["cols"]])
        pf = OUT / f"forward_p_{name.split(':')[0].replace(' ', '_')}_{'-'.join(map(str, seeds))}.npy"
        if pf.exists():
            p = np.load(pf)
        else:
            p = forward_preds(d, A, X, seeds, cfg["recency"], det=cfg.get("det", False))
            np.save(pf, p)
        res, chosen = evaluate(d, A, p, name)
        _, m3rule = evaluate(d, A, p, name, fixed=strat_name(dict(kind="mark3", n=8, edge=0.04, sides="under", markets="four")))
        out[name] = dict(chosen=chosen, summary=summarize(chosen), mark3_rule={str(k): v for k, v in m3rule.items()}, mark3_rule_summary=summarize(m3rule),
                         strategies={nm: {str(k): v for k, v in r["seasons"].items()} for nm, r in res.items()})
        out[name]["power"] = power_metrics(d, A, p)
        pw = out[name]["power"]
        print(f"   POWER {name}: log-loss gain vs the book {pw['logloss_gain']:+.4f} | top-20 straights: unders {pw['top20_under']:+.1%} on {pw['top20_under_bets']}, "
              f"either side {pw['top20_both']:+.1%} on {pw['top20_both_bets']} | by season {({k: round(v, 4) for k, v in pw['logloss_gain_by_season'].items()})}", flush=True)
        sm = out[name]["summary"]
        print(f"\n== {name} ({time.time() - t0:.0f}s): forward-chosen strategy, recency-weighted return {sm['recency_roi']:+.1%}, units {sm['units']:+.1f} on {sm['staked']:.0f}", flush=True)
        for s_, c in chosen.items():
            print(f"   {s_}: {c['roi']:+.1%} on {c['staked']:.0f}  [{c['strategy']}]  (chosen on past {c['picked_on_past_roi']:+.1%})", flush=True)
        m3 = out[name]["mark3_rule_summary"]
        print(f"   Mark 3 rule on these picks: recency-weighted {m3['recency_roi']:+.1%}, units {m3['units']:+.1f}", flush=True)
    jf = OUT / f"forward_{'-'.join(map(str, seeds))}.json"
    old = json.loads(jf.read_text()) if jf.exists() else {}
    old.update(out)
    jf.write_text(json.dumps(old, indent=1, default=str))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    seeds = tuple(int(x) for x in a[0].split(",")) if a else (1, 2, 3)
    main(names=a[1:] or None, seeds=seeds)
