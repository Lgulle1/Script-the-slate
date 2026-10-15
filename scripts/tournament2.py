"""Tournament 2 (yours, 2026-10-15): find the betting setup with the best return per unit staked on 2023-2026, then test it and the locked strategy A
blind on 2022, which neither has seen.

Tuning predictions: each season of 2023-2026 is predicted by a picker trained on the other three (2022 never used). Blind predictions: 2022 by a picker trained
on all graded 2023-2026 lines. FanDuel's four live markets only (receptions, receiving yards, rush yards, pass yards). Stakes: 1 unit per single and per parlay;
parlays are 2 legs with the 30% boost. Singles and parlays are chosen separately. Every candidate is logged (records/tuning/tournament2_*).

Knobs: picker (5 versions) x side (under / over / both) x edge cut-off (0-8 points) x history rule (none / 3+ games for the team) x ranking (picker chance / edge)
x how many (singles 1-10, parlays from the top 2-8) x per-game limit (2 / 3 / none) x per-player limit (1 / none; parlays always 1).

  python scripts/tournament2.py
"""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CR, ST = _load("card_replay"), _load("strategy")
CF, TN = CR.CF, CR.TN
OUTD = config.ROOT / "records" / "tuning"
FOUR = ["rec", "rec_yds", "rush_yds", "pass_yds"]
TUNE = (2023, 2024, 2025, 2026)
BLIND = 2022
MIN_SINGLES, MIN_PARLAYS = 60, 40


def history_counts(d: pl.DataFrame) -> np.ndarray:
    from live import results as RS
    plog = pl.concat([RS.played(s)[1].select("player_id", "team", "gameday") for s in range(2020, 2027)])
    day = lambda c: pl.col(c).cast(pl.String).str.slice(0, 10).str.to_date()
    h = plog.select("player_id", "team", day("gameday").alias("gameday")).unique()
    x = d.select("player_id", "team", day("gameday").alias("gameday")).with_row_index("i")
    j = x.join(h, on=["player_id", "team"], how="left", suffix="_h").filter(pl.col("gameday_h") < pl.col("gameday"))
    return x.join(j.group_by("i").agg(n=pl.len()), on="i", how="left")["n"].fill_null(0).to_numpy()


def pickers():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    hgb = lambda d, lr, leaf, it=200: (lambda: HistGradientBoostingClassifier(max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=leaf, l2_regularization=1.0, random_state=7))
    return {"P1 six averaged + layers (locked)": (CF.ensemble_makers(), True),
            "P2 six averaged, no layers": (CF.ensemble_makers(), False),
            "P3 one model + layers (tournament 1 winner)": ([hgb(3, 0.1, 150)], True),
            "P4 logistic stacker + layers": ([lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))], True),
            "P5 bigger model + layers": ([hgb(5, 0.05, 100, 400)], True)}


def predict(d, A, makers, layered):
    X = TN.feats(d, layered=layered)
    season = d["season"].to_numpy()
    graded = ~np.isnan(A.over_won.astype(float))
    p = np.full(d.height, np.nan)
    folds = [(s, [t for t in TUNE if t != s]) for s in TUNE] + [(BLIND, list(TUNE))]
    for target, train in folds:
        tr = np.flatnonzero(np.isin(season, train) & graded)
        te = np.flatnonzero(season == target)
        p[te] = np.mean([m().fit(X[tr], A.over_won[tr]).predict_proba(X[te])[:, 1] for m in makers], axis=0)
    return p


def bet_tables(d, A, p_over):
    """Per side: one row per (line) with the chosen side's chance, edge, decimal price, won."""
    base = d.select("season", "week", "game_id", "player_id", "market", "history").with_columns(over_won=pl.Series(A.over_won))
    eo, eu = p_over - A.be_o, (1 - p_over) - A.be_u
    out = {}
    for side in ("under", "over", "both"):
        if side == "under":
            ch, ed, dec, won = 1 - p_over, eu, A.dec_u, 1 - A.over_won
        elif side == "over":
            ch, ed, dec, won = p_over, eo, A.dec_o, A.over_won
        else:
            o = eo >= eu
            ch, ed, dec, won = np.where(o, p_over, 1 - p_over), np.maximum(eo, eu), np.where(o, A.dec_o, A.dec_u), np.where(o, A.over_won, 1 - A.over_won)
        t = base.with_columns(chance=pl.Series(ch), edge=pl.Series(ed), dec=pl.Series(dec), won=pl.Series(won))
        out[side] = t.filter(pl.col("market").is_in(FOUR) & pl.col("season").is_in([*TUNE, BLIND]) & pl.col("won").is_not_nan())
    return out


def by_week(t: pl.DataFrame) -> dict:
    W = {}
    for r in t.iter_rows(named=True):
        W.setdefault((r["season"], r["week"]), []).append(r)
    return W


def top(rows, n, gcap, pcap):
    out, g, p = [], {}, {}
    for r in rows:
        if pcap and p.get(r["player_id"], 0) >= pcap:
            continue
        if gcap and g.get(r["game_id"], 0) >= gcap:
            continue
        out.append(r)
        g[r["game_id"]] = g.get(r["game_id"], 0) + 1
        p[r["player_id"]] = p.get(r["player_id"], 0) + 1
        if len(out) == n:
            break
    return out


def parlays(legs):
    used, res = set(), []
    for i, a in enumerate(legs):
        if i in used:
            continue
        for j in range(i + 1, len(legs)):
            if j not in used and legs[j]["game_id"] != a["game_id"]:
                used |= {i, j}
                dec = a["dec"] * legs[j]["dec"]
                res.append((dec - 1) * (1 + ST.BOOST) if (a["won"] == 1 and legs[j]["won"] == 1) else -1.0)
                break
    return res


def evaluate(W, theta, hist, rank, kind, n, gcap, pcap):
    """(tuning bets, tuning units, blind bets, blind units, tuning weeks)."""
    tn = tu = bn = bu = 0.0
    for (s, w), rows in W.items():
        rr = [r for r in rows if r["edge"] >= theta and r["history"] >= hist]
        rr.sort(key=lambda r: -r[rank])
        if kind == "single":
            picks = top(rr, n, gcap, pcap)
            res = [(r["dec"] - 1) if r["won"] == 1 else -1.0 for r in picks]
        else:
            res = parlays(top(rr, n, gcap, 1))
        if s == BLIND:
            bn += len(res); bu += sum(res)
        else:
            tn += len(res); tu += sum(res)
    return tn, tu, bn, bu


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    d = TN.build()
    A = TN.Arr(d)
    d = d.with_columns(history=pl.Series(history_counts(d)))
    rows = []
    for pname, (makers, layered) in pickers().items():
        p_over = predict(d, A, makers, layered)
        tabs = bet_tables(d, A, p_over)
        print(pname, "predicted", flush=True)
        for side, t in tabs.items():
            W = by_week(t)
            for theta, hist, rank in itertools.product((0.0, 0.02, 0.04, 0.06, 0.08), (0, 3), ("chance", "edge")):
                for n, gcap, pcap in itertools.product((1, 2, 3, 4, 5, 6, 8, 10), (2, 3, None), (1, None)):
                    tn, tu, bn, bu = evaluate(W, theta, hist, rank, "single", n, gcap, pcap)
                    rows.append(dict(kind="singles", picker=pname, side=side, edge_min=theta, history_min=hist, rank_by=rank, top_n=n, per_game=str(gcap or "none"),
                                     per_player=str(pcap or "none"), tune_bets=int(tn), tune_units=tu, blind_bets=int(bn), blind_units=bu))
                for n, gcap in itertools.product((2, 4, 6, 8), (2, 3, None)):
                    tn, tu, bn, bu = evaluate(W, theta, hist, rank, "parlay", n, gcap, 1)
                    rows.append(dict(kind="parlays", picker=pname, side=side, edge_min=theta, history_min=hist, rank_by=rank, top_n=n, per_game=str(gcap or "none"),
                                     per_player="1", tune_bets=int(tn), tune_units=tu, blind_bets=int(bn), blind_units=bu))
    R = pl.DataFrame(rows).with_columns(tune_roi=pl.col("tune_units") / pl.col("tune_bets"), blind_roi=pl.col("blind_units") / pl.col("blind_bets"))
    R.write_parquet(OUTD / "tournament2_all.parquet")
    locked = dict(picker="P1 six averaged + layers (locked)", side="under", edge_min=0.04, history_min=3, rank_by="chance")
    lk = lambda kind, n, g, p: R.filter((pl.col("kind") == kind) & (pl.col("top_n") == n) & (pl.col("per_game") == g) & (pl.col("per_player") == p)
                                        & pl.all_horizontal([pl.col(k) == v for k, v in locked.items()]))
    out = {"candidates": R.height, "singles_candidates": R.filter(pl.col("kind") == "singles").height, "parlay_candidates": R.filter(pl.col("kind") == "parlays").height}
    for kind, mn in (("singles", MIN_SINGLES), ("parlays", MIN_PARLAYS)):
        ok = R.filter((pl.col("kind") == kind) & (pl.col("tune_bets") >= mn)).sort("tune_roi", descending=True)
        out[f"{kind}_top10"] = ok.head(10).to_dicts()
        out[f"{kind}_winner"] = ok.row(0, named=True)
        # does tuning rank predict 2022?
        q = ok.with_columns(r=pl.col("tune_roi").rank(descending=True) / ok.height)
        out[f"{kind}_rank_vs_blind"] = {nm: float(q.filter((pl.col("r") > lo) & (pl.col("r") <= hi))["blind_roi"].mean())
                                        for nm, lo, hi in (("top 1%", 0, .01), ("top 10%", 0, .1), ("middle", .45, .55), ("bottom 10%", .9, 1.0))}
    out["locked_singles"] = lk("singles", 2, "3", "none").row(0, named=True)
    out["locked_parlay"] = lk("parlays", 2, "3", "1").row(0, named=True)
    (OUTD / "tournament2_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("top10")}, indent=1, default=str))


if __name__ == "__main__":
    main()
