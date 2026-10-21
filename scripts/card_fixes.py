"""Backtest three fixes to the weekly slate card on 2026 weeks 1-5 (yours, 2026-10-15):
  A  at most one pick per player and two per game when filling the tiers
  B  skip players with fewer than 3 earlier games for their current team
  C  average several picker versions (depth 2/3/4, rate 0.1, leaf 50/150) instead of one
and all three together. Same retrain-every-week procedure as scripts/slate_cards.py. Writes records/cards/fixes_*.

  python scripts/card_fixes.py
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("slate_cards", HERE / "slate_cards.py")
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)
WB, TN = SC.WB, SC.TN
OUTD = config.ROOT / "records" / "cards"
MIN_HISTORY = 3


SEEDS = (1, 2, 3, 4, 5)       # live picker (2026-10-16): six versions x five seeds averaged; one seed moved hidden-season returns by +-15 points (seed_ensembles.py)
LIVE_DETERMINISTIC = True    # 2026-10-20: live cards use the same six versions with NO random hold-out (fixed rounds), so the same data always gives the same
                             # picks; seed-averaged runs still swung forward results by 20-40 points (records/lab/forward_*.json). Inputs and rules unchanged.


def ensemble_makers(seeds=(7,)):
    """The six picker versions (depth 2/3/4 x leaf 50/150), once per seed. Default: the single seed every study before 2026-10-16 used."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return [lambda d=d, l=l, s=s: HistGradientBoostingClassifier(max_depth=d, learning_rate=0.1, max_iter=200, min_samples_leaf=l, l2_regularization=1.0, random_state=s)
            for s in seeds for d in (2, 3, 4) for l in (50, 150)]


def deterministic_makers():
    """The six picker versions (depth 2/3/4 x leaf 50/150) without early stopping: no random validation split, a fixed number of rounds (mark4.forward.det_makers)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return [lambda d=d, l=l: HistGradientBoostingClassifier(max_depth=d, learning_rate=0.05, max_iter=120, min_samples_leaf=l, l2_regularization=1.0,
                                                            early_stopping=False) for d in (2, 3, 4) for l in (50, 150)]


def history_counts(d: pl.DataFrame) -> np.ndarray:
    """Games the player played for the same team before this game (2024 on)."""
    from live import results as RS
    plog = pl.concat([RS.played(s)[1].select("player_id", "team", "gameday") for s in (2024, 2025, 2026)])
    pl_ = plog.select("player_id", "team", pl.col("gameday").cast(pl.String).str.slice(0, 10).str.to_date()).unique()
    x = d.select("game_id", "player_id", "team", pl.col("gameday").cast(pl.String).str.slice(0, 10).str.to_date()).with_row_index("i")
    j = x.join(pl_, on=["player_id", "team"], how="left", suffix="_h").filter(pl.col("gameday_h") < pl.col("gameday"))
    c = j.group_by("i").agg(n=pl.len())
    return x.join(c, on="i", how="left")["n"].fill_null(0).to_numpy()


def picks_for(d, A, X, season, week, ensemble):
    # graded lines only: a line whose player did not play (void) has no result (2026-10-21: week 6's voided lines reached week 7's training rows)
    tr = np.flatnonzero((((d["season"] < season) | ((d["season"] == season) & (d["week"] < week))).to_numpy()) & ~np.isnan(A.over_won.astype(float)))
    te = np.flatnonzero(((d["season"] == season) & (d["week"] == week)).to_numpy())
    makers = (deterministic_makers() if LIVE_DETERMINISTIC else ensemble_makers(SEEDS)) if ensemble else [WB.make]   # live cards and replay pools
    p_over = np.mean([m().fit(X[tr], A.over_won[tr]).predict_proba(X[te])[:, 1] for m in makers], axis=0)
    p_under = 1 - p_over
    edge = p_under - A.be_u[te]
    keep = (edge >= WB.THETA) & np.isin(A.mi[te], [TN.MARKETS.index(m) for m in WB.MARKETS_ON])
    x = d[te].with_columns(picker_under_chance=pl.Series(p_under), edge=pl.Series(edge), keep=pl.Series(keep)).filter(pl.col("keep"))
    x = x.with_columns(price=pl.Series(SC.american(x["dec_u"].to_numpy())), won=(1 - pl.col("over_won")),
                       profit=pl.when(pl.col("over_won") == 0).then(pl.col("dec_u") - 1).otherwise(-1.0))
    return x.sort("picker_under_chance", descending=True)


def tier(x: pl.DataFrame, caps: bool) -> pl.DataFrame:
    """Rank 1..n; with caps, a pick that would be a player's second pick, or a game's third, is skipped (tier "Skipped (cap)", not ranked)."""
    rows = x.to_dicts()
    if caps:
        kept, skipped, per_player, per_game = [], [], {}, {}
        for r in rows:
            if per_player.get(r["player_id"], 0) < 1 and per_game.get(r["game_id"], 0) < 2:
                kept.append(r)
                per_player[r["player_id"]] = 1
                per_game[r["game_id"]] = per_game.get(r["game_id"], 0) + 1
            else:
                skipped.append(r)
        rows = kept
    else:
        skipped = []
    out = (pl.DataFrame(rows) if rows else x.head(0)).with_row_index("rank", offset=1)
    out = out.with_columns(tier=pl.when(pl.col("rank") <= 10).then(pl.lit("Top 10")).when(pl.col("rank") <= 20).then(pl.lit("Next 10")).otherwise(pl.lit("Rest")))
    if skipped:
        sk = pl.DataFrame(skipped).with_columns(rank=pl.lit(None, dtype=pl.UInt32), tier=pl.lit("Skipped (cap)"))
        out = pl.concat([out, sk.select(out.columns)], how="vertical_relaxed")
    return out


VARIANTS = {"as is": dict(caps=False, hist=False, ens=False), "A: one per player, two per game": dict(caps=True, hist=False, ens=False),
            "B: skip players with < 3 games for team": dict(caps=False, hist=True, ens=False), "C: average 6 picker versions": dict(caps=False, hist=False, ens=True),
            "A + B + C": dict(caps=True, hist=True, ens=True)}


def main(season=2026, weeks=range(1, 6)):
    d = TN.build()
    A = TN.Arr(d)
    X = TN.feats(d, layered=True)
    d = d.with_columns(history=pl.Series(history_counts(d)))
    summ, cards, parls = [], [], []
    for vn, v in VARIANTS.items():
        allc, allp = [], []
        for wk in weeks:
            x = picks_for(d, A, X, season, wk, v["ens"])
            if v["hist"]:
                x = x.filter(pl.col("history") >= MIN_HISTORY)
            x = tier(x, v["caps"]).with_columns(week=pl.lit(wk), variant=pl.lit(vn))
            allc.append(x)
            pp = SC.parlays(x.filter(pl.col("tier") == "Top 10"))
            if pp.height:
                allp.append(pp.with_columns(week=pl.lit(wk), variant=pl.lit(vn)))
        C = pl.concat(allc, how="diagonal_relaxed")
        P = pl.concat(allp, how="diagonal_relaxed") if allp else pl.DataFrame()
        cards.append(C)
        if P.height:
            parls.append(P)
        r = dict(variant=vn, picks_per_week=" / ".join(str(C.filter((pl.col("week") == w) & (pl.col("tier") != "Skipped (cap)")).height) for w in weeks))
        for t in ("Top 10", "Next 10", "Rest"):
            z = C.filter(pl.col("tier") == t)
            w_ = int(z["won"].sum()) if z.height else 0
            r[t] = f"{w_}-{z.height - w_} ({(w_ / z.height if z.height else 0):.0%}) {z['profit'].sum() if z.height else 0:+.1f}u"
        z = C.filter(pl.col("tier") == "Top 10")
        r["top10_units"] = float(z["profit"].sum())
        r["top10_weeks_up"] = int(sum(1 for w in weeks if C.filter((pl.col("week") == w) & (pl.col("tier") == "Top 10"))["profit"].sum() > 0))
        r["parlays"] = f"{int(P['hit'].sum())}/{P.height}" if P.height else "-"
        r["parlay_$"] = float(P["profit"].sum()) if P.height else 0.0
        r["max_same_game_top10"] = int(z.group_by("week", "game_id").len()["len"].max()) if z.height else 0
        summ.append(r)
        print(r, flush=True)
    S = pl.DataFrame(summ)
    S.write_csv(OUTD / "fixes_summary.csv")
    keep = ["variant", "week", "rank", "tier", "player_name", "team", "market", "line", "price", "mean", "picker_under_chance", "edge", "history", "actual", "won", "profit", "game_id"]
    pl.concat([c.select([k for k in keep if k in c.columns]) for c in cards], how="diagonal_relaxed").write_csv(OUTD / "fixes_cards.csv")
    if parls:
        pl.concat(parls, how="diagonal_relaxed").write_csv(OUTD / "fixes_parlays.csv")
    with pl.Config(tbl_width_chars=250, fmt_str_lengths=60, tbl_cols=20):
        print(S)


if __name__ == "__main__":
    main()
