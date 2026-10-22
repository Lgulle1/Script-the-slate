"""Tables for the Lab page's Betting strategy tab (2026-10-20), from the all-combo tournament and the real-card simulation.

  python -m mark4.lab_strategy      # writes records/lab/strat_cards.csv, strat_h2h.csv, strat_dials.csv, strat_2026.csv
"""
from __future__ import annotations

import polars as pl

from mark4 import combo_report as R
from mark4 import compare as CP
from mark4 import forward as F


def worst_run(x: pl.DataFrame) -> dict:
    """The worst peak-to-trough drop of the running total over the WHOLE history (not per season), how long it lasted and how many bets it took."""
    import numpy as np
    x = x.sort("season", "week")
    u, b, s, w = (x[c].to_numpy() for c in ("units", "bets", "season", "week"))
    run = np.concatenate([[0.0], np.cumsum(u)])
    peak = np.maximum.accumulate(run)
    j = int((peak - run).argmax())
    i = int(np.flatnonzero(run[:j + 1] == peak[j])[-1])
    return dict(worst_run_units=float(peak[j] - run[j]), worst_run_weeks=j - i, worst_run_bets=float(b[i:j].sum()),
                worst_run_from=f"{s[i]} wk {w[i]}" if i < len(s) else "", worst_run_to=f"{s[j - 1]} wk {w[j - 1]}" if j > 0 else "",
                winning_weeks=int((u > 0).sum()), weeks=len(u))


def main():
    # the all-combo head to head vs Mark 3's rule, boosted and plain
    C = []
    for v in (*R.VERSIONS, "Mark 4l"):                     # Mark 4l (the bet from 2026 week 7) has no all-combo grid file, so it is in the head to head only
        for legs in (1, 2, 3, 4, 5):
            C.append((v, dict(rank="ev", sides="both", markets="all", edge=0.0, depth=12 if legs > 1 else 10, legs=legs, mix="any")))
        C.append((v, R.MARK3_RULE))
    out = []
    for boosted in (True, False):
        h = R.head_to_head(C, boosted)
        h.write_csv(F.OUT / f"combo_h2h_{'boosted' if boosted else 'plain'}.csv")
        h = h.with_columns(boost=pl.lit("boosted" if boosted else "no boost"), rank=pl.col("fmt").str.extract(r"rank (\w+)"),
                           legs=pl.col("fmt").str.extract(r"legs (\d)").cast(pl.Int32))
        out.append(h)
    h = pl.concat(out, how="diagonal_relaxed")
    seasons = [c for c in h.columns if c.isdigit()]
    h = h.with_columns(rule=pl.when(pl.col("rank") == "chance").then(pl.lit("Mark 3 rule: top 8 unders by chance, 4% edge, 4 markets"))
                       .otherwise(pl.lit("Ranked by expected value, either side, top 12 legs, all 7 markets")),
                       seasons_up=pl.sum_horizontal([(pl.col(s) > 0).cast(pl.Int32) for s in seasons]))
    h = h.with_columns(key=pl.format("{}|{}|{}|{}", "version", "rank", "legs", "boost"))
    h.select("key", "version", "rule", "legs", "boost", "this", "mark3_rule", "chance_better", "seasons_up", *seasons).write_csv(F.OUT / "strat_h2h.csv")
    # dials (boosted, all-combo), one column per version
    dl = R.dials(R.load("boosted")).pivot(on="version", index=["dial", "value"], values="ret")
    dl.with_columns(key=pl.format("{}|{}", "dial", "value")).write_csv(F.OUT / "strat_dials.csv")
    # real cards
    t = pl.read_csv(F.OUT / "cards.csv")
    wk = pl.read_csv(F.OUT / "cards_weeks.csv")
    D = {c: {(r["season"], r["week"]): (r["units"], r["bets"]) for r in x.iter_rows(named=True)} for (c,), x in wk.group_by("card")}
    base = D["Mark 3 rule (frozen)"]
    pb = {c: (CP.boot(base, v)["p_b_better"] if c != "Mark 3 rule (frozen)" else None) for c, v in D.items()}
    w = t.with_columns(w=0.5 ** (2026 - pl.col("season")))
    s = w.group_by("card", maintain_order=True).agg(
        ret_all=pl.col("units").sum() / pl.col("bets").sum(), ret_recency=(pl.col("w") * pl.col("units")).sum() / (pl.col("w") * pl.col("bets")).sum(),
        units=pl.col("units").sum(), bets=pl.col("bets").sum(), seasons_up=(pl.col("units") > 0).sum(), seasons=pl.len(), worst_drawdown=pl.col("worst_drawdown").max(),
        u2026=pl.col("units").filter(pl.col("season") == 2026).sum())
    s = s.with_columns(chance_better=pl.col("card").replace_strict(pb, default=None, return_dtype=pl.Float64))
    s = s.join(t.pivot(on="season", index="card", values="units").rename(lambda c: f"u{c}" if c.isdigit() else c), on="card")
    s = s.join(pl.DataFrame([dict(card=c, **worst_run(x)) for (c,), x in wk.group_by("card")]), on="card")
    s.sort("ret_all", descending=True).write_csv(F.OUT / "strat_cards.csv")
    wk.filter(pl.col("season") == 2026).with_columns(key=pl.format("{}|{}", "card", "week")).write_csv(F.OUT / "strat_2026.csv")
    print(s.sort("ret_all", descending=True))
    print(h.filter(pl.col("boost") == "boosted").select("version", "rank", "legs", "this", "chance_better", "seasons_up"))


if __name__ == "__main__":
    main()
