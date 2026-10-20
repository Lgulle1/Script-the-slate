"""Reads the all-combo tournament (records/lab/combos_<version>.parquet, mark4.tournament combos) and answers, per Mark:
  1. which way each dial points: legs per parlay, leg pool depth, sides, markets, edge bar, same vs mixed markets (recency-weighted return per bet,
     every other dial averaged over), season by season;
  2. NESTED: before each season, the full format (version fixed, or version chosen too) with the best shrunk recency-weighted past return; its
     return that season. That is what following the tournament would really have made.
Returns are per unit bet: a parlay format scores the average of every parlay its pool could build that week (one unit a week), straights per bet.

  python -m mark4.combo_report          # prints, writes records/lab/combo_dials.csv, combo_nested.csv, combo_best.csv
"""
from __future__ import annotations

import polars as pl

from mark4 import forward as F

VERSIONS = ("Mark 3 det", "Mark 4h clean", "Blend", "Mark 4i")
DIALS = ("rank", "legs", "depth", "sides", "markets", "edge", "mix")
SEASONS = (2023, 2024, 2025, 2026)
HALF_LIFE = 1.0     # seasons; the user's rule: the games that just happened count most
SHRINK = 10.0       # weeks of weight pulling a format's past return toward 0
MIN_WEEKS = 30      # a format must have bet at least this many weeks in the past to be chosen
MIN_COVER = 0.8     # dial-wise choice: only formats that bet at least this share of past weeks


def load(col="boosted") -> pl.DataFrame:
    t = pl.concat([pl.read_parquet(F.OUT / f"combos_{v.replace(' ', '_')}.parquet") for v in VERSIONS])
    return t.with_columns(units=pl.col(col)).with_columns(
        fmt=pl.format("by {} | {} legs | depth {} | {} | {} markets | edge {} | mix {}", "rank", "legs", "depth", "sides", "markets", "edge", "mix"))


def weighted(t: pl.DataFrame, target: int, keys) -> pl.DataFrame:
    w = 0.5 ** ((target - pl.col("season")) / HALF_LIFE)
    return t.with_columns(w=w).group_by(keys).agg(
        u=(pl.col("w") * pl.col("units")).sum(), st=(pl.col("w") * pl.col("staked")).sum(), n=pl.col("staked").sum(),
        roi=pl.col("units").sum() / pl.col("staked").sum(), pos=(pl.col("units") > 0).sum(), seasons=pl.len())


def dials(t: pl.DataFrame) -> pl.DataFrame:
    out = []
    for d in DIALS:
        g = t.group_by("version", d, "season").agg(units=pl.col("units").sum(), staked=pl.col("staked").sum())
        r = weighted(g, 2027, ["version", d]).with_columns(ret=pl.col("u") / pl.col("st"), dial=pl.lit(d), value=pl.col(d).cast(pl.Utf8))
        out.append(r.select("version", "dial", "value", "ret", "roi", "pos", "seasons"))
    return pl.concat(out).sort("version", "dial", "value")


def nested(t: pl.DataFrame, by_version=True) -> pl.DataFrame:
    keys = ["version", "fmt"]
    rows = []
    groups = [(v, t.filter(pl.col("version") == v)) for v in VERSIONS] if by_version else [("any Mark", t)]
    for name, x in groups:
        for S in SEASONS:
            past = x.filter(pl.col("season") < S)
            n_weeks = past.filter(pl.col("legs") == 1).group_by("version", "season").agg(pl.col("weeks").max()).group_by("version").agg(pl.col("weeks").sum())["weeks"].max()
            sc = weighted(past, S, keys).join(past.group_by(keys).agg(c=pl.col("weeks").sum() / n_weeks), on=keys)
            sc = sc.filter((pl.col("n") >= MIN_WEEKS) & (pl.col("c") >= MIN_COVER))
            sc = sc.with_columns(score=pl.col("u") / (pl.col("st") + SHRINK)).sort("score", descending=True)
            v, f = sc["version"][0], sc["fmt"][0]
            cur = x.filter((pl.col("season") == S) & (pl.col("version") == v) & (pl.col("fmt") == f))
            u, st = (float(cur["units"][0]), float(cur["staked"][0])) if cur.height else (0.0, 0.0)
            rows.append(dict(chooser=name, season=S, picked=f"{v}: {f}", units=u, staked=st, ret=u / st if st else None, past=float(sc["score"][0])))
    return pl.DataFrame(rows)


def nested_dials(t: pl.DataFrame, fixed: dict | None = None) -> pl.DataFrame:
    """Steadier nested choice: before each season, each dial is set on its own to the value with the best recency-weighted past return (every other
    dial averaged over, so thousands of bets stand behind each choice), then the composed format is bet that season. `fixed` pins dials (e.g. legs)."""
    rows = []
    for v in VERSIONS:
        x = t.filter(pl.col("version") == v)
        for S in SEASONS:
            past = x.filter(pl.col("season") < S)
            # a strategy has to bet nearly every week (four boosts a week): formats that sat out more than (1 - MIN_COVER) of past weeks are out
            n_weeks = past.filter(pl.col("legs") == 1).group_by("season").agg(pl.col("weeks").max())["weeks"].sum()
            cover = past.group_by("fmt").agg(c=pl.col("weeks").sum() / n_weeks).filter(pl.col("c") >= MIN_COVER)
            past = past.join(cover.select("fmt"), on="fmt")
            pick = dict(fixed or {})
            for dl in DIALS:
                if dl in pick:
                    continue
                g = past.group_by(dl, "season").agg(units=pl.col("units").sum(), staked=pl.col("staked").sum())
                r = weighted(g, S, [dl]).with_columns(ret=pl.col("u") / pl.col("st")).sort("ret", descending=True)
                pick[dl] = r[dl][0]
            cur = x.filter(pl.col("season") == S)
            for dl, val in pick.items():
                cur = cur.filter(pl.col(dl) == val)
            u, st = (float(cur["units"].sum()), float(cur["staked"].sum())) if cur.height else (0.0, 0.0)
            rows.append(dict(version=v, season=S, picked=" | ".join(f"{k} {pick[k]}" for k in DIALS), units=u, staked=st, ret=u / st if st else None))
    return pl.DataFrame(rows)


def main():
    pl.Config.set_tbl_rows(200); pl.Config.set_fmt_str_lengths(120); pl.Config.set_tbl_width_chars(260)
    for col in ("boosted", "plain"):
        t = load(col)
        dl = dials(t)
        nv = pl.concat([nested(t), nested(t, by_version=False)])
        print(f"\n######## {col} ########")
        print(dl.pivot(on="version", index=["dial", "value"], values="ret"))
        print(nv)
        print(nv.group_by("chooser").agg(units=pl.col("units").sum(), staked=pl.col("staked").sum()).with_columns(ret=pl.col("units") / pl.col("staked")))
        best = weighted(t, 2027, ["version", "fmt"]).filter(pl.col("n") >= MIN_WEEKS).with_columns(ret=pl.col("u") / pl.col("st")).sort("ret", descending=True)
        print(best.group_by("version", maintain_order=True).head(5).select("version", "fmt", "ret", "roi", "pos", "seasons", "n"))
        if col == "boosted":
            dl.write_csv(F.OUT / "combo_dials.csv"); nv.write_csv(F.OUT / "combo_nested.csv"); best.head(300).write_csv(F.OUT / "combo_best.csv")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ week-level head-to-head (paired week bootstrap)

MARK3_RULE = dict(rank="chance", sides="under", markets="four", edge=0.04, depth=8, legs=2, mix="any")


def week_results(d, A, p, spec, boosted=True):
    """{(season, week): (units, 1)}: the week's average all-combo parlay (or its straights, legs=1) for one format."""
    from mark4 import tournament as T
    W = T.week_pools(d, A, p, spec["sides"], spec["markets"], spec["edge"], spec["rank"])
    out = {}
    for key, rows in W.items():
        one = {key: rows}
        if spec["legs"] == 1:
            r = T.score_straights(one, spec["depth"])
            if r:
                v = next(iter(r.values())); out[key] = (v[0], v[1])
        else:
            r = T.combo_ev(one, spec["legs"], spec["depth"], spec["mix"])
            if r:
                v = next(iter(r.values())); out[key] = (v[1] if boosted else v[0], 1.0)
    return out


def boot(a: dict, b: dict, n=4000, seed=7, target=2026):
    """Chance format b beats format a: paired over weeks, recency-weighted (0.5 a season back), weeks resampled with replacement."""
    from mark4 import compare as CP
    return CP.boot(a, b, n, seed, target)


def head_to_head(candidates, boosted=True):
    import numpy as np
    from mark4 import forward as F, tournament as T
    d, A, *_ = F.frame()
    P = {v: T.load_p(v) for v in VERSIONS}
    base = week_results(d, A, P["Mark 3 det"], MARK3_RULE, boosted)
    rows = []
    for v, spec in candidates:
        r = week_results(d, A, P[v], spec, boosted)
        b = boot(base, r)
        per = {}
        for (s, _), x in r.items():
            per.setdefault(s, [0.0, 0.0]); per[s][0] += x[0]; per[s][1] += x[1]
        rows.append(dict(version=v, fmt=" | ".join(f"{k} {spec[k]}" for k in DIALS), mark3_rule=b["a"], this=b["b"], chance_better=b["p_b_better"],
                         **{str(s): per[s][0] / per[s][1] for s in sorted(per)}))
    return pl.DataFrame(rows)
