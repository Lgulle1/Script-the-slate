"""Replay the weekly card over 2024, 2025 and 2026 (yours, 2026-10-15: "rerun this against all of 2025 and 2024").

Every week the picker is retrained on every line before that week, then the card is made exactly as it will be for week 6: version "A + B + C" (one pick per player,
two per game, players with 3+ games for their team, six picker versions averaged), tiers by the picker's chance, 2-leg boosted parlays from the Top 10.
"As is" (one picker, no caps, no history rule) runs alongside for comparison. 2024 starts at week 4: the line history starts in 2024, so weeks 1-3 have nothing to learn from.

  python scripts/card_replay.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("card_fixes", HERE / "card_fixes.py")
CF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CF)
SC, TN = CF.SC, CF.TN
OUTD = config.ROOT / "records" / "cards"
WEEKS = {2024: range(4, 19), 2025: range(1, 19), 2026: range(1, 6)}
VARIANTS = {"A + B + C": dict(caps=True, hist=True, ens=True), "as is": dict(caps=False, hist=False, ens=False)}


def history_counts(d: pl.DataFrame) -> np.ndarray:
    from live import results as RS
    plog = pl.concat([RS.played(s)[1].select("player_id", "team", "gameday") for s in (2022, 2023, 2024, 2025, 2026)])
    day = lambda c: pl.col(c).cast(pl.String).str.slice(0, 10).str.to_date()
    h = plog.select("player_id", "team", day("gameday").alias("gameday")).unique()
    x = d.select("player_id", "team", day("gameday").alias("gameday")).with_row_index("i")
    j = x.join(h, on=["player_id", "team"], how="left", suffix="_h").filter(pl.col("gameday_h") < pl.col("gameday"))
    return x.join(j.group_by("i").agg(n=pl.len()), on="i", how="left")["n"].fill_null(0).to_numpy()


def main():
    d = TN.build()
    A = TN.Arr(d)
    X = TN.feats(d, layered=True)
    d = d.with_columns(history=pl.Series(history_counts(d)))
    cards, parls = [], []
    for vn, v in VARIANTS.items():
        for season, weeks in WEEKS.items():
            for wk in weeks:
                x = CF.picks_for(d, A, X, season, wk, v["ens"])
                if v["hist"]:
                    x = x.filter(pl.col("history") >= CF.MIN_HISTORY)
                x = CF.tier(x, v["caps"]).with_columns(season=pl.lit(season), week=pl.lit(wk), variant=pl.lit(vn))
                cards.append(x)
                pp = SC.parlays(x.filter(pl.col("tier") == "Top 10"))
                if pp.height:
                    parls.append(pp.with_columns(season=pl.lit(season), week=pl.lit(wk), variant=pl.lit(vn)))
            print(f"{vn} {season} done", flush=True)
    C = pl.concat(cards, how="diagonal_relaxed")
    P = pl.concat(parls, how="diagonal_relaxed")
    rows = []
    for vn in VARIANTS:
        for season in (2024, 2025, 2026, None):
            c = C.filter((pl.col("variant") == vn) & ((pl.col("season") == season) if season else pl.lit(True)))
            p = P.filter((pl.col("variant") == vn) & ((pl.col("season") == season) if season else pl.lit(True)))
            r = dict(variant=vn, season=str(season or "all"), weeks=c.select("season", "week").n_unique())
            for t in ("Top 10", "Next 10", "Rest"):
                z = c.filter(pl.col("tier") == t)
                w = int(z["won"].sum()) if z.height else 0
                r[f"{t} W"], r[f"{t} L"], r[f"{t} units"] = w, z.height - w, float(z["profit"].sum()) if z.height else 0.0
            top = c.filter(pl.col("tier") == "Top 10")
            wk = top.group_by("season", "week").agg(u=pl.col("profit").sum())
            r["top10_weeks_up"], r["top10_weeks"] = int((wk["u"] > 0).sum()), wk.height
            r["parlays_hit"], r["parlays"], r["parlay_profit"], r["parlay_profit_no_boost"] = int(p["hit"].sum()), p.height, float(p["profit"].sum()), float(p["profit_no_boost"].sum())
            rows.append(r)
    S = pl.DataFrame(rows)
    S.write_csv(OUTD / "replay_summary.csv")
    keep = ["variant", "season", "week", "rank", "tier", "player_name", "team", "market", "line", "price", "mean", "picker_under_chance", "edge", "history", "actual", "won", "profit", "game_id"]
    C.select([k for k in keep if k in C.columns]).write_csv(OUTD / "replay_cards.csv")
    P.write_csv(OUTD / "replay_parlays.csv")
    with pl.Config(tbl_width_chars=250, tbl_cols=30, float_precision=1):
        print(S)
    # by market, Top 10 of the chosen version
    t = C.filter((pl.col("variant") == "A + B + C") & (pl.col("tier") == "Top 10"))
    print(t.group_by("season", "market").agg(picks=pl.len(), hit=pl.col("won").mean().round(3), units=pl.col("profit").sum().round(1)).sort("season", "market"))


if __name__ == "__main__":
    main()
