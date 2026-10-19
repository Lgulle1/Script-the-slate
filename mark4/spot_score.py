"""The blind-spot score (Mark 4d, 2026-10-19): for each line, how wrong has FanDuel been, in the past, in the situations this line is in?

For season S, every situation tag (mark4.blindspots.tags plus the book-lag game numbers of mark4.book_lag) is scored on lines from seasons BEFORE S only:
per market group and side, the recency-weighted return shrunk toward that group's base return (K bets of weight), minus the base. A line's scores are
the sum and the strongest of those excess returns over the tags it belongs to, for the under and for the over, plus how many tags it has. So the picker
gets one compact "the book has misjudged spots like this" signal instead of 22 raw columns.

  python -m mark4.spot_score       # writes data/processed/mark4_spot_score.parquet
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config
from mark4 import blindspots as BS
from mark4 import book_lag as BL

K = 200.0
OUT = config.PROCESSED_DIR / "mark4_spot_score.parquet"
GROUP_OF = {m: g for g, ms in BS.GROUPS.items() if g != "all" for m in ms}


def tag_matrix(ov: pl.DataFrame) -> tuple[np.ndarray, list]:
    ev = BL.event_nth()
    ov = ov.join(ev, on=["game_id", "player_id", "market"], how="left").with_columns([pl.col(c).fill_null(0) for c in BL.EVENT_KEYS.values()])
    tags = {k: v for k, v in BS.tags().items() if k != "every line"}
    for key in BL.EVENT_KEYS.values():
        for g, cond in (("1", pl.col(key) == 1), ("2", pl.col(key) == 2), ("3", pl.col(key) == 3), ("4+", pl.col(key) >= 4)):
            tags[f"{key} game {g}"] = cond
    # quantile tags (defense vs line) must use only past values: replace them with fixed cut points on the shift itself
    tags["defense has been beaten vs the line lately (top fifth)"] = pl.col("dfn_line") >= 0.08
    tags["defense has been holding players under the line lately (bottom fifth)"] = pl.col("dfn_line") <= -0.08
    M = ov.select([e.fill_null(False).alias(f"t{i}") for i, e in enumerate(tags.values())]).to_numpy().astype(bool)
    return M, list(tags)


def build() -> pl.DataFrame:
    d = BS.frame()
    k = ["game_id", "player_id", "market"]
    ov = d.filter(pl.col("side") == "over").unique(k).sort(k)
    un = d.filter(pl.col("side") == "under").unique(k).select(*k, under_profit="profit")
    ov = ov.join(un, on=k, how="left")
    M, names = tag_matrix(ov)
    season = ov["season"].to_numpy()
    t_row = season + (ov["week"].to_numpy() - 1) / 18.0
    grp = np.array([GROUP_OF.get(m, "other") for m in ov["market"].to_list()])
    prof = {"over": ov["profit"].to_numpy().astype(float), "under": ov["under_profit"].to_numpy().astype(float)}
    out = {f"spot_{s}_{a}": np.full(ov.height, np.nan) for s in ("under", "over") for a in ("sum", "best")}
    out["spot_tags"] = M.sum(1).astype(float)
    for S in sorted(set(season)):
        if S <= season.min():
            continue
        tr = season < S
        te = season == S
        w = 0.5 ** ((S - t_row) / 1.0)
        for g in np.unique(grp):
            gtr, gte = tr & (grp == g), te & (grp == g)
            if not gte.any() or gtr.sum() < 100:
                continue
            for side in ("under", "over"):
                pr = prof[side]
                ok = gtr & ~np.isnan(pr)
                base = (w[ok] * pr[ok]).sum() / w[ok].sum()
                Mt = M[ok]
                sw = (Mt * w[ok][:, None]).sum(0)
                sp = (Mt * (w[ok] * pr[ok])[:, None]).sum(0)
                excess = (sp + K * base) / (sw + K) - base            # per tag
                Me = M[gte]
                s_sum = (Me * excess[None, :]).sum(1)
                s_best = np.where(Me.any(1), np.where(Me, excess[None, :], -np.inf).max(1), 0.0)
                out[f"spot_{side}_sum"][gte] = s_sum
                out[f"spot_{side}_best"][gte] = s_best
    return ov.select(k).with_columns([pl.Series(c, v) for c, v in out.items()])


if __name__ == "__main__":
    s = build()
    s.write_parquet(OUT)
    print(s.shape)
    print(s.drop("game_id", "player_id", "market").describe())
