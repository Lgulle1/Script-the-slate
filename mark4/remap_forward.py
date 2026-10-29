"""Carry the saved forward chances (records/lab/forward_p_*_0.npy) onto a rebuilt line table (2026-10-29).

The line study is rebuilt every week: new lines join, and lines that were voided drop out, so row positions move. Each saved chance is matched to the new
table by its line (game, player, market, line); new lines get no chance (NaN) until the forward build covers them; a line that disappeared is reported.

  python -m mark4.remap_forward /path/to/the/previous/line_study_sel.parquet
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

import config
from mark4 import forward as F

KEY = ["game_id", "player_id", "market", "line"]


def keys(parquet: Path | None = None) -> pl.DataFrame:
    """The frame's line keys, built from `parquet` in place of the current line study (restored afterwards)."""
    cur = config.PROCESSED_DIR / "line_study_sel.parquet"
    if parquet is None:
        return F.frame()[0].select(KEY)
    keep = cur.with_suffix(".keep.parquet")
    shutil.copy(cur, keep)
    try:
        shutil.copy(parquet, cur)
        return F.frame()[0].select(KEY)
    finally:
        shutil.move(keep, cur)


def main(old_parquet: str):
    old, new = keys(Path(old_parquet)), keys()
    for name, k in (("old", old), ("new", new)):
        dup = k.height - k.unique().height
        if dup:
            raise SystemExit(f"{name} line keys are not unique ({dup} duplicates): cannot remap safely")
    m = new.with_row_index("j").join(old.with_row_index("i"), on=KEY, how="left")
    i, j = m["i"].to_numpy(), m["j"].to_numpy()
    hit = ~np.isnan(i.astype(float))
    gone = old.join(new, on=KEY, how="anti")
    print(f"old {old.height} lines, new {new.height}: {int(hit.sum())} carried over, {int((~hit).sum())} new (no chance yet), {gone.height} dropped")
    if gone.height:
        print("   dropped:", gone.rows()[:20])
    for f in sorted(F.OUT.glob("forward_p_*_0.npy")):
        p = np.load(f)
        if len(p) == new.height:
            print(f"   {f.name}: already {new.height} rows, left as is")
            continue
        if len(p) != old.height:
            print(f"   {f.name}: {len(p)} rows, matches neither table, left as is")
            continue
        q = np.full(new.height, np.nan)
        q[j[hit]] = p[i[hit].astype(int)]
        np.save(f, q)
        print(f"   {f.name}: remapped {old.height} -> {new.height}")


if __name__ == "__main__":
    main(sys.argv[1])
