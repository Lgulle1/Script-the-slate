"""Comps scored against the BOOK'S LINE (2026-10-19, after comps v2 failed its exam: the residual to our projection is not what a bet is paid on).

Same searches and situation comps as comps2, but every past game is a FanDuel line row (2021-2026, the line study's pre-kickoff lines) and its outcome is
how it finished against that line: rel = actual / line - 1 (clipped), over = actual > line. The question each search answers becomes "in situations like
tonight's, did the player beat the number the book posted?" (e.g. backs promoted by an injury vs their line; receivers coming off a DNP week vs their line).

  python -m comps2.line_comps            # practice ladder (2022-2023) -> records/tuning/comps2_ladder.csv, label "vs line: ..."
"""
from __future__ import annotations

import time

import polars as pl

import config
from comps2 import evaluate as V
from comps2 import run_ladder as RL
from comps2 import searches as S
from comps2 import situations as SI

OUT = config.PROCESSED_DIR / "comps2_line_components.parquet"
PRACTICE = (2022, 2023)


def frame() -> pl.DataFrame:
    sel = pl.read_parquet(config.PROCESSED_DIR / "line_study_sel.parquet").filter((pl.col("side") == "over") & (pl.col("tier") != "late"))
    sel = sel.unique(["game_id", "player_id", "market"], keep="first").select("game_id", "player_id", "market", line="line", lactual="actual")
    e = S.load().drop("rid")
    x = sel.join(e, on=["game_id", "player_id", "market"], how="inner")
    x = x.with_columns(expected=pl.col("line"), actual=pl.col("lactual"), rel=pl.col("lactual") / pl.col("line") - 1).drop("lactual")
    x = x.with_columns(relc=pl.col("rel").clip(S.REL_LO, S.REL_HI))
    return x.sort("gd", "game_id", "player_id", "market").with_row_index("rid")


def components(x: pl.DataFrame, seasons) -> pl.DataFrame:
    tg = x.filter(pl.col("season").is_in(seasons))["rid"].to_numpy()
    flat = S.run(x, S.Params(h_def=None, h_off=None, h_arch=None, h_pair=None), tg, ("own", "dfn"))
    it = S.run(x, S.Params(h_pair=0.75, h_arch=0.75), tg, ("int",)).select("rid", "int_shift", "int_neff", "int_sw")
    sit = SI.search(x, pl.read_parquet(SI.OUT), tg, h=0.5)
    return flat.join(it, on="rid").join(sit, on="rid")


if __name__ == "__main__":
    t0 = time.time()
    x = frame()
    print("line rows:", x.height, x.group_by("season").len().sort("season").rows())
    c = components(x, PRACTICE)
    lad = V.ladder(c, ("own", "dfn", "int", "sit"), PRACTICE).with_columns(label=pl.lit("vs line: searches"))
    pl.concat([pl.read_csv(RL.LOG), lad], how="diagonal_relaxed").write_csv(RL.LOG)
    print(lad.select("group", "comp", "n", "corr", "corr_min_season", "over_lift"), f"{time.time() - t0:.0f}s")
