"""Week-5 2026: model point predictions vs the FanDuel prop lines imported from the BettingPros export (docs/overnight_plan.md, "BettingPros FanDuel export, import and week-5 comparison").

Descriptive only: the model rows are the retrospective ones of retro_check_2026_week5.parquet (made after the fact, never fit on a 2026 outcome, never shown a line), one week, so nothing here
clears or fails a gate. Players who did not play are removed from both sides; pushes are removed from the hit rates.

  python scripts/prop_import_comparison.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

LOG = config.ROOT / "records" / "lines" / "prop_lines_log.csv"
RETRO = config.ROOT / "retro_check_2026_week5.parquet"
RECON = config.ROOT / "data" / "processed" / "reconstruct_2026_graded.parquet"
OUT = config.ROOT / "data" / "processed" / "prop_import_week5_comparison.parquet"
OUT_RECON = config.ROOT / "data" / "processed" / "prop_import_weeks1_5_comparison.parquet"
OUT_RECON_WEEKS = config.ROOT / "data" / "processed" / "prop_import_weeks1_5_by_week.parquet"
MARKETS = ["pass_yds", "qb_rush_yds", "rush_yds", "rec", "rec_yds"]
KEY = ["game_id", "player_id", "market"]
DRAWS, SEED = 2000, 7


def lines_table() -> pl.DataFrame:
    """One row per (game, player, market): the OPENING line and the LAST line (over side), from the imported FanDuel rows."""
    d = pl.read_csv(LOG, infer_schema_length=0).filter((pl.col("source") == "bettingpros_import") & (pl.col("side") == "over") & pl.col("market").is_in(MARKETS))
    d = d.with_row_index("_i").with_columns(line=pl.col("line").cast(pl.Float64), is_open=pl.col("note").str.starts_with("open")).sort("captured_at", "_i")      # file order breaks ties
    last = d.group_by(KEY, maintain_order=True).last().select(*KEY, last_line=pl.col("line"), last_at=pl.col("captured_at"))
    opn = d.filter(pl.col("is_open")).group_by(KEY, maintain_order=True).first().select(*KEY, open_line=pl.col("line"))
    return last.join(opn, on=KEY, how="left")


def joined(lines: pl.DataFrame, retro: pl.DataFrame) -> pl.DataFrame:
    m = retro.filter(pl.col("market").is_in(MARKETS) & ~pl.col("did_not_play")).select(*KEY, model=pl.col("mean"), actual=pl.col("actual"))
    return lines.join(m, on=KEY, how="inner").filter(pl.col("actual").is_not_null())


def boot_diff(x: pl.DataFrame, line_col: str) -> tuple:
    """(MAE line - MAE model, 95% game-cluster bootstrap interval): positive = the model is closer to the real number."""
    d = x.with_columns(gain=(pl.col(line_col) - pl.col("actual")).abs() - (pl.col("model") - pl.col("actual")).abs())
    per = d.group_by("game_id").agg(s=pl.col("gain").sum(), n=pl.len()).sort("game_id")
    s, n = per["s"].to_numpy(), per["n"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(s), size=(DRAWS, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def hit_rates(x: pl.DataFrame, line_col: str) -> dict:
    """How often the real number went the way the model leaned (above the line -> over, below -> under); all rows and the third with the largest |model - line| / line.
    Reported next to it: the share of rows the model leaned over and how often the real number went over at all (the hit rate of "always over"), plus a game-cluster 95% interval."""
    d = x.filter(pl.col("actual") != pl.col(line_col)).filter(pl.col("model") != pl.col(line_col))
    d = d.with_columns(hit=((pl.col("model") > pl.col(line_col)) == (pl.col("actual") > pl.col(line_col))), gap=(pl.col("model") - pl.col(line_col)).abs() / pl.col(line_col))
    top = d.filter(pl.col("gap") >= d["gap"].quantile(2 / 3)) if d.height >= 3 else d.clear()
    per = d.group_by("game_id").agg(h=pl.col("hit").sum(), n=pl.len()).sort("game_id")
    h, n = per["h"].to_numpy(), per["n"].to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(h), size=(DRAWS, len(h)))
    boots = h[idx].sum(1) / n[idx].sum(1)
    return dict(n_lean=d.height, hit_all=float(d["hit"].mean()), hit_lo=float(np.quantile(boots, 0.025)), hit_hi=float(np.quantile(boots, 0.975)),
                lean_over_share=float((d["model"] > d[line_col]).mean()), always_over=float((d["actual"] > d[line_col]).mean()),
                n_top=top.height, hit_top=float(top["hit"].mean()) if top.height else None)


def summarize(x: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for market in MARKETS + ["ALL"]:
        s = x if market == "ALL" else x.filter(pl.col("market") == market)
        if s.is_empty():
            continue
        for which, col in (("last", "last_line"), ("open", "open_line")):
            t = s.filter(pl.col(col).is_not_null())
            if t.is_empty():
                continue
            mae = dict(mae_model=None, mae_line=None, mae_line_minus_model=None, ci_lo=None, ci_hi=None)     # markets have different units: no pooled MAE
            if market != "ALL":
                gain, lo, hi = boot_diff(t, col)
                mae = dict(mae_model=float((t["model"] - t["actual"]).abs().mean()), mae_line=float((t[col] - t["actual"]).abs().mean()), mae_line_minus_model=gain, ci_lo=lo, ci_hi=hi)
            rows.append(dict(market=market, line=which, n=t.height, **mae, **hit_rates(t, col)))
    return pl.DataFrame(rows)


def main(argv=None):
    """Default: the single-week retro run (week 5). `--reconstruction`: the weeks 1-5 reconstruction rows, pooled by market, and the all-market hit rate by week."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconstruction", action="store_true")
    a = ap.parse_args(argv)
    lines, retro = lines_table(), pl.read_parquet(RECON if a.reconstruction else RETRO)
    x = joined(lines, retro)
    print(f"line rows {lines.height}; with a model prediction, played and an actual: {x.height}")
    out = summarize(x)
    OUT_FILE = OUT_RECON if a.reconstruction else OUT
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_FILE)
    with pl.Config(tbl_rows=40, tbl_cols=20, float_precision=3):
        print(out)
    if a.reconstruction:
        wk = x.join(retro.select(*KEY, "week").unique(), on=KEY, how="left")
        byw = pl.concat([summarize(wk.filter(pl.col("week") == w)).filter(pl.col("market") == "ALL").with_columns(week=pl.lit(w)) for w in sorted(wk["week"].unique().to_list())])
        byw.write_parquet(OUT_RECON_WEEKS)
        with pl.Config(tbl_rows=40, tbl_cols=20, float_precision=3):
            print(byw.select("week", "line", "n", "hit_all", "hit_lo", "hit_hi", "lean_over_share", "always_over"))
    return out, x


if __name__ == "__main__":
    main()
