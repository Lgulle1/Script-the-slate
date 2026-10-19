"""Mark 4, step 3: does FanDuel's line lag behind a role change? (the Braelon Allen question, 2026-10-19)

A ROLE EVENT starts when a player's situation changes before kickoff:
  promoted back     a back holding 25%+ of carries is out and this back's own share was under 35%
  promoted receiver 15%+ of targets are out and this receiver's own share was under 18%
  QB change         his team starts a different QB than last game (receiving and passing props)
For the 1st, 2nd, 3rd and 4th+ game of the same event (consecutive games while it lasts): how the line compares with what happened (actual / line),
how often the over hit, and the return of betting the over and the under at the real price. If the book lags, game 1 shows the biggest miss and it
fades as the line catches up.

  python -m mark4.book_lag        # writes records/lab/book_lag.csv
"""
from __future__ import annotations

import polars as pl

import config
from mark4 import blindspots as BS

OUT = config.ROOT / "records" / "lab" / "book_lag.csv"


def main():
    d = BS.frame()
    ov = d.filter(pl.col("side") == "over").select("season", "game_id", "player_id", "market", "gameday", "family", "line", "actual", "won", "profit", "w",
                                                   "vac_car", "vac_tgt", "own_car", "own_tgt", "qb_change", "team", "prev4")
    un = d.filter(pl.col("side") == "under").select("game_id", "player_id", "market", under_won="won", under_profit="profit")
    x = ov.join(un, on=["game_id", "player_id", "market"], how="inner")
    rush = pl.col("market").is_in(["rush_att", "rush_yds"])
    recv = pl.col("market").is_in(["rec", "rec_yds"])
    events = {
        "promoted back (rushing props)": rush & (pl.col("family") == "RB") & (pl.col("vac_car") >= 0.25) & (pl.col("own_car") < 0.35),
        "promoted receiver (receiving props)": recv & (pl.col("vac_tgt") >= 0.15) & (pl.col("own_tgt") < 0.18),
        "QB change (receiving props)": recv & pl.col("qb_change"),
        "QB change (passing props)": pl.col("market").is_in(["pass_att", "pass_cmp", "pass_yds"]) & pl.col("qb_change"),
    }
    rows = []
    for name, cond in events.items():
        y = x.with_columns(ev=cond.fill_null(False)).sort("player_id", "market", "gameday")
        # game number inside a run of consecutive event games for the same player and market
        y = y.with_columns(brk=(~pl.col("ev")).cum_sum().over("player_id", "market"))
        y = y.with_columns(nth=pl.when(pl.col("ev")).then(pl.col("ev").cast(pl.Int32).cum_sum().over("player_id", "market", "brk")).otherwise(0))
        y = y.filter(pl.col("ev"))
        y = y.with_columns(g=pl.when(pl.col("nth") >= 4).then(pl.lit("4th+")).otherwise(pl.col("nth").cast(pl.Utf8)))
        for (g,), z in y.group_by("g"):
            rows.append(dict(event=name, game=g, lines=z.height, lines_2025_26=z.filter(pl.col("season") >= 2025).height,
                             actual_vs_line=float((z["actual"] / z["line"]).median()), line_vs_last4=float((z["line"] / z["prev4"]).median()),
                             over_hit=float(z["won"].mean()), over_return=float(z["profit"].mean()), under_return=float(z["under_profit"].mean()),
                             over_return_recency=float((z["w"] * z["profit"]).sum() / z["w"].sum()), under_return_recency=float((z["w"] * z["under_profit"]).sum() / z["w"].sum())))
    t = pl.DataFrame(rows).sort("event", "game")
    t.write_csv(OUT)
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(220)
    print(t.with_columns([pl.col(c).round(3) for c in t.columns if t[c].dtype == pl.Float64]))


if __name__ == "__main__":
    main()
