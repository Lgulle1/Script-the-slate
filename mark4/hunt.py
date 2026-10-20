"""The edge hunt (2026-10-20): every cheap, high-power question about where FanDuel's player-prop prices are wrong, on every line 2021-2026.

For each grouping of lines: how often the over hit vs FanDuel's own no-vig chance (positive = overs underpriced), the return of betting every over /
every under at the real price, and how many seasons point the same way. A group only counts if it is consistent across seasons (the game changes;
one hot season is luck). Also, per group, whether our model (Mark 4l) knows more than the price there (log-loss gain per 1,000 lines).

  python -m mark4.hunt        # prints every table, writes records/lab/hunt.csv
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from mark4 import forward as F

LOG = config.ROOT / "records" / "lines" / "prop_lines_log.csv"
OUT = F.OUT / "hunt.csv"


def base() -> pl.DataFrame:
    d, A, *_ = F.frame()
    p = np.load(F.OUT / "forward_p_Mark_4l_0.npy")
    x = d.with_row_index("i").select("i", "season", "week", "game_id", "player_id", "market", "line", "gameday", "team", "home_team", "history")
    x = x.with_columns(won=pl.Series(A.over_won.astype(float)), fair=pl.Series(A.fair_o.astype(float)), be_o=pl.Series(A.be_o), be_u=pl.Series(A.be_u),
                       dec_o=pl.Series(A.dec_o), dec_u=pl.Series(A.dec_u), p=pl.Series(p))
    x = x.filter(pl.col("won").is_not_nan() & pl.col("fair").is_not_nan())
    x = x.with_columns(hold=pl.col("be_o") + pl.col("be_u") - 1, r_over=pl.when(pl.col("won") == 1).then(pl.col("dec_o") - 1).otherwise(-1.0),
                       r_under=pl.when(pl.col("won") == 0).then(pl.col("dec_u") - 1).otherwise(-1.0))
    cl = lambda c: pl.col(c).clip(1e-4, 1 - 1e-4)
    ll = lambda c: -(pl.col("won") * cl(c).log() + (1 - pl.col("won")) * (1 - cl(c)).log())
    return x.with_columns(ll_gain=ll("fair") - ll("p"))


def extras(x: pl.DataFrame) -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, weekday, gametime, home_team, away_team FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    x = x.join(sc.select("game_id", "weekday", "gametime", "away_team"), on="game_id", how="left")
    slot = (pl.when(pl.col("weekday") == "Thursday").then(pl.lit("Thursday")).when(pl.col("weekday") == "Monday").then(pl.lit("Monday night"))
            .when((pl.col("weekday") == "Sunday") & (pl.col("gametime") < "15:00")).then(pl.lit("Sunday 1pm"))
            .when((pl.col("weekday") == "Sunday") & (pl.col("gametime") < "19:00")).then(pl.lit("Sunday 4pm"))
            .when(pl.col("weekday") == "Sunday").then(pl.lit("Sunday night")).otherwise(pl.lit("other day")))
    x = x.with_columns(slot=slot, opp=pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("away_team")).otherwise(pl.col("home_team")))
    # line size within market and season (stars have the biggest lines)
    x = x.with_columns(line_pct=pl.col("line").rank("average").over("market", "season") / pl.len().over("market", "season"))
    # player and defense tendencies vs the line: recency-weighted mean of (over - book) on their EARLIER lines in the market
    x = x.sort("gameday")
    x = x.with_columns(res=pl.col("won") - pl.col("fair"))
    for key, name in ((["player_id", "market"], "player_tend"), (["opp", "market"], "def_tend"), (["team", "market"], "team_tend")):
        x = x.with_columns(pl.col("res").shift(1).ewm_mean(half_life=8, ignore_nulls=True).over(key).alias(name),
                           pl.col("res").cum_count().over(key).alias(name + "_n"))
    # opening vs last line and price (FanDuel log, mostly 2024-2026)
    lg = pl.read_csv(LOG, infer_schema_length=0).filter((pl.col("book") == "FanDuel") & (pl.col("side") == "over"))
    lg = lg.with_columns(pl.col("line").cast(pl.Float64), pl.col("price").cast(pl.Float64), kind=pl.when(pl.col("note") == "open").then(pl.lit("open")).otherwise(pl.lit("last")),
                         t=pl.col("captured_at").str.slice(0, 19).str.to_datetime(strict=False))
    op = lg.filter(pl.col("kind") == "open").sort("t").unique(["game_id", "player_id", "market"], keep="first").select("game_id", "player_id", "market", open_line="line", open_price="price", open_t="t")
    la = lg.filter(pl.col("kind") == "last").sort("t").unique(["game_id", "player_id", "market"], keep="last").select("game_id", "player_id", "market", last_t="t")
    x = x.join(op, on=["game_id", "player_id", "market"], how="left").join(la, on=["game_id", "player_id", "market"], how="left")
    x = x.with_columns(line_move=(pl.col("line") - pl.col("open_line")) / pl.col("open_line").abs().clip(0.5, None),
                       open_age_h=(pl.col("last_t") - pl.col("open_t")).dt.total_hours())
    return x


def table(x: pl.DataFrame, by: str, title: str, min_n=150) -> pl.DataFrame:
    g = x.filter(pl.col(by).is_not_null()).group_by(by).agg(
        n=pl.len(), over_hit=pl.col("won").mean(), book_over=pl.col("fair").mean(), bias=(pl.col("won") - pl.col("fair")).mean(),
        over_ret=pl.col("r_over").mean(), under_ret=pl.col("r_under").mean(), model_ll=pl.col("ll_gain").mean() * 1000)
    s = x.filter(pl.col(by).is_not_null()).group_by(by, "season").agg(b=(pl.col("won") - pl.col("fair")).mean(), n=pl.len()).filter(pl.col("n") >= 30)
    s = s.group_by(by).agg(seasons=pl.len(), over_seasons=(pl.col("b") > 0).sum(), by_season=pl.col("b").sort_by("season").round(3))
    g = g.join(s, on=by, how="left").filter(pl.col("n") >= min_n).sort(by)
    print(f"\n### {title}")
    for r in g.iter_rows(named=True):
        print(f"   {str(r[by]):26s} n {r['n']:6d}  over hit {r['over_hit']:.3f} vs book {r['book_over']:.3f} ({r['bias']:+.3f})  every over {r['over_ret']:+.1%}  every under {r['under_ret']:+.1%}"
              f"  overs underpriced in {r['over_seasons']}/{r['seasons']} seasons  | model vs book {r['model_ll']:+.2f}")
    return g.with_columns(group=pl.lit(title), key=pl.col(by).cast(pl.Utf8)).drop(by)


def main():
    x = extras(base())
    out = []
    bk = lambda c, edges, labels: pl.col(c).cut(edges, labels=labels)
    x = x.with_columns(
        price_shape=bk("fair", [0.40, 0.45, 0.50, 0.55, 0.60], ["over < 40%", "40-45%", "45-50%", "50-55%", "55-60%", "over > 60%"]),
        hold_b=bk("hold", [0.04, 0.05, 0.06, 0.08], ["hold < 4%", "4-5%", "5-6%", "6-8%", "8%+"]),
        week_b=bk("week", [3.5, 8.5, 13.5], ["weeks 1-3", "weeks 4-8", "weeks 9-13", "weeks 14-18"]),
        size_b=bk("line_pct", [0.1, 0.3, 0.7, 0.9], ["smallest 10% of lines", "10-30%", "30-70%", "70-90%", "biggest 10% (stars)"]),
        hist_b=bk("history", [7.5, 15.5, 31.5], ["< 8 games history", "8-15", "16-31", "32+"]),
        ptend=pl.when(pl.col("player_tend_n") >= 8).then(bk("player_tend", [-0.10, -0.03, 0.03, 0.10], ["player well under his lines", "under", "neutral", "over", "player well over his lines"])),
        dtend=pl.when(pl.col("def_tend_n") >= 12).then(bk("def_tend", [-0.06, -0.02, 0.02, 0.06], ["defense: opponents well under", "under", "neutral", "over", "defense: opponents well over"])),
        ttend=pl.when(pl.col("team_tend_n") >= 12).then(bk("team_tend", [-0.06, -0.02, 0.02, 0.06], ["team: players well under", "under", "neutral", "over", "team: players well over"])),
        move_b=pl.when(pl.col("open_line").is_not_null()).then(bk("line_move", [-0.10, -0.02, 0.02, 0.10], ["line dropped 10%+", "dropped 2-10%", "no move", "rose 2-10%", "rose 10%+"])),
        age_b=pl.when(pl.col("open_age_h").is_not_null()).then(bk("open_age_h", [24, 72, 120], ["posted < 1 day before", "1-3 days", "3-5 days", "5+ days"])),
    )
    for by, title in (("market", "by market"), ("price_shape", "price shape (book's over chance)"), ("hold_b", "book's hold (vig) on the line"),
                      ("week_b", "time of season"), ("slot", "slot (your boost windows)"), ("size_b", "line size within market (stars = biggest lines)"),
                      ("hist_b", "how much history the player has"), ("ptend", "player's past results vs his lines"), ("dtend", "defense's past results vs opponents' lines"),
                      ("ttend", "team's past results vs its players' lines"), ("move_b", "line move from opening to bet time (2024-26)"), ("age_b", "how long the line was up (2024-26)")):
        out.append(table(x, by, title))
    # over/under by market and size together (a known prop bias: small lines' overs)
    x = x.with_columns(mk_size=pl.col("market") + " | " + pl.col("size_b").cast(pl.Utf8))
    out.append(table(x, "mk_size", "market x line size", min_n=300))
    pl.concat(out, how="diagonal_relaxed").drop("by_season").write_csv(OUT)


if __name__ == "__main__":
    main()
