"""The live weekly card (yours, 2026-10-15): the picker retrained on every graded line before the week, version "A + B + C" (six picker versions averaged,
players with 3+ games for their team, one pick per player, two per game), ranked by the picker's chance; Top 10 / Next 10; 2-leg boosted parlays from the Top 10.
FanDuel is the guide; markets FanDuel does not post (rush attempts, pass attempts, pass completions) use Caesars, the regular sportsbook that prices closest to
FanDuel (same line 72% of the time on shared markets, 2026 weeks 1-6). Each pick also shows the best price for the same under at any regular sportsbook.

  python scripts/week_card.py --season 2026 --week 6 --books /path/to/All_Books_Player_Props.csv
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CR = _load("card_replay")
CF, SC, TN = CR.CF, CR.SC, CR.TN
OUTD = config.ROOT / "records" / "cards"
REGULAR = ["FanDuel", "DraftKings", "Caesars", "Fanatics", "BetMGM", "theScore Bet", "Hard Rock", "BetRivers", "bet365"]
BOOK_MARKET = {"pass_att": "passing-attempts", "pass_cmp": "passing-completions", "pass_yds": "passing-yards", "rush_att": "rushing-attempts", "rush_yds": "rushing-yards",
               "rec": "receptions", "rec_yds": "receiving-yards"}
COMP = {"pass_att": "Caesars", "pass_cmp": "Caesars", "rush_att": "Caesars"}
LABEL = {"rec": "Receptions", "rec_yds": "Receiving yds", "rush_yds": "Rush yds", "rush_att": "Rush attempts", "pass_yds": "Pass yds", "pass_att": "Pass attempts", "pass_cmp": "Pass completions"}


def best_prices(books_csv: Path, season: int, week: int) -> pl.DataFrame:
    d = pl.read_csv(books_csv, infer_schema_length=0, encoding="utf8-lossy")
    d = d.rename({c: c.lstrip("﻿") for c in d.columns})
    d = d.filter((pl.col("week") == str(week)) & (pl.col("selection") == "Under") & pl.col("sportsbook").is_in(REGULAR) & pl.col("market").is_in(list(BOOK_MARKET.values())))
    d = d.with_columns(line=pl.col("line").cast(pl.Float64), odds=pl.col("american_odds").cast(pl.Float64, strict=False), key=pl.col("player").str.to_lowercase())
    d = d.with_columns(dec=pl.when(pl.col("odds") > 0).then(1 + pl.col("odds") / 100).otherwise(1 + 100 / -pl.col("odds")))
    best = d.sort("dec", descending=True).group_by("key", "market", "line").agg(best_book=pl.col("sportsbook").first(), best_price=pl.col("odds").first(), books_quoting=pl.len())
    return best


def ungraded_parlays(top: pl.DataFrame) -> pl.DataFrame:
    rows = top.sort("rank").to_dicts()
    used, out = set(), []
    for i, a in enumerate(rows):
        if a["rank"] in used:
            continue
        for b in rows[i + 1:]:
            if b["rank"] not in used and b["game_id"] != a["game_id"]:
                used |= {a["rank"], b["rank"]}
                dec = a["dec_u"] * b["dec_u"]
                out.append(dict(leg1=f"#{a['rank']} {a['player_name']} {LABEL[a['market']]} under {a['line']:g}", leg2=f"#{b['rank']} {b['player_name']} {LABEL[b['market']]} under {b['line']:g}",
                                price=int(SC.american(np.array([dec]))[0]), picker_chance=a["picker_under_chance"] * b["picker_under_chance"],
                                payout_on_10_boosted=round(10 * (dec - 1) * 1.3, 2)))
                break
    return pl.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=6)
    ap.add_argument("--books", type=Path, required=True)
    ap.add_argument("--include-comp", action="store_true", help="also card the markets FanDuel does not post (priced at Caesars); off by default: you bet FanDuel only")
    a = ap.parse_args(argv)
    made_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    d = TN.build()
    A = TN.Arr(d)
    X = TN.feats(d, layered=True)
    d = d.with_columns(history=pl.Series(CR.history_counts(d)))
    x = CF.picks_for(d, A, X, a.season, a.week, True)
    x = x.filter(pl.col("history") >= CF.MIN_HISTORY)
    if not a.include_comp:                                                  # you bet FanDuel only: Caesars-comp markets stay in training, not on the card
        x = x.filter(~pl.col("market").is_in(list(COMP)))
    x = CF.tier(x, True).with_columns(price=pl.Series(SC.american(x["dec_u"].to_numpy())) if False else pl.col("price"))
    x = x.with_columns(book=pl.col("market").replace_strict(COMP, default="FanDuel"), key=pl.col("player_name").str.to_lowercase(),
                       book_market=pl.col("market").replace_strict(BOOK_MARKET))
    bp = best_prices(a.books, a.season, a.week)
    x = x.join(bp, left_on=["key", "book_market", "line"], right_on=["key", "market", "line"], how="left")
    card = x.select("rank", "tier", "player_name", "team", pl.col("market").replace_strict(LABEL).alias("market"), "line", "book", "price",
                    pl.col("mean").round(1).alias("our_prediction"), pl.col("picker_under_chance").round(3), pl.col("edge").round(3), "best_book", "best_price", "books_quoting",
                    "gameday", "game_id").sort("rank", nulls_last=True)
    P = ungraded_parlays(x.filter(pl.col("tier") == "Top 10"))
    OUTD.mkdir(parents=True, exist_ok=True)
    stem = f"{a.season}_wk{a.week:02d}_card"
    card.write_csv(OUTD / f"{stem}.csv")
    P.write_csv(OUTD / f"{stem}_parlays.csv")
    (OUTD / f"{stem}_meta.json").write_text(json.dumps(dict(made_at_utc=made_at, season=a.season, week=a.week, version="A + B + C", picks=card.height,
                                                            trained_on="every graded line before the week", comp_book_for_attempts="Caesars"), indent=1))
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250, fmt_str_lengths=40):
        print(card.filter(pl.col("tier").is_in(["Top 10", "Next 10"])))
        print(P)
        print("picks:", card.height, card.group_by("tier").len().rows())


if __name__ == "__main__":
    main()
