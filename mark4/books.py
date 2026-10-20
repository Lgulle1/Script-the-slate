"""The rest of the market vs FanDuel (2026-10-20): does FanDuel's price lag the other books, and does that make better legs?

Source: the all-books BettingPros export for 2026 weeks 1-6 (22 books and exchanges: DraftKings, Caesars, BetMGM, bet365, Fanatics, Novig, Kalshi,
ProphetX, Polymarket, ... and the BettingPros consensus). For every FanDuel line we price, each other book quoting the SAME line gives a no-vig
over chance (its last quote stamped no later than GRACE seconds after kickoff, the same rule as FanDuel's). The market's chance is the average over
those books (on the logit scale). Graded on the line study (records via mark4.forward.frame).

Questions:
  1. Who knows more: FanDuel's no-vig price or the rest of the market's? (log loss on the same lines)
  2. Betting FanDuel's side where the market says FanDuel is wrong: return.
  3. The market as a third voice next to Mark 3 / Mark 4h: does it beat Mark 4i?

  python -m mark4.books
"""
from __future__ import annotations

import glob

import duckdb
import numpy as np
import polars as pl

import config
from live import prop_lines as PL
from mark4 import forward as F

FILES = ["/tmp/claude-0/allbooks/NFL_2026_Weeks1-6_All_Books_Player_Props.csv"]
MARKETS = {"receiving-yards": "rec_yds", "receptions": "rec", "rushing-yards": "rush_yds", "passing-yards": "pass_yds",
           "rushing-attempts": "rush_att", "passing-attempts": "pass_att", "passing-completions": "pass_cmp"}
TEAM_FIX = {"JAC": "JAX", "LAR": "LA"}
GRACE = 300
OUT = F.OUT / "books_2026.parquet"


def dec(a):
    return pl.when(a > 0).then(1 + a / 100).otherwise(1 + 100 / -a)


def quotes() -> pl.DataFrame:
    """Per (book, game, player name, market): line and no-vig over chance from its last pre-kickoff quote."""
    d = pl.concat([pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy") for f in FILES], how="diagonal_relaxed")
    d = d.filter(pl.col("market").is_in(list(MARKETS)) & pl.col("selection").is_in(["Over", "Under"]) & (pl.col("season_type") == "REG"))
    ts = lambda c: pl.col(c).str.to_datetime(time_zone="UTC", strict=False)
    d = d.with_columns(late=(ts("line_updated_utc") - ts("game_start_utc")).dt.total_seconds(), market=pl.col("market").replace_strict(MARKETS),
                       odds=pl.col("american_odds").cast(pl.Float64, strict=False), line=pl.col("line").cast(pl.Float64, strict=False),
                       season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32),
                       home=pl.col("home_team").replace(TEAM_FIX), away=pl.col("away_team").replace(TEAM_FIX), upd=ts("line_updated_utc"))
    d = d.filter(pl.col("late").is_not_null() & (pl.col("late") <= GRACE) & pl.col("odds").is_not_null() & pl.col("line").is_not_null())
    d = d.sort("upd").unique(["sportsbook", "season", "week", "home", "away", "player", "market", "selection"], keep="last")
    k = ["sportsbook", "season", "week", "home", "away", "player", "market"]
    o = d.filter(pl.col("selection") == "Over").select(*k, "line", po="odds")
    u = d.filter(pl.col("selection") == "Under").select(*k, lu="line", pu="odds")
    x = o.join(u, on=k).filter(pl.col("line") == pl.col("lu")).drop("lu")
    x = x.with_columns(io=1 / dec(pl.col("po")), iu=1 / dec(pl.col("pu")))
    return x.with_columns(fair=pl.col("io") / (pl.col("io") + pl.col("iu")), hold=pl.col("io") + pl.col("iu") - 1)


def attach(q: pl.DataFrame) -> pl.DataFrame:
    """game_id and player_id (gsis) for each quote: schedule on season/week/teams, name inside the game's played roster."""
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, season, week, home_team, away_team FROM schedules WHERE game_type = 'REG' "
                        "ORDER BY game_id, pulled_at DESC)").pl().with_columns(pl.col("season", "week").cast(pl.Int32))
    con.close()
    q = q.join(sched, left_on=["season", "week", "home", "away"], right_on=["season", "week", "home_team", "away_team"], how="inner")
    import importlib.util
    spec = importlib.util.spec_from_file_location("td_import", config.ROOT / "scripts" / "td_import.py")
    ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(ti)
    ro = ti.PL_roster(2026).with_columns(nm=pl.col("display_name").map_elements(PL.normalize_name, return_dtype=pl.Utf8))
    ro = ro.group_by("game_id", "nm").agg(pl.col("gsis_id").first(), n=pl.len()).filter(pl.col("n") == 1)
    q = q.with_columns(nm=pl.col("player").map_elements(PL.normalize_name, return_dtype=pl.Utf8))
    return q.join(ro.select("game_id", "nm", player_id="gsis_id"), on=["game_id", "nm"], how="inner")


def main():
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
    q = attach(quotes())
    q.write_parquet(OUT)
    d, A, *_ = F.frame()
    p3 = np.load(F.OUT / "forward_p_Mark_3_det_0.npy"); p4 = np.load(F.OUT / "forward_p_Mark_4h_clean_0.npy"); p4i = np.load(F.OUT / "forward_p_Mark_4i_0.npy")
    base = d.with_row_index("i").filter(pl.col("season") == 2026).select("i", "game_id", "player_id", "market", "line")
    others = q.filter(~pl.col("sportsbook").is_in(["FanDuel"]))
    m = base.join(others.select("game_id", "player_id", "market", "line", "sportsbook", "fair", "hold"), on=["game_id", "player_id", "market", "line"], how="inner")
    lg = lambda c: (pl.col(c).clip(1e-4, 1 - 1e-4) / (1 - pl.col(c).clip(1e-4, 1 - 1e-4))).log()
    per_book = m.with_columns(l=lg("fair"))
    agg = per_book.group_by("i").agg(mkt_logit=pl.col("l").mean(), n_books=pl.len(),
                                     cons=pl.col("l").filter(pl.col("sportsbook") == "BettingPros Consensus").first(),
                                     sharp=pl.col("l").filter(pl.col("sportsbook").is_in(["Novig", "ProphetX", "Kalshi", "Polymarket", "Polymarket US"])).mean())
    i = agg["i"].to_numpy()
    y = A.over_won.astype(float)[i]
    fd = A.fair_o.astype(float)[i]
    ok = ~np.isnan(y) & ~np.isnan(fd)
    sig = lambda z: 1 / (1 + np.exp(-z))
    mk = sig(agg["mkt_logit"].to_numpy())
    sharp = sig(agg["sharp"].fill_null(np.nan).to_numpy())
    cl = lambda v: np.clip(v, 1e-4, 1 - 1e-4)
    ll = lambda v: -(y * np.log(cl(v)) + (1 - y) * np.log(1 - cl(v)))
    print(f"FanDuel 2026 lines matched to other books at the same line: {ok.sum()} (of {base.height}); books per line median {np.median(agg['n_books']):.0f}")
    print(per_book.group_by("sportsbook").len().sort("len", descending=True).rows())
    res = {}
    for name, v in (("FanDuel's own no-vig", fd), ("other books (average)", mk), ("exchanges / prediction markets", sharp),
                    ("Mark 3", p3[i]), ("Mark 4h", p4[i]), ("Mark 4i", p4i[i]),
                    ("Mark 4i + other books (50/50)", 0.5 * p4i[i] + 0.5 * mk), ("Mark 3/4h blend + other books (30/70)", 0.3 * (0.5 * p3[i] + 0.5 * p4[i]) + 0.7 * mk)):
        g = ok & ~np.isnan(v)
        res[name] = 1000 * float((ll(fd) - ll(v))[g].mean())
        print(f"   {name:40s} log-loss gain vs FanDuel per 1,000 lines: {res[name]:+.2f}  (lines {g.sum()})")
    # betting FanDuel where the rest of the market disagrees
    be_o, be_u = A.be_o[i], A.be_u[i]
    dec_o, dec_u = A.dec_o[i], A.dec_u[i]
    for name, v in (("other books", mk), ("Mark 4i", p4i[i]), ("Mark 4i + other books", 0.5 * p4i[i] + 0.5 * mk)):
        eo, eu = v - be_o, (1 - v) - be_u
        over = eo > eu
        ed = np.where(over, eo, eu)
        won = np.where(over, y, 1 - y)
        r = np.where(won == 1, np.where(over, dec_o, dec_u) - 1, -1.0)
        for e in (0.0, 0.02, 0.04):
            s = ok & (ed >= e) & ~np.isnan(v)
            print(f"   bet FanDuel where {name:22s} edge >= {e:.0%}: {s.sum():4d} bets, hit {won[s].mean():.1%}, return {r[s].mean():+.1%}")
    return res


if __name__ == "__main__":
    main()
