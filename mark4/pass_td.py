"""Passing touchdowns, a new market for Mark 4 (2026-10-20): FanDuel's over/under on a QB's passing TDs (mostly 1.5, some 0.5), 2021-2026.

Lines: the BettingPros exports in records/lines/imports, market `passing-touchdowns`, FanDuel, regular season, main line, both sides priced. The price is the
last one posted up to 300 s after kickoff, else the opening one (the feed overwrites a line once the game starts), else the line is dropped. The QB is
matched by name inside the game's played roster. Graded on player_stats passing_tds.

Model (forward, like every Mark 4 piece): a Poisson count model of a QB's passing TDs, trained on every QB game (10+ attempts) since 2016 that was played
before the 6-week block being predicted, from pre-game inputs only:
  the QB's recency-weighted passing TDs, attempts, yards per attempt, TD rate per attempt (half-life 8 games, across seasons) and games behind them;
  the team's implied points (total and spread), the opponent's recency-weighted passing TDs allowed, dome, wind.
P(over line) = 1 - PoissonCDF(floor(line), lambda). Then, like Mark 4i, mixed with FanDuel's no-vig chance, the weight chosen per season from earlier
seasons only.

  python -m mark4.pass_td      # writes records/lab/pass_td_lines.parquet (per line: p_over, prices, result) and prints the forward test
"""
from __future__ import annotations

import glob
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from live import prop_lines as PL

IMPORTS = config.ROOT / "records" / "lines" / "imports"
OUT = config.ROOT / "records" / "lab" / "pass_td_lines.parquet"
TEAM_FIX = {"JAC": "JAX", "LAR": "LA"}
GRACE = 300
HL = 8.0
BLOCKS = (1, 7, 13)
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
MIX_B = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
FEATS = ["qb_td", "qb_att", "qb_ypa", "qb_tdr", "qb_n", "team_pts", "opp_tda", "dome", "wind"]


def _dec(a):
    return pl.when(a > 0).then(1 + a / 100).otherwise(1 + 100 / -a)


def lines() -> pl.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(IMPORTS / "*FanDuel_NFL_20*_Player_Props.csv"))):
        d = pl.read_csv(f, infer_schema_length=0, encoding="utf8-lossy")
        d = d.rename({c: c.lstrip("﻿") for c in d.columns})
        for pre in ("book_opening_", "selected_opening_"):
            if "fanduel_opening_odds" not in d.columns and f"{pre}odds" in d.columns:
                d = d.rename({c: c.replace(pre, "fanduel_opening_") for c in d.columns if c.startswith(pre)})
        d = d.filter(pl.col("market") == "passing-touchdowns")
        if "season_type" in d.columns:
            d = d.filter(pl.col("season_type") == "REG")
        if "sportsbook" in d.columns:
            d = d.filter(pl.col("sportsbook") == "FanDuel")
        if "is_main_line" in d.columns:
            d = d.filter(pl.col("is_main_line") != "False")
        for c in ("fanduel_opening_odds", "fanduel_opening_line"):
            if c not in d.columns:
                d = d.with_columns(pl.lit(None, pl.Utf8).alias(c))
        parts.append(d.select("season", "week", "away_team", "home_team", "player", "selection", "line", "american_odds", "game_start_utc", "line_updated_utc",
                              "fanduel_opening_odds", "fanduel_opening_line"))
    d = pl.concat(parts, how="diagonal_relaxed")
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, season, week, home_team, away_team FROM schedules WHERE game_type = 'REG' "
                        "ORDER BY game_id, pulled_at DESC)").pl().with_columns(pl.col("season", "week").cast(pl.Int32))
    con.close()
    fix = lambda c: pl.col(c).replace(TEAM_FIX)
    d = d.with_columns(away=fix("away_team"), home=fix("home_team"), season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32))
    d = d.join(sched, left_on=["season", "week", "home", "away"], right_on=["season", "week", "home_team", "away_team"], how="inner")
    f = lambda c: pl.col(c).cast(pl.Float64, strict=False)
    ts = lambda c: pl.col(c).str.to_datetime(time_zone="UTC", strict=False)
    d = d.with_columns(late_s=(ts("line_updated_utc") - ts("game_start_utc")).dt.total_seconds())
    ok = pl.col("late_s").is_null() | (pl.col("late_s") <= GRACE)
    d = d.with_columns(price=pl.when(ok).then(f("american_odds")).otherwise(f("fanduel_opening_odds")),
                       ln=pl.when(ok).then(f("line")).otherwise(f("fanduel_opening_line")), upd=ts("line_updated_utc"))
    d = d.filter(pl.col("price").is_not_null() & pl.col("ln").is_not_null())
    d = d.sort("upd", descending=True).unique(["game_id", "player", "selection"], keep="first")
    o = d.filter(pl.col("selection") == "Over").select("game_id", "season", "week", "player", line="ln", po="price")
    u = d.filter(pl.col("selection") == "Under").select("game_id", "player", lu="ln", pu="price")
    x = o.join(u, on=["game_id", "player"]).filter(pl.col("line") == pl.col("lu")).drop("lu")
    # match the QB in the game's played roster
    import importlib.util
    spec = importlib.util.spec_from_file_location("td_import", config.ROOT / "scripts" / "td_import.py")
    ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(ti)
    roster = ti.PL_roster
    rows = []
    for season in sorted(x["season"].unique().to_list()):
        rm = {}
        for r in roster(season).filter(pl.col("position") == "QB").iter_rows(named=True):
            rm.setdefault((r["game_id"], PL.normalize_name(r["display_name"])), []).append(r)
        for r in x.filter(pl.col("season") == season).iter_rows(named=True):
            hit = rm.get((r["game_id"], PL.normalize_name(r["player"])), [])
            if len(hit) == 1:
                rows.append({**r, "player_id": hit[0]["gsis_id"], "team": hit[0]["team"]})
    m = pl.DataFrame(rows)
    m = m.with_columns(dec_o=_dec(pl.col("po")), dec_u=_dec(pl.col("pu")))
    m = m.with_columns(be_o=1 / pl.col("dec_o"), be_u=1 / pl.col("dec_u"))
    return m.with_columns(fair_o=pl.col("be_o") / (pl.col("be_o") + pl.col("be_u")))


def qb_games() -> pl.DataFrame:
    """Every QB game since 2016 (10+ attempts) with pre-game inputs."""
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    ps = con.execute("""SELECT * FROM (SELECT DISTINCT ON (player_id, season, week) player_id, season, week, team, opponent_team, attempts, passing_yards,
                        passing_tds FROM player_stats WHERE season_type = 'REG' AND position = 'QB' ORDER BY player_id, season, week, pulled_at DESC)""").pl()
    sc = con.execute("""SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, season, week, gameday, home_team, away_team, spread_line, total_line, roof, wind
                        FROM schedules WHERE game_type = 'REG' ORDER BY game_id, pulled_at DESC)""").pl()
    con.close()
    ps = ps.with_columns(pl.col("season", "week").cast(pl.Int32)).filter(pl.col("attempts") >= 10)
    sc = sc.with_columns(pl.col("season", "week").cast(pl.Int32), gameday=pl.col("gameday").cast(pl.Utf8).str.slice(0, 10).str.to_date(strict=False))
    long = pl.concat([sc.select("game_id", "season", "week", "gameday", team="home_team", opp="away_team", pts=(pl.col("total_line") + pl.col("spread_line")) / 2,
                                dome=pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64), wind="wind"),
                      sc.select("game_id", "season", "week", "gameday", team="away_team", opp="home_team", pts=(pl.col("total_line") - pl.col("spread_line")) / 2,
                                dome=pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64), wind="wind")])
    g = ps.join(long, on=["season", "week", "team"], how="inner").sort("gameday")
    a = 1 - 0.5 ** (1 / HL)
    ew = lambda c: pl.col(c).shift(1).ewm_mean(alpha=a, ignore_nulls=True).over("player_id")
    g = g.with_columns(qb_td=ew("passing_tds"), qb_att=ew("attempts"), qb_yds=ew("passing_yards"), qb_n=pl.col("attempts").cum_count().over("player_id") - 1)
    g = g.with_columns(qb_ypa=pl.col("qb_yds") / pl.col("qb_att"), qb_tdr=pl.col("qb_td") / pl.col("qb_att"))
    # opponent's passing TDs allowed per game, recency-weighted over its earlier games
    allowed = g.group_by("game_id", "opp", "gameday").agg(tda=pl.col("passing_tds").sum()).sort("gameday")
    allowed = allowed.with_columns(opp_tda=pl.col("tda").shift(1).ewm_mean(alpha=a, ignore_nulls=True).over("opp"))
    g = g.join(allowed.select("game_id", "opp", "opp_tda"), on=["game_id", "opp"], how="left")
    return g.rename({"pts": "team_pts"})


def forward(g: pl.DataFrame, L: pl.DataFrame) -> np.ndarray:
    """lambda per line, each 6-week block predicted by a model trained only on QB games before the block."""
    L = L.join(g.select("game_id", "player_id", *FEATS, "gameday"), on=["game_id", "player_id"], how="left")
    lam = np.full(L.height, np.nan)
    for S in SEASONS:
        for i, b0 in enumerate(BLOCKS):
            b1 = BLOCKS[i + 1] if i + 1 < len(BLOCKS) else 99
            blk = (L["season"] == S) & (L["week"] >= b0) & (L["week"] < b1)
            if not blk.any():
                continue
            start = g.filter((pl.col("season") == S) & (pl.col("week") >= b0))["gameday"].min()
            tr = g.filter((pl.col("gameday") < start) & pl.col("qb_td").is_not_null())
            m = HistGradientBoostingRegressor(loss="poisson", max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=100, early_stopping=False)
            m.fit(tr.select(FEATS).to_numpy().astype(float), tr["passing_tds"].to_numpy().astype(float))
            idx = np.flatnonzero(blk.to_numpy())
            lam[idx] = m.predict(L[idx].select(FEATS).to_numpy().astype(float))
    return L, lam


def main():
    pl.Config.set_tbl_rows(50)
    L = lines()
    print("passing-TD lines matched:", L.height, L.group_by("season").len().sort("season").rows())
    g = qb_games()
    L = L.join(g.select("game_id", "player_id", "passing_tds"), on=["game_id", "player_id"], how="inner")
    L, lam = forward(g, L)
    p_model = 1 - poisson.cdf(np.floor(L["line"].to_numpy()), lam)
    y = (L["passing_tds"].to_numpy() > L["line"].to_numpy()).astype(float)
    fair = L["fair_o"].to_numpy()
    season = L["season"].to_numpy()
    cl = lambda q: np.clip(q, 1e-4, 1 - 1e-4)
    ll = lambda q: -(y * np.log(cl(q)) + (1 - y) * np.log(1 - cl(q)))
    p = np.full(len(y), np.nan)
    for S in SEASONS:
        cur = season == S
        past = (season < S) & ~np.isnan(p_model)
        if past.sum() < 100:
            b = 0.5
        else:
            w = 0.5 ** (S - season[past])
            b = max(MIX_B, key=lambda b: (w * (ll(fair) - ll(b * p_model + (1 - b) * fair))[past]).sum())
        p[cur] = b * p_model[cur] + (1 - b) * fair[cur]
        print(S, f"model weight {b}", f"lines {cur.sum()}",
              f"log-loss gain vs book (x1000): model alone {1000 * (ll(fair) - ll(p_model))[cur].mean():+.2f}, mixed {1000 * (ll(fair) - ll(p))[cur].mean():+.2f}")
    L = L.with_columns(p_model=pl.Series(p_model), p_over=pl.Series(p), over_won=pl.Series(y), market=pl.lit("pass_tds"))
    L.write_parquet(OUT)
    # straights: either side, by edge
    eo, eu = p - L["be_o"].to_numpy(), (1 - p) - L["be_u"].to_numpy()
    over = eo > eu
    ed = np.where(over, eo, eu)
    won = np.where(over, y, 1 - y)
    dec = np.where(over, L["dec_o"].to_numpy(), L["dec_u"].to_numpy())
    ret = np.where(won == 1, dec - 1, -1.0)
    for e in (0.0, 0.02, 0.04, 0.06):
        sel = ed >= e
        print(f"edge >= {e:.0%}: {sel.sum()} bets, return {ret[sel].mean():+.3f}, by season",
              {int(s): (int((sel & (season == s)).sum()), round(float(ret[sel & (season == s)].mean()), 3) if (sel & (season == s)).any() else None) for s in SEASONS})
    print("unders only, edge>=2%:", ret[(~over) & (ed >= 0.02)].mean(), "overs only, edge>=2%:", ret[over & (ed >= 0.02)].mean())


if __name__ == "__main__":
    main()
