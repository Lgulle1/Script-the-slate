"""Unit fingerprints, as of each game (yours, 2026-10-19: "one team can resemble Detroit in the running game and Green Bay in passing structure").

Per team and game, from play-by-play (every season 2016-2026, so the final exam has them too), each stat is kept as a numerator and a denominator for the
OFFENSE (posteam) and for the DEFENSE (defteam: what it allowed). The fingerprint a team carries INTO a game is the recency-weighted sum of its earlier
games' numerators over the same sum of denominators (half-life HALF_LIFE games, a season boundary counts SEASON_GAP extra games), so the target game and
everything after it never enter. Then every stat is standardized against the teams' fingerprints in the same season and week (z = how unusual vs the league).

UNITS lists which stats form each fingerprint. Offense and defense use the same stat names, so the run offense of one team and the run defense of another
live in the same space (that is what the interaction search uses).
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from live import history as H

HALF_LIFE = 6.0
SEASON_GAP = 4.0
OUT = config.PROCESSED_DIR / "comps2_fingerprints.parquet"

UNITS = {
    "run": ("rush_epa", "rush_succ", "ypc", "expl_run", "stuff", "loc_left", "loc_mid", "loc_right", "gap_end", "gap_tackle", "gap_guard", "sg_run",
            "rb1_share", "early_rush", "proe", "qb_run"),
    "pass": ("epa_db", "sack_rate", "scramble", "adot", "deep", "short", "cpoe", "comp_rate", "ypa", "expl_pass", "yac", "ploc_left", "ploc_mid",
             "ploc_right", "tgt_wr", "tgt_te", "tgt_rb", "top1", "top3", "neutral_pass", "no_huddle", "shotgun"),
    "pace": ("plays", "neutral_pass", "no_huddle"),
}
STATS = sorted({s for u in UNITS.values() for s in u})


def plays(raw_db=config.RAW_DUCKDB_PATH) -> pl.DataFrame:
    con = duckdb.connect(str(raw_db), read_only=True)
    d = con.execute("""
        SELECT * FROM (SELECT DISTINCT ON (game_id, play_id) game_id, play_id, season, week, season_type, posteam, defteam, play_type, rush, pass, qb_dropback,
               qb_scramble, qb_kneel, sack, pass_attempt, complete_pass, epa, success, yards_gained, air_yards, yards_after_catch, cpoe, pass_oe, run_location,
               run_gap, pass_location, shotgun, no_huddle, down, wp, rusher_player_id, receiver_player_id
        FROM pbp ORDER BY game_id, play_id, pulled_at DESC)
        WHERE season_type = 'REG' AND play_type IN ('run', 'pass') AND posteam IS NOT NULL""").pl()
    con.close()
    return d


def game_stats(p: pl.DataFrame, fam: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team, side) with <stat>_n and <stat>_d columns."""
    p = p.join(fam.rename({"player_id": "rusher_player_id", "family": "rfam"}), on="rusher_player_id", how="left")
    p = p.join(fam.rename({"player_id": "receiver_player_id", "family": "tfam"}), on="receiver_player_id", how="left")
    f = lambda c: pl.col(c).fill_null(0).cast(pl.Float64)
    designed = (f("rush") == 1) & (f("qb_scramble") == 0) & (f("qb_kneel") == 0)
    att = (f("pass_attempt") == 1) & (f("sack") == 0)
    neutral = pl.col("wp").is_between(0.2, 0.8) & (f("down") <= 3) & (f("down") >= 1)
    early = neutral & (f("down") <= 2)
    has_air = att & pl.col("air_yards").is_not_null()
    tgt = att & pl.col("receiver_player_id").is_not_null()
    p = p.with_columns(designed=designed, att=att, neutral=neutral, early=early, has_air=has_air, tgt=tgt, y=f("yards_gained"))
    n = lambda e: e.cast(pl.Float64).sum()
    W = lambda cond, val: pl.when(cond).then(val).otherwise(0.0).sum()
    aggs = dict(
        rush_epa=(W(designed, f("epa")), n(designed)), rush_succ=(W(designed, f("success")), n(designed)), ypc=(W(designed, pl.col("y")), n(designed)),
        expl_run=(n(designed & (pl.col("y") >= 10)), n(designed)), stuff=(n(designed & (pl.col("y") <= 0)), n(designed)),
        loc_left=(n(designed & (pl.col("run_location") == "left")), n(designed & pl.col("run_location").is_not_null())),
        loc_mid=(n(designed & (pl.col("run_location") == "middle")), n(designed & pl.col("run_location").is_not_null())),
        loc_right=(n(designed & (pl.col("run_location") == "right")), n(designed & pl.col("run_location").is_not_null())),
        gap_end=(n(designed & (pl.col("run_gap") == "end")), n(designed)), gap_tackle=(n(designed & (pl.col("run_gap") == "tackle")), n(designed)),
        gap_guard=(n(designed & (pl.col("run_gap") == "guard")), n(designed)), sg_run=(n(designed & (f("shotgun") == 1)), n(designed)),
        qb_run=(n(designed & (pl.col("rfam") == "QB")), n(designed)),
        early_rush=(n(early & designed), n(early & (designed | pl.col("att") | (f("sack") == 1)))),
        proe=(W(pl.col("pass_oe").is_not_null(), f("pass_oe")), n(pl.col("pass_oe").is_not_null())),
        epa_db=(W(f("qb_dropback") == 1, f("epa")), n(f("qb_dropback") == 1)), sack_rate=(n(f("sack") == 1), n(f("qb_dropback") == 1)),
        scramble=(n(f("qb_scramble") == 1), n(f("qb_dropback") == 1)),
        adot=(W(has_air, f("air_yards")), n(has_air)), deep=(n(has_air & (f("air_yards") >= 20)), n(has_air)), short=(n(has_air & (f("air_yards") <= 5)), n(has_air)),
        cpoe=(W(pl.col("cpoe").is_not_null(), f("cpoe")), n(pl.col("cpoe").is_not_null())), comp_rate=(n(att & (f("complete_pass") == 1)), n(att)),
        ypa=(W(att, pl.col("y")), n(att)), expl_pass=(n(att & (pl.col("y") >= 20)), n(att)),
        yac=(W(att & (f("complete_pass") == 1), f("yards_after_catch")), n(att & (f("complete_pass") == 1))),
        ploc_left=(n(att & (pl.col("pass_location") == "left")), n(att & pl.col("pass_location").is_not_null())),
        ploc_mid=(n(att & (pl.col("pass_location") == "middle")), n(att & pl.col("pass_location").is_not_null())),
        ploc_right=(n(att & (pl.col("pass_location") == "right")), n(att & pl.col("pass_location").is_not_null())),
        tgt_wr=(n(tgt & (pl.col("tfam") == "WR")), n(tgt)), tgt_te=(n(tgt & (pl.col("tfam") == "TE")), n(tgt)), tgt_rb=(n(tgt & (pl.col("tfam") == "RB")), n(tgt)),
        neutral_pass=(n(neutral & (pl.col("att") | (f("sack") == 1) | (f("qb_scramble") == 1))), n(neutral & (designed | pl.col("att") | (f("sack") == 1) | (f("qb_scramble") == 1)))),
        no_huddle=(n(f("no_huddle") == 1), pl.len().cast(pl.Float64)), shotgun=(n(f("shotgun") == 1), pl.len().cast(pl.Float64)),
        plays=(pl.len().cast(pl.Float64), pl.lit(1.0)),
    )
    exprs = [e.alias(f"{k}_{s}") for k, (a, b) in aggs.items() for s, e in (("n", a), ("d", b))]
    # receiver concentration and RB1 share need a per-player count first
    tg = p.filter(pl.col("tgt")).group_by("game_id", "posteam", "receiver_player_id").len()
    tg = tg.sort("len", descending=True).group_by("game_id", "posteam").agg(top1_n=pl.col("len").first().cast(pl.Float64), top3_n=pl.col("len").head(3).sum().cast(pl.Float64),
                                                                              top1_d=pl.col("len").sum().cast(pl.Float64))
    tg = tg.with_columns(top3_d=pl.col("top1_d"))
    rb = p.filter(pl.col("designed") & (pl.col("rfam") == "RB")).group_by("game_id", "posteam", "rusher_player_id").len()
    rb = rb.sort("len", descending=True).group_by("game_id", "posteam").agg(rb1_share_n=pl.col("len").first().cast(pl.Float64), rb1_share_d=pl.col("len").sum().cast(pl.Float64))
    out = []
    for side, team in (("off", "posteam"), ("def", "defteam")):
        g = p.group_by("game_id", "season", "week", team).agg(exprs).rename({team: "team"})
        extra = tg.join(rb, on=["game_id", "posteam"], how="full", coalesce=True).rename({"posteam": "team"})
        if side == "def":
            opp = p.select("game_id", "posteam", "defteam").unique().rename({"posteam": "o", "defteam": "team"})
            extra = extra.rename({"team": "o"}).join(opp, on=["game_id", "o"], how="inner").drop("o")
        g = g.join(extra, on=["game_id", "team"], how="left").with_columns(side=pl.lit(side))
        out.append(g)
    return pl.concat(out, how="diagonal_relaxed").fill_null(0.0)


def as_of(g: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Fingerprint each team carries INTO each game: recency-weighted ratios of earlier games only."""
    g = g.join(sched, on="game_id", how="left").sort("team", "side", "gameday")
    rows = []
    for (team, side), t in g.group_by("team", "side", maintain_order=True):
        N = t.select([f"{s}_n" for s in STATS]).to_numpy()
        D = t.select([f"{s}_d" for s in STATS]).to_numpy()
        seasons = t["season"].to_numpy()
        # games-ago with a season penalty: position index plus SEASON_GAP per season boundary crossed
        pos = np.arange(len(t)) + SEASON_GAP * (seasons - seasons.min())
        for i in range(len(t)):
            if i == 0:
                fp = np.full(len(STATS), np.nan)
                eff = 0.0
            else:
                w = 0.5 ** ((pos[i] - pos[:i]) / HALF_LIFE)
                num = (w[:, None] * N[:i]).sum(0)
                den = (w[:, None] * D[:i]).sum(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    fp = np.where(den > 0, num / den, np.nan)
                eff = float(w.sum())
            rows.append((t["game_id"][i], team, side, eff, *fp))
    a = pl.DataFrame(rows, schema=["game_id", "team", "side", "weight_games", *STATS], orient="row")
    return a.join(sched, on="game_id", how="left")


def standardize(a: pl.DataFrame) -> pl.DataFrame:
    """z against the same season and week's cross-section of fingerprints (offense and defense pooled per stat, so they share one scale)."""
    a = a.with_columns([pl.col(s).fill_nan(None) for s in STATS])
    zs = [((pl.col(s) - pl.col(s).mean().over("season", "week")) / pl.col(s).std().over("season", "week")).alias(s) for s in STATS]
    return a.with_columns(zs)


def build() -> pl.DataFrame:
    log = H.read_played_logs(max(config.BACKTEST_SEASONS) + 2)[0]
    fam = log.sort("gameday").unique("player_id", keep="last").select("player_id", "family")
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, season, week, gameday FROM schedules WHERE game_type = 'REG' ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    sched = sched.with_columns(pl.col("gameday").cast(pl.Utf8).str.slice(0, 10).str.to_date(strict=False), season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32))
    g = game_stats(plays(), fam)
    g = g.drop("season", "week")
    a = as_of(g, sched)
    return standardize(a).sort("season", "week", "game_id", "team", "side")


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = build()
    d.write_parquet(OUT)
    print(d.shape, f"{time.time() - t0:.0f}s")
    print(d.group_by("season").agg(n=pl.len(), nulls=pl.col("ypc").null_count()).sort("season"))
