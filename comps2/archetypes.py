"""Player archetypes, as of each game ("does Detroit's back resemble Bijan?").

Per player and game, from play-by-play: numerator / denominator pairs, then the recency-weighted ratio of the player's EARLIER games only (half-life
HALF_LIFE of his games, a season boundary counts SEASON_GAP extra games), standardized within his family (RB, WR, TE, QB) against the same season and
week's cross-section. FEATS lists the archetype of each family.
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from live import history as H

HALF_LIFE = 8.0
SEASON_GAP = 4.0
OUT = config.PROCESSED_DIR / "comps2_archetypes.parquet"
FEATS = {
    "RB": ("carry_share", "tgt_share", "ypc", "rush_epa", "expl_run", "inside_share", "gl_share", "adot"),
    "WR": ("tgt_share", "air_share", "adot", "deep_share", "catch_rate", "yac", "mid_share", "rz_tgt_share"),
    "TE": ("tgt_share", "air_share", "adot", "deep_share", "catch_rate", "yac", "mid_share", "rz_tgt_share"),
    "QB": ("adot", "scramble", "qb_runs", "sack_rate", "epa_db", "cpoe", "ypa"),
}
ALL = sorted({f for v in FEATS.values() for f in v})


def _plays():
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    d = con.execute("""
        SELECT * FROM (SELECT DISTINCT ON (game_id, play_id) game_id, play_id, season_type, posteam, rush, qb_dropback, qb_scramble, qb_kneel, sack, pass_attempt,
               complete_pass, epa, yards_gained, air_yards, yards_after_catch, cpoe, run_location, pass_location, yardline_100, rusher_player_id,
               receiver_player_id, passer_player_id
        FROM pbp ORDER BY game_id, play_id, pulled_at DESC)
        WHERE season_type = 'REG' AND posteam IS NOT NULL""").pl()
    con.close()
    return d


def player_game_stats(p: pl.DataFrame) -> pl.DataFrame:
    f = lambda c: pl.col(c).fill_null(0).cast(pl.Float64)
    p = p.with_columns(designed=(f("rush") == 1) & (f("qb_scramble") == 0) & (f("qb_kneel") == 0), att=(f("pass_attempt") == 1) & (f("sack") == 0),
                       y=f("yards_gained"))
    p = p.with_columns(tgt=pl.col("att") & pl.col("receiver_player_id").is_not_null())
    team = p.group_by("game_id", "posteam").agg(t_car=pl.col("designed").sum().cast(pl.Float64), t_tgt=pl.col("tgt").sum().cast(pl.Float64),
                                                t_air=pl.when(pl.col("tgt")).then(f("air_yards")).otherwise(0.0).sum(),
                                                t_gl=(pl.col("designed") & (f("yardline_100") <= 5)).sum().cast(pl.Float64),
                                                t_rz=(pl.col("tgt") & (f("yardline_100") <= 20)).sum().cast(pl.Float64))
    S = lambda cond, v: pl.when(cond).then(v).otherwise(0.0).sum()
    C = lambda cond: cond.cast(pl.Float64).sum()
    ru = p.filter(pl.col("designed")).group_by("game_id", "posteam", pid="rusher_player_id").agg(
        car=pl.len().cast(pl.Float64), ry=pl.col("y").sum(), repa=f("epa").sum(), r10=C(pl.col("y") >= 10), mid=C(pl.col("run_location") == "middle"),
        loc=C(pl.col("run_location").is_not_null()), gl=C(f("yardline_100") <= 5))
    re = p.filter(pl.col("tgt")).group_by("game_id", "posteam", pid="receiver_player_id").agg(
        tg=pl.len().cast(pl.Float64), air=f("air_yards").sum(), nair=C(pl.col("air_yards").is_not_null()), deep=C(f("air_yards") >= 20),
        rec=C(f("complete_pass") == 1), yac=S(f("complete_pass") == 1, f("yards_after_catch")), pmid=C(pl.col("pass_location") == "middle"),
        ploc=C(pl.col("pass_location").is_not_null()), rz=C(f("yardline_100") <= 20))
    qb = p.filter(f("qb_dropback") == 1).group_by("game_id", "posteam", pid=pl.coalesce("passer_player_id", "rusher_player_id")).agg(
        db=pl.len().cast(pl.Float64), sk=C(f("sack") == 1), scr=C(f("qb_scramble") == 1), epa=f("epa").sum(), qair=S(pl.col("att"), f("air_yards")),
        qnair=C(pl.col("att") & pl.col("air_yards").is_not_null()), cp=S(pl.col("cpoe").is_not_null(), f("cpoe")), ncp=C(pl.col("cpoe").is_not_null()),
        qatt=C(pl.col("att")), qy=S(pl.col("att"), pl.col("y")))
    g = ru.join(re, on=["game_id", "posteam", "pid"], how="full", coalesce=True).join(qb, on=["game_id", "posteam", "pid"], how="full", coalesce=True)
    g = g.join(team, on=["game_id", "posteam"], how="left").fill_null(0.0).filter(pl.col("pid").is_not_null())
    qr = ru.select("game_id", "pid", qcar="car")
    g = g.join(qr, on=["game_id", "pid"], how="left").with_columns(pl.col("qcar").fill_null(0.0))
    pairs = dict(carry_share=("car", "t_car"), tgt_share=("tg", "t_tgt"), ypc=("ry", "car"), rush_epa=("repa", "car"), expl_run=("r10", "car"),
                 inside_share=("mid", "loc"), gl_share=("gl", "t_gl"), adot=("air", "nair"), air_share=("air", "t_air"), deep_share=("deep", "nair"),
                 catch_rate=("rec", "tg"), yac=("yac", "rec"), mid_share=("pmid", "ploc"), rz_tgt_share=("rz", "t_rz"),
                 scramble=("scr", "db"), qb_runs=("qcar", None), sack_rate=("sk", "db"), epa_db=("epa", "db"), cpoe=("cp", "ncp"), ypa=("qy", "qatt"))
    cols = []
    for k, (a, b) in pairs.items():
        cols += [pl.col(a).alias(f"{k}_n"), (pl.col(b) if b else pl.lit(1.0)).alias(f"{k}_d")]
    # a QB's adot is his passes' depth, not his targets'
    g = g.with_columns(cols).with_columns(adot_n=pl.when(pl.col("db") > 0).then(pl.col("qair")).otherwise(pl.col("adot_n")),
                                          adot_d=pl.when(pl.col("db") > 0).then(pl.col("qnair")).otherwise(pl.col("adot_d")))
    return g.select("game_id", pl.col("posteam").alias("team"), pl.col("pid").alias("player_id"), *[f"{k}_{s}" for k in pairs for s in ("n", "d")])


def build() -> pl.DataFrame:
    log = H.read_played_logs(max(config.BACKTEST_SEASONS) + 2)[0].select("player_id", "game_id", "season", "week", "gameday", "family")
    g = player_game_stats(_plays()).join(log, on=["player_id", "game_id"], how="inner").sort("player_id", "gameday")
    rows = []
    for (pid,), t in g.group_by("player_id", maintain_order=True):
        N = t.select([f"{s}_n" for s in ALL]).to_numpy()
        D = t.select([f"{s}_d" for s in ALL]).to_numpy()
        se = t["season"].to_numpy()
        pos = np.arange(len(t)) + SEASON_GAP * (se - se.min())
        for i in range(len(t)):
            if i == 0:
                rows.append((pid, t["game_id"][i], 0.0, *([np.nan] * len(ALL))))
                continue
            w = 0.5 ** ((pos[i] - pos[:i]) / HALF_LIFE)
            num, den = (w[:, None] * N[:i]).sum(0), (w[:, None] * D[:i]).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                rows.append((pid, t["game_id"][i], float(w.sum()), *np.where(den > 0, num / den, np.nan)))
    a = pl.DataFrame(rows, schema=["player_id", "game_id", "weight_games", *ALL], orient="row", infer_schema_length=None)
    a = a.join(log, on=["player_id", "game_id"], how="left")
    # standardize inside the family, same season and week
    a = a.with_columns([((pl.col(s) - pl.col(s).mean().over("family", "season", "week")) / pl.col(s).std().over("family", "season", "week")).alias(s) for s in ALL])
    return a


if __name__ == "__main__":
    import time
    t0 = time.time()
    a = build()
    a.write_parquet(OUT)
    print(a.shape, f"{time.time() - t0:.0f}s")
    print(a.group_by("family").len())
