"""Comps v4: player vs style matchups (2026-10-20).

The comps v1-v3 compared whole games ("units that looked alike") and averaged the final stat's miss: too coarse and too noisy. v4 works at the play
level: how THIS player produces against a STYLE of defense, and how much of that style THIS defense plays.

Styles (play level; unknown = left out):
  man        man coverage (1) vs zone (0)                                  participation 2018-2025
  two_high   two-high shells (Cover 2/4/6, 2-man) vs single-high (0/1/3)  participation 2018-2025
  blitz      5+ rushers / any blitzer                                      FTN charting 2022-2026, else participation rushers 2016-2025
  pressure   QB pressured                                                  participation 2016-2025
  heavy_box  8+ in the box (1) vs 6 or fewer (0), run plays                FTN 2022-2026, else participation 2016-2025

Entering each game (only plays before it count; recency: half-life HALF_LIFE games of that team or player, across seasons):
  defense style rate      share of its labelled snaps in the style, shrunk toward the league
  player split rates      per on-field dropback: targets, catches, receiving yards, by style value (1 / 0), shrunk toward his overall rate (K plays);
                          QB: completions and passing yards per dropback by style; back: rushing yards per carry vs heavy / light box
  matchup multiplier      expected rate against THIS defense's style mix / expected rate against a league-average mix, per style and combined

  python -m comps4.style      # writes data/processed/comps4_style.parquet: (game_id, player_id, market) -> multipliers and defense rates
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config

OUT = config.PROCESSED_DIR / "comps4_style.parquet"
HALF_LIFE = 8.0          # team / player games
K_PLAYER = 60.0          # dropbacks (carries) of shrinkage toward the player's overall rate
K_DEF = 80.0             # labelled plays of shrinkage toward the league rate
PASS_DIMS = ("man", "two_high", "blitz", "pressure")
RUN_DIMS = ("heavy_box",)
TWO_HIGH = ("COVER_2", "COVER_4", "COVER_6", "2_MAN")
ONE_HIGH = ("COVER_0", "COVER_1", "COVER_3")


def plays() -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    p = con.execute("""
        WITH pb AS (SELECT * FROM (SELECT DISTINCT ON (game_id, play_id) game_id, play_id, season, week, posteam, defteam, qb_dropback, pass, rush,
                      passer_player_id, receiver_player_id, rusher_player_id, complete_pass, yards_gained, sack
                    FROM pbp WHERE season >= 2016 AND season_type = 'REG' ORDER BY game_id, play_id, pulled_at DESC)),
             pa AS (SELECT * FROM (SELECT DISTINCT ON (nflverse_game_id, play_id) nflverse_game_id, play_id, offense_players, defense_man_zone_type,
                      defense_coverage_type, was_pressure, number_of_pass_rushers, defenders_in_box
                    FROM participation ORDER BY nflverse_game_id, play_id, pulled_at DESC)),
             ft AS (SELECT * FROM (SELECT DISTINCT ON (nflverse_game_id, nflverse_play_id) nflverse_game_id, nflverse_play_id, n_blitzers, n_defense_box
                    FROM ftn_charting ORDER BY nflverse_game_id, nflverse_play_id, pulled_at DESC))
        SELECT pb.*, pa.offense_players, pa.defense_man_zone_type, pa.defense_coverage_type, pa.was_pressure, pa.number_of_pass_rushers,
               pa.defenders_in_box, ft.n_blitzers, ft.n_defense_box
        FROM pb LEFT JOIN pa ON pa.nflverse_game_id = pb.game_id AND pa.play_id = pb.play_id
                LEFT JOIN ft ON ft.nflverse_game_id = pb.game_id AND CAST(ft.nflverse_play_id AS DOUBLE) = pb.play_id
        WHERE (pb.qb_dropback = 1 OR pb.rush = 1) AND pb.posteam IS NOT NULL""").pl()
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, gameday FROM schedules ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    p = p.join(sched.with_columns(gameday=pl.col("gameday").cast(pl.Utf8).str.slice(0, 10)), on="game_id", how="left")
    mz, cv = pl.col("defense_man_zone_type"), pl.col("defense_coverage_type")
    box = pl.coalesce(pl.col("n_defense_box").cast(pl.Float64), pl.col("defenders_in_box").cast(pl.Float64))
    blitz = pl.coalesce((pl.col("n_blitzers").cast(pl.Float64) > 0).cast(pl.Float64), (pl.col("number_of_pass_rushers").cast(pl.Float64) >= 5).cast(pl.Float64))
    db = pl.col("qb_dropback") == 1
    return p.with_columns(
        man=pl.when(db & (mz == "MAN_COVERAGE")).then(1.0).when(db & (mz == "ZONE_COVERAGE")).then(0.0),
        two_high=pl.when(db & cv.is_in(TWO_HIGH)).then(1.0).when(db & cv.is_in(ONE_HIGH)).then(0.0),
        blitz=pl.when(db).then(blitz),
        pressure=pl.when(db & pl.col("was_pressure").is_not_null()).then(pl.col("was_pressure").cast(pl.Float64)),
        heavy_box=pl.when((pl.col("rush") == 1) & (box >= 8)).then(1.0).when((pl.col("rush") == 1) & (box <= 6)).then(0.0),
        yds=pl.col("yards_gained").fill_null(0).cast(pl.Float64), cmp=pl.col("complete_pass").fill_null(0).cast(pl.Float64))


def decayed(keys: pl.DataFrame, by: str, order: str, cols: list[str]) -> pl.DataFrame:
    """For each entity (`by`), per game in `order`, the recency-weighted sums of `cols` over its EARLIER games (half-life HALF_LIFE games)."""
    d = 0.5 ** (1 / HALF_LIFE)
    keys = keys.sort(by, order)
    ent = keys[by].to_numpy()
    X = keys.select(cols).to_numpy().astype(float)
    out = np.zeros_like(X)
    acc = np.zeros(len(cols))
    for i in range(len(ent)):
        if i == 0 or ent[i] != ent[i - 1]:
            acc = np.zeros(len(cols))
        out[i] = acc
        acc = acc * d + np.nan_to_num(X[i])
    return keys.with_columns([pl.Series(f"pre_{c}", out[:, j]) for j, c in enumerate(cols)])


def defense_rates(p: pl.DataFrame) -> pl.DataFrame:
    dims = PASS_DIMS + RUN_DIMS
    g = p.group_by("game_id", "gameday", "defteam").agg([pl.col(x).sum().alias(f"{x}_1") for x in dims] + [pl.col(x).is_not_null().sum().alias(f"{x}_n") for x in dims])
    cols = [f"{x}_{s}" for x in dims for s in ("1", "n")]
    g = decayed(g, "defteam", "gameday", cols)
    lg = {x: float(p[x].drop_nulls().mean()) for x in dims}
    return g.select("game_id", "defteam", *[((pl.col(f"pre_{x}_1") + K_DEF * lg[x]) / (pl.col(f"pre_{x}_n") + K_DEF)).alias(f"def_{x}") for x in dims]), lg


def receiver_splits(p: pl.DataFrame) -> pl.DataFrame:
    """Per (game, player on the field for dropbacks): his targets / catches / yards per dropback, by style value; then pre-game decayed sums."""
    db = p.filter((pl.col("qb_dropback") == 1) & pl.col("offense_players").is_not_null())
    db = db.with_columns(pid=pl.col("offense_players").str.split(";")).explode("pid")
    db = db.with_columns(tgt=(pl.col("receiver_player_id") == pl.col("pid")).cast(pl.Float64))
    db = db.with_columns(rec=pl.col("tgt") * pl.col("cmp"), ryd=pl.col("tgt") * pl.col("cmp") * pl.col("yds"))
    aggs = [pl.len().alias("n_all"), pl.col("tgt").sum().alias("tgt_all"), pl.col("rec").sum().alias("rec_all"), pl.col("ryd").sum().alias("ryd_all")]
    for x in PASS_DIMS:
        for v in (1.0, 0.0):
            s = f"{x}{int(v)}"
            m = pl.col(x) == v
            aggs += [m.sum().alias(f"n_{s}"), (pl.col("tgt") * m).sum().alias(f"tgt_{s}"), (pl.col("rec") * m).sum().alias(f"rec_{s}"),
                     (pl.col("ryd") * m).sum().alias(f"ryd_{s}")]
    g = db.group_by("game_id", "gameday", "pid", "posteam").agg(aggs)
    g = g.filter(pl.col("tgt_all") > 0)  # only players who were targeted at least once in the game carry a split (on-field linemen excluded)
    cols = [c for c in g.columns if c not in ("game_id", "gameday", "pid", "posteam")]
    return decayed(g, "pid", "gameday", cols)


def qb_splits(p: pl.DataFrame) -> pl.DataFrame:
    db = p.filter((pl.col("qb_dropback") == 1) & pl.col("passer_player_id").is_not_null())
    db = db.with_columns(pyd=pl.col("cmp") * pl.col("yds"))
    aggs = [pl.len().alias("n_all"), pl.col("cmp").sum().alias("cmp_all"), pl.col("pyd").sum().alias("pyd_all")]
    for x in PASS_DIMS:
        for v in (1.0, 0.0):
            s = f"{x}{int(v)}"
            m = pl.col(x) == v
            aggs += [m.sum().alias(f"n_{s}"), (pl.col("cmp") * m).sum().alias(f"cmp_{s}"), (pl.col("pyd") * m).sum().alias(f"pyd_{s}")]
    g = db.group_by("game_id", "gameday", pid="passer_player_id").agg(aggs)
    cols = [c for c in g.columns if c not in ("game_id", "gameday", "pid")]
    return decayed(g, "pid", "gameday", cols)


def rusher_splits(p: pl.DataFrame) -> pl.DataFrame:
    r = p.filter((pl.col("rush") == 1) & pl.col("rusher_player_id").is_not_null())
    aggs = [pl.len().alias("n_all"), pl.col("yds").sum().alias("ruy_all")]
    for v in (1.0, 0.0):
        m = pl.col("heavy_box") == v
        aggs += [m.sum().alias(f"n_heavy_box{int(v)}"), (pl.col("yds") * m).sum().alias(f"ruy_heavy_box{int(v)}")]
    g = r.group_by("game_id", "gameday", pid="rusher_player_id").agg(aggs)
    cols = [c for c in g.columns if c not in ("game_id", "gameday", "pid")]
    return decayed(g, "pid", "gameday", cols)


def multipliers(s: pl.DataFrame, metric: str, dims, D: pl.DataFrame, lg: dict) -> pl.DataFrame:
    """Matchup multiplier per style for `metric` (per play), shrunk split rates; and the product over styles."""
    overall = pl.col(f"pre_{metric}_all") / pl.col("pre_n_all")
    out = []
    for x in dims:
        r1 = (pl.col(f"pre_{metric}_{x}1") + K_PLAYER * overall) / (pl.col(f"pre_n_{x}1") + K_PLAYER)
        r0 = (pl.col(f"pre_{metric}_{x}0") + K_PLAYER * overall) / (pl.col(f"pre_n_{x}0") + K_PLAYER)
        m = pl.col(f"def_{x}")
        out.append(((m * r1 + (1 - m) * r0) / (lg[x] * r1 + (1 - lg[x]) * r0)).alias(f"mult_{x}"))
    s = s.join(D, on=["game_id", "defteam"], how="left").with_columns(out)
    return s.with_columns(mult_all=pl.reduce(lambda a, b: a * b, [pl.col(f"mult_{x}").fill_nan(1.0).fill_null(1.0) for x in dims]))


def build() -> pl.DataFrame:
    p = plays()
    D, lg = defense_rates(p)
    opp = p.select("game_id", "posteam", "defteam").unique()
    rec = receiver_splits(p).join(opp, on=["game_id", "posteam"], how="left")
    qb = qb_splits(p).join(p.select("game_id", pid="passer_player_id", defteam="defteam").drop_nulls().unique(["game_id", "pid"]), on=["game_id", "pid"], how="left")
    ru = rusher_splits(p).join(p.select("game_id", pid="rusher_player_id", defteam="defteam").drop_nulls().unique(["game_id", "pid"]), on=["game_id", "pid"], how="left")
    parts = []
    keep = lambda s, mk: s.select("game_id", player_id="pid", market=pl.lit(mk), mult_all="mult_all", **{f"mult_{x}": f"mult_{x}" for x in PASS_DIMS if f"mult_{x}" in s.columns},
                                   **{f"mult_{x}": f"mult_{x}" for x in RUN_DIMS if f"mult_{x}" in s.columns})
    for metric, mk in (("ryd", "rec_yds"), ("rec", "rec"), ("tgt", "targets")):
        parts.append(keep(multipliers(rec, metric, PASS_DIMS, D, lg), mk))
    for metric, mk in (("pyd", "pass_yds"), ("cmp", "pass_cmp")):
        parts.append(keep(multipliers(qb, metric, PASS_DIMS, D, lg), mk))
    parts.append(keep(multipliers(ru, "ruy", RUN_DIMS, D, lg), "rush_yds"))
    out = pl.concat(parts, how="diagonal_relaxed")
    return out.with_columns(pl.col(pl.Float64).fill_nan(None))


if __name__ == "__main__":
    import time
    t0 = time.time()
    s = build()
    s.write_parquet(OUT)
    print(s.shape, f"{time.time() - t0:.0f}s")
    print(s.group_by("market").agg(pl.len(), pl.col("mult_all").mean().alias("mean"), pl.col("mult_all").std().alias("sd"), pl.col("mult_all").quantile(0.05).alias("p05"), pl.col("mult_all").quantile(0.95).alias("p95")))
