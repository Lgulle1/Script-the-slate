"""Rich unit stats per team and game (comps v3, 2026-10-19): the fingerprints the way they were asked for.

Every play is play-by-play joined with FTN charting (2022-2026: motion, play-action, screens, RPO, QB location, box count, blitzers, pass rushers,
out-of-pocket, catchable / contested / drop) and participation (2016-2025: personnel, defenders in box, man / zone, coverage shell, time to throw, pressure,
route of the targeted receiver). Each stat is a numerator / denominator pair per (game, team, side): side "off" groups by the offense, side "def" by the
defense (what it allowed, and how it played: its boxes, blitzes, coverages). A stat whose source is missing for a game has denominator 0 there and simply
does not count in the profile (profiles are ratios of weighted sums).

UNITS: which stats form each unit's fingerprint. Sub-units for position defenses (te_*, wr1_*, rb_rec_*) answer "how does this defense handle TEs / the
top receiver / backs in the passing game".
"""
from __future__ import annotations

import duckdb
import polars as pl

import config
from live import history as H

OUT = config.PROCESSED_DIR / "comps3_unit_games.parquet"


def plays() -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    p = con.execute("""
        SELECT * FROM (SELECT DISTINCT ON (game_id, play_id) game_id, play_id, season, week, season_type, posteam, defteam, play_type, rush, pass, qb_dropback,
               qb_scramble, qb_kneel, sack, pass_attempt, complete_pass, epa, success, yards_gained, air_yards, yards_after_catch, cpoe, pass_oe, run_location,
               run_gap, pass_location, shotgun, no_huddle, down, wp, yardline_100, rusher_player_id, receiver_player_id, passer_player_id
        FROM pbp ORDER BY game_id, play_id, pulled_at DESC)
        WHERE season_type = 'REG' AND play_type IN ('run', 'pass') AND posteam IS NOT NULL""").pl()
    ftn = con.execute("""
        SELECT * FROM (SELECT DISTINCT ON (nflverse_game_id, nflverse_play_id) nflverse_game_id AS game_id, nflverse_play_id AS play_id, trim(qb_location) AS qb_loc,
               n_offense_backfield, n_defense_box AS ftn_box, is_motion, is_play_action, is_screen_pass, is_rpo, is_qb_out_of_pocket, is_catchable_ball,
               is_contested_ball, is_drop, n_blitzers, n_pass_rushers AS ftn_rushers
        FROM ftn_charting ORDER BY nflverse_game_id, nflverse_play_id, pulled_at DESC)""").pl()
    par = con.execute("""
        SELECT * FROM (SELECT DISTINCT ON (nflverse_game_id, play_id) nflverse_game_id AS game_id, play_id, offense_personnel, defenders_in_box AS par_box,
               number_of_pass_rushers AS par_rushers, defense_man_zone_type, defense_coverage_type, time_to_throw, was_pressure, route
        FROM participation ORDER BY nflverse_game_id, play_id, pulled_at DESC)""").pl()
    con.close()
    p = p.with_columns(pl.col("play_id").cast(pl.Int64))
    ftn = ftn.with_columns(pl.col("play_id").cast(pl.Int64))
    par = par.with_columns(pl.col("play_id").cast(pl.Int64))
    p = p.join(ftn, on=["game_id", "play_id"], how="left").join(par, on=["game_id", "play_id"], how="left")
    pers = pl.col("offense_personnel").fill_null("")
    return p.with_columns(
        n_rb=pers.str.extract(r"(\d) RB", 1).cast(pl.Int32, strict=False), n_te=pers.str.extract(r"(\d) TE", 1).cast(pl.Int32, strict=False),
        box=pl.coalesce(pl.col("ftn_box").cast(pl.Float64), pl.col("par_box").cast(pl.Float64)),
        rushers=pl.coalesce(pl.col("ftn_rushers").cast(pl.Float64), pl.col("par_rushers").cast(pl.Float64)))


def stats(p: pl.DataFrame, fam: pl.DataFrame) -> pl.DataFrame:
    p = p.join(fam.rename({"player_id": "rusher_player_id", "family": "rfam"}), on="rusher_player_id", how="left")
    p = p.join(fam.rename({"player_id": "receiver_player_id", "family": "tfam"}), on="receiver_player_id", how="left")
    f = lambda c: pl.col(c).fill_null(0).cast(pl.Float64)
    b = lambda c: pl.col(c).cast(pl.Float64).fill_null(0) > 0
    R = (f("rush") == 1) & (f("qb_scramble") == 0) & (f("qb_kneel") == 0)          # designed run
    DB = f("qb_dropback") == 1
    A = (f("pass_attempt") == 1) & (f("sack") == 0)
    T = A & pl.col("receiver_player_id").is_not_null()
    ftn = pl.col("qb_loc").is_not_null()
    hasbox = pl.col("box").is_not_null()
    haspers = pl.col("n_te").is_not_null()
    hascov = pl.col("defense_coverage_type").is_not_null() & (pl.col("defense_coverage_type") != "")
    hasmz = pl.col("defense_man_zone_type").is_in(["MAN_COVERAGE", "ZONE_COVERAGE"])
    neutral = pl.col("wp").is_between(0.2, 0.8) & f("down").is_between(1, 3)
    y = f("yards_gained")
    outside = R & ((pl.col("run_gap") == "end") | ((pl.col("run_location") != "middle") & pl.col("run_gap").is_null() & pl.col("run_location").is_not_null()))
    inside = R & ((pl.col("run_location") == "middle") | (pl.col("run_gap") == "guard"))
    top = p.filter(T).group_by("game_id", "posteam", "receiver_player_id").len().sort("len", descending=True).group_by("game_id", "posteam").agg(
        wr1=pl.col("receiver_player_id").first())
    p = p.join(top, on=["game_id", "posteam"], how="left").with_columns(is_wr1=(pl.col("receiver_player_id") == pl.col("wr1")).fill_null(False))
    S = lambda cond, v: pl.when(cond).then(v).otherwise(0.0).sum()
    C = lambda cond: cond.cast(pl.Float64).sum()
    defs = {
        # ---- run game
        "r_epa": (S(R, f("epa")), C(R)), "r_succ": (S(R, f("success")), C(R)), "r_ypc": (S(R, y), C(R)), "r_expl": (C(R & (y >= 10)), C(R)),
        "r_stuff": (C(R & (y <= 0)), C(R)), "r_out_share": (C(outside), C(R)), "r_in_share": (C(inside), C(R)),
        "r_out_epa": (S(outside, f("epa")), C(outside)), "r_in_epa": (S(inside, f("epa")), C(inside)),
        "r_sg": (C(R & (f("shotgun") == 1)), C(R)), "r_uc": (C(R & (pl.col("qb_loc") == "U")), C(R & ftn)), "r_motion": (C(R & b("is_motion")), C(R & ftn)),
        "r_2back": (C(R & (f("n_offense_backfield") >= 2)), C(R & ftn)), "r_rpo": (C(R & b("is_rpo")), C(R & ftn)),
        "r_light": (C(R & (pl.col("box") <= 6)), C(R & hasbox)), "r_heavy": (C(R & (pl.col("box") >= 8)), C(R & hasbox)),
        "r_light_epa": (S(R & (pl.col("box") <= 6), f("epa")), C(R & (pl.col("box") <= 6))), "r_heavy_epa": (S(R & (pl.col("box") >= 8), f("epa")), C(R & (pl.col("box") >= 8))),
        "r_box": (S(R & hasbox, pl.col("box")), C(R & hasbox)),
        "r_heavy_pers": (C(R & ((pl.col("n_te") >= 2) | (pl.col("n_rb") >= 2))), C(R & haspers)),
        "r_rate_early": (C(neutral & (f("down") <= 2) & R), C(neutral & (f("down") <= 2) & (R | A | (f("sack") == 1)))),
        "proe": (S(pl.col("pass_oe").is_not_null(), f("pass_oe")), C(pl.col("pass_oe").is_not_null())),
        "rb1_share": None, "qb_run": (C(R & (pl.col("rfam") == "QB")), C(R)),
        # ---- pass game
        "p_epa": (S(DB, f("epa")), C(DB)), "p_sack": (C(f("sack") == 1), C(DB)), "p_scr": (C(f("qb_scramble") == 1), C(DB)),
        "p_adot": (S(A & pl.col("air_yards").is_not_null(), f("air_yards")), C(A & pl.col("air_yards").is_not_null())),
        "p_deep": (C(A & (f("air_yards") >= 20)), C(A & pl.col("air_yards").is_not_null())), "p_short": (C(A & (f("air_yards") <= 5)), C(A & pl.col("air_yards").is_not_null())),
        "p_cpoe": (S(pl.col("cpoe").is_not_null(), f("cpoe")), C(pl.col("cpoe").is_not_null())), "p_comp": (C(A & (f("complete_pass") == 1)), C(A)),
        "p_ypa": (S(A, y), C(A)), "p_expl": (C(A & (y >= 20)), C(A)), "p_yac": (S(A & (f("complete_pass") == 1), f("yards_after_catch")), C(A & (f("complete_pass") == 1))),
        "p_mid": (C(A & (pl.col("pass_location") == "middle")), C(A & pl.col("pass_location").is_not_null())),
        "p_pa": (C(DB & b("is_play_action")), C(DB & ftn)), "p_motion": (C(DB & b("is_motion")), C(DB & ftn)), "p_screen": (C(DB & b("is_screen_pass")), C(DB & ftn)),
        "p_oop": (C(DB & b("is_qb_out_of_pocket")), C(DB & ftn)), "p_uc": (C(DB & (pl.col("qb_loc") == "U")), C(DB & ftn)),
        "p_pa_epa": (S(DB & b("is_play_action"), f("epa")), C(DB & b("is_play_action"))),
        "p_ttt": (S(DB & pl.col("time_to_throw").is_not_null(), f("time_to_throw")), C(DB & pl.col("time_to_throw").is_not_null())),
        "p_press": (C(DB & b("was_pressure")), C(DB & pl.col("was_pressure").is_not_null())),
        "p_blitz": (C(DB & (pl.col("rushers") >= 5)), C(DB & pl.col("rushers").is_not_null())), "p_rushers": (S(DB & pl.col("rushers").is_not_null(), pl.col("rushers")), C(DB & pl.col("rushers").is_not_null())),
        "p_man": (C(DB & (pl.col("defense_man_zone_type") == "MAN_COVERAGE")), C(DB & hasmz)),
        **{f"p_cov{k}": (C(DB & pl.col("defense_coverage_type").is_in(v)), C(DB & hascov)) for k, v in
           (("0", ["COVER_0"]), ("1", ["COVER_1"]), ("2", ["COVER_2", "2_MAN"]), ("3", ["COVER_3"]), ("4", ["COVER_4", "COVER_6"]))},
        "p_catchable": (C(T & b("is_catchable_ball")), C(T & ftn)), "p_contested": (C(T & b("is_contested_ball")), C(T & ftn)), "p_drop": (C(T & b("is_drop")), C(T & ftn)),
        "p_route_deep": (C(T & pl.col("route").is_in(["GO", "POST", "CORNER", "WHEEL"])), C(T & pl.col("route").is_not_null() & (pl.col("route") != ""))),
        "p_route_short": (C(T & pl.col("route").is_in(["FLAT", "SCREEN", "HITCH", "QUICK OUT", "SLANT", "SHALLOW CROSS/DRAG", "ANGLE"])), C(T & pl.col("route").is_not_null() & (pl.col("route") != ""))),
        "tgt_wr": (C(T & (pl.col("tfam") == "WR")), C(T)), "tgt_te": (C(T & (pl.col("tfam") == "TE")), C(T)), "tgt_rb": (C(T & (pl.col("tfam") == "RB")), C(T)),
        # ---- position sub-units: how targets to each position go
        "te_ypt": (S(T & (pl.col("tfam") == "TE"), y), C(T & (pl.col("tfam") == "TE"))), "te_epa": (S(T & (pl.col("tfam") == "TE"), f("epa")), C(T & (pl.col("tfam") == "TE"))),
        "te_catch": (C(T & (pl.col("tfam") == "TE") & (f("complete_pass") == 1)), C(T & (pl.col("tfam") == "TE"))),
        "wr_ypt": (S(T & (pl.col("tfam") == "WR"), y), C(T & (pl.col("tfam") == "WR"))), "wr_catch": (C(T & (pl.col("tfam") == "WR") & (f("complete_pass") == 1)), C(T & (pl.col("tfam") == "WR"))),
        "wr1_ypt": (S(T & pl.col("is_wr1"), y), C(T & pl.col("is_wr1"))), "wr1_share": (C(T & pl.col("is_wr1")), C(T)),
        "rb_rec_ypt": (S(T & (pl.col("tfam") == "RB"), y), C(T & (pl.col("tfam") == "RB"))), "rb_rec_catch": (C(T & (pl.col("tfam") == "RB") & (f("complete_pass") == 1)), C(T & (pl.col("tfam") == "RB"))),
        # ---- pace and tendency
        "neutral_pass": (C(neutral & (A | (f("sack") == 1) | (f("qb_scramble") == 1))), C(neutral & (R | A | (f("sack") == 1) | (f("qb_scramble") == 1)))),
        "no_huddle": (C(f("no_huddle") == 1), pl.len().cast(pl.Float64)), "shotgun": (C(f("shotgun") == 1), pl.len().cast(pl.Float64)),
        "plays": (pl.len().cast(pl.Float64), pl.lit(1.0)),
    }
    rb = p.filter(R & (pl.col("rfam") == "RB")).group_by("game_id", "posteam", "rusher_player_id").len().sort("len", descending=True).group_by("game_id", "posteam").agg(
        rb1_share_n=pl.col("len").first().cast(pl.Float64), rb1_share_d=pl.col("len").sum().cast(pl.Float64))
    exprs = [e.alias(f"{k}_{s}") for k, v in defs.items() if v for s, e in zip("nd", v)]
    out = []
    for side, team in (("off", "posteam"), ("def", "defteam")):
        g = p.group_by("game_id", "season", "week", team).agg(exprs).rename({team: "team"})
        if side == "off":
            g = g.join(rb.rename({"posteam": "team"}), on=["game_id", "team"], how="left")
        else:
            g = g.join(rb.join(p.select("game_id", "posteam", "defteam").unique(), on=["game_id", "posteam"]).drop("posteam").rename({"defteam": "team"}),
                       on=["game_id", "team"], how="left")
        out.append(g.with_columns(side=pl.lit(side)))
    return pl.concat(out, how="diagonal_relaxed").fill_null(0.0)


UNITS = {
    "run": ("r_epa", "r_succ", "r_ypc", "r_expl", "r_stuff", "r_out_share", "r_in_share", "r_out_epa", "r_in_epa", "r_sg", "r_uc", "r_motion", "r_2back",
            "r_rpo", "r_light", "r_heavy", "r_light_epa", "r_heavy_epa", "r_box", "r_heavy_pers", "r_rate_early", "proe", "rb1_share", "qb_run"),
    "pass": ("p_epa", "p_sack", "p_scr", "p_adot", "p_deep", "p_short", "p_cpoe", "p_comp", "p_ypa", "p_expl", "p_yac", "p_mid", "p_pa", "p_motion", "p_screen",
             "p_oop", "p_uc", "p_pa_epa", "p_ttt", "p_press", "p_blitz", "p_rushers", "p_man", "p_cov0", "p_cov1", "p_cov2", "p_cov3", "p_cov4",
             "p_catchable", "p_contested", "p_drop", "p_route_deep", "p_route_short", "tgt_wr", "tgt_te", "tgt_rb", "neutral_pass"),
    "te": ("te_ypt", "te_epa", "te_catch", "tgt_te"),
    "wr": ("wr_ypt", "wr_catch", "wr1_ypt", "wr1_share", "tgt_wr"),
    "rb_rec": ("rb_rec_ypt", "rb_rec_catch", "tgt_rb"),
    "pace": ("plays", "neutral_pass", "no_huddle", "shotgun"),
}
STATS = sorted({s for u in UNITS.values() for s in u})


def build() -> pl.DataFrame:
    log = H.read_played_logs(max(config.BACKTEST_SEASONS) + 2)[0]
    fam = log.sort("gameday").unique("player_id", keep="last").select("player_id", "family")
    return stats(plays(), fam)


if __name__ == "__main__":
    import time
    t0 = time.time()
    g = build()
    g.write_parquet(OUT)
    print(g.shape, f"{time.time() - t0:.0f}s")
    cov = g.group_by("season").agg([(pl.col(f"{s}_d") > 0).mean().alias(s) for s in ("r_motion", "r_light", "p_press", "p_cov3", "p_route_deep", "r_heavy_pers")]).sort("season")
    print(cov)
