"""Residual pattern screen (docs/overnight_plan.md, "Residual pattern screen"): do signals the Phase 3 models do not use carry information about their misses?

Development seasons 2020-2024 only (asserted); no sportsbook column is read. Every candidate is as-of: a trailing statistic of games strictly before the target game
(join_asof without exact matches), or a schedule fact known in advance. Output: data/processed/residual_screen_{cells,coverage}.parquet and a printed table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from eval import compare

LAST = config.HOLDOUT_SEASON - 1
PRED = config.PROCESSED_DIR / f"volume_efficiency_predictions_{config.ELIGIBLE_PLAYER_RULE['version']}.parquet"
MARKETS = ["targets", "rush_att", "pass_att", "rec_yds", "rush_yds", "pass_yds"]
REC, RUSH, PASS = ["targets", "rec_yds"], ["rush_att", "rush_yds"], ["pass_att", "pass_yds"]
ALL = MARKETS
MIN_COVERAGE = 0.70
FOLDS = [2021, 2022, 2023, 2024]

# candidate -> markets (the plan's fixed list; X21 counts as two candidates)
CANDIDATES = {
    "x1_own_rest": ALL, "x2_opp_rest": ALL, "x3_after_bye": ALL, "x4_primetime": ALL, "x5_divisional": ALL, "x6_week": ALL,
    "x7_ref_plays": ALL, "x8_ref_penalties": ALL,
    "x9_snap_trend": REC + RUSH, "x10_target_share_trend": REC, "x11_air_yards_share": REC, "x12_adot": REC, "x13_rec_luck": REC,
    "x14_separation": REC, "x15_yac_above_exp": REC,
    "x16_carry_share_trend": RUSH, "x17_rush_luck": RUSH,
    "x18_neutral_pass_tendency": ["pass_att", "rush_att", "targets"], "x19_no_huddle": ["pass_att", "rush_att"],
    "x20_opp_pressure": ["pass_att", "pass_yds", "targets"],
    "x21a_opp_man_rate": ["targets", "rec_yds", "pass_yds"], "x21b_opp_blitz_rate": ["targets", "rec_yds", "pass_yds"],
}


def _con():
    return duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)


def trailing(df: pl.DataFrame, by: str, col: str, k: int, min_n: int, name: str, date="gameday") -> pl.DataFrame:
    """Per row: mean of `col` over this row and the k-1 earlier rows of the same `by` entity (an as-of table for `asof`)."""
    return (df.sort(by, date).with_columns(pl.col(col).rolling_mean(window_size=k, min_samples=min_n).over(by).alias(name)).select(by, date, name))


def asof(target: pl.DataFrame, table: pl.DataFrame, by_t: str, by_f: str, name: str, date="gameday") -> pl.DataFrame:
    """Attach the latest `name` of `table` from a game strictly before the target's game date."""
    t = table.rename({by_f: by_t}).drop_nulls(name).sort(date)
    return target.sort(date).join_asof(t, on=date, by=by_t, strategy="backward", allow_exact_matches=False)


def base_rows() -> pl.DataFrame:
    p = pl.read_parquet(PRED).filter((pl.col("method") == "vol_x_eff") & (pl.col("kind") == "player") & pl.col("market").is_in(MARKETS)
                                     & pl.col("prediction").is_not_null() & pl.col("actual").is_not_null() & (pl.col("season") <= LAST))
    assert p["season"].max() <= LAST
    con = _con()
    ps = con.execute(f"SELECT DISTINCT player_id, season, week, game_id, team, opponent_team AS opponent FROM player_stats WHERE season_type = 'REG' AND season BETWEEN 2018 AND {LAST}").pl()
    sch = con.execute(f"""SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, season, week, gameday, weekday, gametime, home_team, away_team, home_rest, away_rest,
                          div_game, referee FROM schedules WHERE season BETWEEN 2018 AND {LAST} AND game_type = 'REG' ORDER BY game_id, pulled_at DESC)""").pl()
    con.close()
    sch = sch.with_columns(gameday=pl.col("gameday").str.to_date())
    out = (p.rename({"entity": "player_id"}).drop("gameday").join(ps, on=["player_id", "season", "week"], how="inner").join(sch.drop("season", "week"), on="game_id", how="inner"))
    out = out.with_columns(resid=pl.col("actual") - pl.col("prediction"), home=pl.col("team") == pl.col("home_team"),
                           cluster=pl.col("season") * 100 + pl.col("week"))
    return out.with_columns(own_rest=pl.when(pl.col("home")).then(pl.col("home_rest")).otherwise(pl.col("away_rest")).cast(pl.Float64),
                            opp_rest=pl.when(pl.col("home")).then(pl.col("away_rest")).otherwise(pl.col("home_rest")).cast(pl.Float64))


def schedule_features(b: pl.DataFrame) -> pl.DataFrame:
    hour = pl.col("gametime").str.slice(0, 2).cast(pl.Int32, strict=False)
    b = b.with_columns(x1_own_rest=pl.col("own_rest"), x2_opp_rest=pl.col("opp_rest"), x3_after_bye=(pl.col("own_rest") >= 13).cast(pl.Float64),
                       x4_primetime=((pl.col("weekday").is_in(["Thursday", "Monday"])) | (hour >= 19)).cast(pl.Float64), x5_divisional=pl.col("div_game").cast(pl.Float64),
                       x6_week=pl.col("week").cast(pl.Float64))
    con = _con()
    plays = con.execute(f"""SELECT game_id, count(*) FILTER (WHERE play_type IN ('pass','run')) AS plays, count(*) FILTER (WHERE penalty = 1) AS pens
                            FROM (SELECT DISTINCT game_id, play_id, play_type, penalty FROM pbp WHERE season BETWEEN 2018 AND {LAST}) GROUP BY game_id""").pl()
    con.close()
    g = (b.select("game_id", "gameday", "referee").unique().join(plays, on="game_id", how="left").drop_nulls("referee").sort("gameday", "game_id"))
    # referee trailing mean over EARLIER games (cumulative, >= 20 games) minus the league's cumulative mean over earlier games
    g = g.with_columns(n_prev=pl.col("plays").cum_count().over("referee") - 1,
                       rp=(pl.col("plays").cum_sum().over("referee") - pl.col("plays")), rn=(pl.col("pens").cum_sum().over("referee") - pl.col("pens")),
                       lp=pl.col("plays").cum_sum() - pl.col("plays"), ln=pl.col("pens").cum_sum() - pl.col("pens"), nl=pl.int_range(pl.len()))
    g = g.with_columns(x7_ref_plays=pl.when(pl.col("n_prev") >= 20).then(pl.col("rp") / pl.col("n_prev") - pl.col("lp") / pl.col("nl")),
                       x8_ref_penalties=pl.when(pl.col("n_prev") >= 20).then(pl.col("rn") / pl.col("n_prev") - pl.col("ln") / pl.col("nl")))
    return b.join(g.select("game_id", "x7_ref_plays", "x8_ref_penalties"), on="game_id", how="left")


def player_features(b: pl.DataFrame) -> pl.DataFrame:
    con = _con()
    st = con.execute(f"""SELECT DISTINCT player_id, team, game_id, season, week, carries, targets, receiving_air_yards, air_yards_share, target_share FROM player_stats
                         WHERE season_type = 'REG' AND season BETWEEN 2018 AND {LAST}""").pl()
    sch = con.execute(f"SELECT DISTINCT ON (game_id) game_id, gameday FROM schedules WHERE season BETWEEN 2018 AND {LAST} ORDER BY game_id, pulled_at DESC").pl()
    st = st.join(sch.with_columns(gameday=pl.col("gameday").str.to_date()), on="game_id")
    tc = st.group_by("game_id", "team").agg(team_carries=pl.col("carries").sum())
    st = st.join(tc, on=["game_id", "team"]).with_columns(carry_share=pl.col("carries") / pl.col("team_carries"), adot=pl.col("receiving_air_yards") / pl.col("targets").replace(0, None))
    for col, nm in (("target_share", "x10_target_share_trend"), ("carry_share", "x16_carry_share_trend")):
        a, c = trailing(st, "player_id", col, 3, 3, "a3"), trailing(st, "player_id", col, 8, 4, "a8")
        tr = a.join(c, on=["player_id", "gameday"]).with_columns((pl.col("a3") - pl.col("a8")).alias(nm)).select("player_id", "gameday", nm)
        b = asof(b, tr, "player_id", "player_id", nm)
    for col, nm in (("air_yards_share", "x11_air_yards_share"), ("adot", "x12_adot")):
        b = asof(b, trailing(st, "player_id", col, 8, 4, nm), "player_id", "player_id", nm)
    # snap share (pfr id -> gsis id)
    ids = duckdb.connect(str(config.DUCKDB_PATH), read_only=True).execute("SELECT DISTINCT gsis_id AS player_id, pfr_id FROM players WHERE pfr_id IS NOT NULL").pl()
    sn = con.execute(f"SELECT DISTINCT pfr_player_id AS pfr_id, game_id, offense_pct FROM snap_counts WHERE game_type = 'REG' AND season BETWEEN 2018 AND {LAST}").pl()
    sn = sn.join(ids, on="pfr_id").join(sch.with_columns(gameday=pl.col("gameday").str.to_date()), on="game_id")
    a, c = trailing(sn, "player_id", "offense_pct", 3, 3, "a3"), trailing(sn, "player_id", "offense_pct", 8, 4, "a8")
    b = asof(b, a.join(c, on=["player_id", "gameday"]).with_columns(x9_snap_trend=pl.col("a3") - pl.col("a8")).select("player_id", "gameday", "x9_snap_trend"), "player_id", "player_id", "x9_snap_trend")
    # luck: actual - expected per game (ff_opportunity)
    ff = con.execute(f"""SELECT DISTINCT player_id, game_id, rec_yards_gained - rec_yards_gained_exp AS rec_luck, rush_yards_gained - rush_yards_gained_exp AS rush_luck
                         FROM ff_opportunity WHERE season::INT BETWEEN 2018 AND {LAST}""").pl().join(sch.with_columns(gameday=pl.col("gameday").str.to_date()), on="game_id")
    b = asof(b, trailing(ff.drop_nulls("rec_luck"), "player_id", "rec_luck", 8, 4, "x13_rec_luck"), "player_id", "player_id", "x13_rec_luck")
    b = asof(b, trailing(ff.drop_nulls("rush_luck"), "player_id", "rush_luck", 8, 4, "x17_rush_luck"), "player_id", "player_id", "x17_rush_luck")
    # nextgen (weekly rows only); the game date comes from the player's schedule row that week
    ng = con.execute(f"""SELECT DISTINCT player_gsis_id AS player_id, season, week, avg_separation, avg_yac_above_expectation FROM nextgen_stats_receiving
                         WHERE week > 0 AND season BETWEEN 2018 AND {LAST}""").pl()
    ng = ng.join(st.select("player_id", "season", "week", "gameday"), on=["player_id", "season", "week"])
    b = asof(b, trailing(ng.drop_nulls("avg_separation"), "player_id", "avg_separation", 8, 4, "x14_separation"), "player_id", "player_id", "x14_separation")
    b = asof(b, trailing(ng.drop_nulls("avg_yac_above_expectation"), "player_id", "avg_yac_above_expectation", 8, 4, "x15_yac_above_exp"), "player_id", "player_id", "x15_yac_above_exp")
    con.close()
    return b


def team_features(b: pl.DataFrame) -> pl.DataFrame:
    con = _con()
    sch = con.execute(f"SELECT DISTINCT ON (game_id) game_id, gameday FROM schedules WHERE season BETWEEN 2018 AND {LAST} ORDER BY game_id, pulled_at DESC").pl().with_columns(gameday=pl.col("gameday").str.to_date())
    pb = con.execute(f"""SELECT game_id, play_id, posteam, defteam, "pass" AS is_pass, rush AS is_rush, qb_dropback, wp, qtr, no_huddle, qb_hit, sack FROM
                         (SELECT DISTINCT game_id, play_id, posteam, defteam, "pass", rush, qb_dropback, wp, qtr, no_huddle, qb_hit, sack FROM pbp
                          WHERE season BETWEEN 2018 AND {LAST} AND play_type IN ('pass','run') AND posteam IS NOT NULL)""").pl()
    pb = pb.filter((pl.col("is_pass") == 1) | (pl.col("is_rush") == 1))
    neutral = pb.filter((pl.col("wp").is_between(0.2, 0.8)) & (pl.col("qtr") <= 3))
    nt = neutral.group_by("game_id", "posteam").agg(neutral_pass=(pl.col("is_pass") == 1).mean())
    al = pb.group_by("game_id", "posteam").agg(all_pass=(pl.col("is_pass") == 1).mean(), nohuddle=pl.col("no_huddle").mean())
    tg = nt.join(al, on=["game_id", "posteam"]).join(sch, on="game_id").with_columns(gap=pl.col("neutral_pass") - pl.col("all_pass")).rename({"posteam": "team"})
    b = asof(b, trailing(tg, "team", "gap", 8, 4, "x18_neutral_pass_tendency"), "team", "team", "x18_neutral_pass_tendency")
    b = asof(b, trailing(tg, "team", "nohuddle", 8, 4, "x19_no_huddle"), "team", "team", "x19_no_huddle")
    pr = pb.filter(pl.col("qb_dropback") == 1).group_by("game_id", "defteam").agg(pressure=((pl.col("qb_hit") == 1) | (pl.col("sack") == 1)).mean()).join(sch, on="game_id").rename({"defteam": "opponent"})
    b = asof(b, trailing(pr, "opponent", "pressure", 8, 4, "x20_opp_pressure"), "opponent", "opponent", "x20_opp_pressure")
    part = con.execute(f"""SELECT p.nflverse_game_id AS game_id, b.defteam, avg(CASE WHEN p.defense_man_zone_type = 'MAN_COVERAGE' THEN 1.0 WHEN p.defense_man_zone_type = 'ZONE_COVERAGE' THEN 0.0 END) AS man,
                                  avg(CASE WHEN p.number_of_pass_rushers >= 5 THEN 1.0 WHEN p.number_of_pass_rushers IS NOT NULL THEN 0.0 END) AS blitz
                           FROM (SELECT DISTINCT nflverse_game_id, play_id, defense_man_zone_type, number_of_pass_rushers FROM participation WHERE substr(nflverse_game_id, 1, 4)::INT BETWEEN 2018 AND {LAST}) p
                           JOIN (SELECT DISTINCT game_id, play_id, defteam FROM pbp WHERE season BETWEEN 2018 AND {LAST} AND "pass" = 1) b ON b.game_id = p.nflverse_game_id AND b.play_id = p.play_id
                           GROUP BY 1, 2""").pl().join(sch, on="game_id").rename({"defteam": "opponent"})
    con.close()
    b = asof(b, trailing(part.drop_nulls("man"), "opponent", "man", 8, 4, "x21a_opp_man_rate"), "opponent", "opponent", "x21a_opp_man_rate")
    b = asof(b, trailing(part.drop_nulls("blitz"), "opponent", "blitz", 8, 4, "x21b_opp_blitz_rate"), "opponent", "opponent", "x21b_opp_blitz_rate")
    return b


def lad_fit(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(intercept, slope) of the median regression y ~ a + b z (plan amendment 1: the models are fit to the median, so least squares' mean-shifting intercept costs absolute error)."""
    from sklearn.linear_model import QuantileRegressor
    m = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(z.reshape(-1, 1), y)
    return np.array([m.intercept_, m.coef_[0]])


def screen(b: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cells, cover = [], []
    for cand, markets in CANDIDATES.items():
        for m in markets:
            d = b.filter(pl.col("market") == m)
            n_all = d.height
            d = d.drop_nulls(cand)
            cov = d.height / n_all if n_all else 0.0
            cover.append(dict(candidate=cand, market=m, rows=d.height, coverage=cov, tested=cov >= MIN_COVERAGE))
            if cov < MIN_COVERAGE:
                continue
            gains, sq, per_season, signs, cl, g_rows, null_gain = [], [], {}, [], [], [], []
            for f in FOLDS:
                tr, te = d.filter(pl.col("season") < f), d.filter(pl.col("season") == f)
                if tr.height < 500 or te.height < 100:
                    continue
                mu, sd = tr[cand].mean(), tr[cand].std()
                if not sd:
                    continue
                z = ((tr[cand] - mu) / sd).to_numpy()
                beta = lad_fit(z, tr["resid"].to_numpy())
                r_hat = beta[0] + beta[1] * ((te[cand] - mu) / sd).to_numpy()
                r = te["resid"].to_numpy()
                a0 = float(np.median(tr["resid"].to_numpy()))                           # intercept-only (median) correction: the null the candidate must beat
                null_gain.append(np.abs(r) - np.abs(r - a0))
                gains.append(np.abs(r - a0) - np.abs(r - r_hat))                       # incremental gain of the slope over the intercept-only correction (plan amendment 2)
                sq.append((r - a0) ** 2 - (r - r_hat) ** 2)
                cl.append(te["cluster"].to_numpy())
                per_season[f] = float(gains[-1].mean())
                signs.append(np.sign(beta[1]))
            if not gains:
                continue
            g, s, c = np.concatenate(gains), np.concatenate(sq), np.concatenate(cl)
            cells.append(dict(candidate=cand, market=m, n=len(g), gain=float(g.mean()), intercept_only_gain=float(np.concatenate(null_gain).mean()), sq_gain=float(s.mean()), seasons_won=int(sum(v > 0 for v in per_season.values())),
                              sign_stable=bool(len(set(signs)) == 1), slope_sign=float(signs[-1]), per_season=str({k: round(v, 5) for k, v in per_season.items()}),
                              mean_abs_resid=float(np.abs(np.concatenate([x for x in [d.filter(pl.col("season") >= 2021)["resid"].to_numpy()]])).mean()),
                              _g=g, _c=c))
    out = pl.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")} for c in cells])
    m_tested = len(cells)
    level = 1 - 0.05 / m_tested
    rows = []
    for c, base in zip(cells, out.iter_rows(named=True)):
        lo, hi = compare.cluster_bootstrap(c["_g"], c["_c"], level=level)
        v = compare.verdict(base["gain"], lo, hi, base["seasons_won"])
        rows.append(dict(ci_lo=lo, ci_hi=hi, verdict="CLEARS" if (v == "CLEARS" and base["sign_stable"]) else ("EDGE" if v == "CLEARS" else v)))
    out = pl.concat([out, pl.DataFrame(rows)], how="horizontal").with_columns(m_tested=pl.lit(m_tested), level=pl.lit(level))
    return out, pl.DataFrame(cover)


def main():
    b = base_rows()
    assert b["season"].max() <= LAST
    for fn in (schedule_features, player_features, team_features):
        b = fn(b)
    cells, cover = screen(b)
    cells.write_parquet(config.PROCESSED_DIR / "residual_screen_cells.parquet")
    cover.write_parquet(config.PROCESSED_DIR / "residual_screen_coverage.parquet")
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(60)
    print("rows", b.height, "cells tested", cells.height, "bonferroni level", float(cells["level"][0]))
    print(cover.filter(~pl.col("tested")))
    print(cells.sort("gain", descending=True).select("candidate", "market", "n", pl.col("gain").round(5), pl.col("sq_gain").round(5), "seasons_won", "sign_stable", pl.col("ci_lo").round(5), pl.col("ci_hi").round(5), "verdict"))
    print(cells.group_by("verdict").len())


if __name__ == "__main__":
    main()
