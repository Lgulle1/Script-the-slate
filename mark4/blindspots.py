"""Mark 4, step 2: where is FanDuel's line systematically wrong? (2026-10-19)

Every FanDuel line 2021-2026 (the line study's pre-kickoff lines, both sides with their real prices) is tagged with the situation it was set in, using only
what was known before kickoff. For each tag x market group x side: how often the bet won, the return per unit at the real price, per season, and
RECENCY-WEIGHTED (a bet's weight halves every HALF_LIFE_DAYS back from the latest game, so 2025-26 count most). Because many tags are tried, each
return is also SHRUNK toward the base rate of that side and group: (sum w*profit + K*base) / (sum w + K), so a tag needs real volume to stand out.

  python -m mark4.blindspots          # writes records/lab/blindspots.parquet and .json
"""
from __future__ import annotations

import json

import duckdb
import numpy as np
import polars as pl

import config
from live import history as H

HALF_LIFE_DAYS = 365.0
K = 200.0
OUT = config.ROOT / "records" / "lab" / "blindspots"
GROUPS = {"rushing": ("rush_att", "rush_yds"), "receiving": ("rec", "rec_yds"), "passing": ("pass_att", "pass_cmp", "pass_yds"),
          "all": ("rush_att", "rush_yds", "rec", "rec_yds", "pass_att", "pass_cmp", "pass_yds")}
STAT = dict(rush_att="carries", rush_yds="rushing_yards", rec="receptions", rec_yds="receiving_yards", pass_att="attempts", pass_cmp="completions",
            pass_yds="passing_yards")


def _american_to_prob(a):
    return pl.when(a > 0).then(100 / (a + 100)).otherwise(-a / (-a + 100))


def frame() -> pl.DataFrame:
    sel = pl.read_parquet(config.PROCESSED_DIR / "line_study_sel.parquet").filter(pl.col("tier") != "late")
    sel = sel.unique(["game_id", "player_id", "market", "side"], keep="first").select(
        "season", "week", "game_id", "player_id", "market", "line", "side", "price", "open_price", "dec", "won", "profit", "actual", "mean")
    log = H.read_played_logs(2026)[0]
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sc = con.execute("""SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, gameday, weekday, gametime, home_team, away_team, home_rest, away_rest, spread_line,
                        total_line, div_game, roof, temp, wind, home_qb_id, away_qb_id, home_coach, away_coach, season FROM schedules WHERE game_type = 'REG'
                        ORDER BY game_id, pulled_at DESC)""").pl()
    con.close()
    sc = sc.with_columns(pl.col("gameday").cast(pl.Utf8).str.slice(0, 10).str.to_date(strict=False), season=pl.col("season").cast(pl.Int32))
    # per team-game: qb, coach, rest, spread from the team's side
    side = []
    for t in ("home", "away"):
        o = "away" if t == "home" else "home"
        side.append(sc.select("game_id", "gameday", "season", "weekday", "gametime", "total_line", "div_game", "roof", "temp", "wind",
                              team=pl.col(f"{t}_team"), is_home=pl.lit(t == "home"), rest=pl.col(f"{t}_rest"), opp_rest=pl.col(f"{o}_rest"),
                              team_spread=(pl.col("spread_line") if t == "home" else -pl.col("spread_line")), qb=pl.col(f"{t}_qb_id"), coach=pl.col(f"{t}_coach")))
    tg = pl.concat(side).sort("team", "gameday")
    tg = tg.with_columns(qb_change=(pl.col("qb") != pl.col("qb").shift(1).over("team")).fill_null(False),
                         prev_season_coach=pl.col("coach").filter(pl.col("season") < pl.col("season").max()).last().over("team"))
    # first-year head coach: the coach differs from the team's coach in its last game of the previous season
    last = tg.group_by("team", "season").agg(last_coach=pl.col("coach").sort_by("gameday").last()).with_columns(season=pl.col("season") + 1)
    tg = tg.join(last, on=["team", "season"], how="left").with_columns(new_coach=(pl.col("coach") != pl.col("last_coach")).fill_null(False)).drop("prev_season_coach")
    who = log.select("game_id", "player_id", "team", "family", "gameday", *set(STAT.values())).unique(["game_id", "player_id"])
    d = sel.join(who, on=["game_id", "player_id"], how="inner").join(tg.drop("gameday", "season"), on=["game_id", "team"], how="left")
    # recent form: the player's last 4 games of that stat before this game
    form = []
    for m, st in STAT.items():
        f = who.sort("player_id", "gameday").with_columns(prev4=pl.col(st).cast(pl.Float64).shift(1).rolling_mean(4, min_samples=2).over("player_id")).select(
            "game_id", "player_id", "prev4").with_columns(market=pl.lit(m))
        form.append(f)
    d = d.join(pl.concat(form), on=["game_id", "player_id", "market"], how="left")
    # streak against the line: how the player's last 3 lines in this market finished (over count)
    ov = sel.filter(pl.col("side") == "over").join(who.select("game_id", "player_id", "gameday"), on=["game_id", "player_id"]).sort("player_id", "market", "gameday")
    ov = ov.with_columns(o=(pl.col("actual") > pl.col("line")).cast(pl.Float64))
    ov = ov.with_columns(last3_overs=pl.col("o").shift(1).rolling_sum(3).over("player_id", "market")).select("game_id", "player_id", "market", "last3_overs")
    d = d.join(ov, on=["game_id", "player_id", "market"], how="left")
    sit = pl.read_parquet(config.PROCESSED_DIR / "comps2_situations.parquet").select("game_id", "player_id", "vac_car", "vac_tgt", "own_car", "own_tgt",
                                                                                       "games_since_last", "status", "hurt_last")
    d = d.join(sit, on=["game_id", "player_id"], how="left")
    lc = config.PROCESSED_DIR / "comps2_line_components.parquet"
    if lc.exists():
        d = d.join(pl.read_parquet(lc).select("game_id", "player_id", "market", dfn_line="dfn_shift"), on=["game_id", "player_id", "market"], how="left")
    over_open = _american_to_prob(pl.col("open_price"))
    over_now = _american_to_prob(pl.col("price"))
    # price movement of the OVER (on under rows the price is the under's, so flip)
    d = d.with_columns(move_over=pl.when(pl.col("side") == "over").then(over_now - over_open).otherwise(over_open - over_now))
    latest = d["gameday"].max()
    d = d.with_columns(w=0.5 ** ((pl.lit(latest) - pl.col("gameday")).dt.total_days() / HALF_LIFE_DAYS))
    return d.filter(pl.col("profit").is_not_null())


def tags() -> dict:
    c = pl.col
    prime = (c("weekday") != "Sunday") | (c("gametime") >= "19:00")
    q = lambda x: c("dfn_line").quantile(x)
    return {
        "every line": pl.lit(True),
        "backup promoted: a back holding 25%+ of carries is out, his own share under 35%": (c("vac_car") >= 0.25) & (c("own_car") < 0.35) & (c("family") == "RB"),
        "receiver promoted: 15%+ of targets out, his own share under 18%": (c("vac_tgt") >= 0.15) & (c("own_tgt") < 0.18),
        "a big teammate is out (25%+ carries or 15%+ targets)": (c("vac_car") >= 0.25) | (c("vac_tgt") >= 0.15),
        "questionable / limited this week": c("status") == 1,
        "doubtful or did not practice, played anyway": c("status") == 2,
        "played hurt last game": c("hurt_last") == 1,
        "back from missing a game": c("games_since_last") >= 2,
        "QB change this week (team)": c("qb_change"),
        "first-year head coach": c("new_coach"),
        "short week (5 days or less)": c("rest") <= 5,
        "off a bye (10+ days)": c("rest") >= 10,
        "primetime / non-Sunday": prime,
        "divisional game": c("div_game") == 1,
        "home": c("is_home"), "away": ~c("is_home"),
        "dome / closed roof": c("roof").is_in(["dome", "closed"]),
        "wind 15+ mph (outdoors)": (c("wind") >= 15) & ~c("roof").is_in(["dome", "closed"]),
        "cold: 35F or less": (c("temp") <= 35) & ~c("roof").is_in(["dome", "closed"]),
        "team favored by 7+": c("team_spread") >= 7, "team underdog by 7+": c("team_spread") <= -7,
        "game total 49+": c("total_line") >= 49, "game total 40 or less": c("total_line") <= 40,
        "line 15%+ above his last-4 average": c("line") >= 1.15 * c("prev4"), "line 15%+ below his last-4 average": c("line") <= 0.85 * c("prev4"),
        "went over his last 3 lines": c("last3_overs") == 3, "went under his last 3 lines": c("last3_overs") == 0,
        "price moved toward the over (3+ pts)": c("move_over") >= 0.03, "price moved toward the under (3+ pts)": c("move_over") <= -0.03,
        "weeks 1-3": c("week") <= 3, "weeks 15+": c("week") >= 15,
        "defense has been beaten vs the line lately (top fifth)": c("dfn_line") >= q(0.8),
        "defense has been holding players under the line lately (bottom fifth)": c("dfn_line") <= q(0.2),
    }


def table(d: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for gname, mk in GROUPS.items():
        g = d.filter(pl.col("market").is_in(list(mk)))
        base = {s: float((x["w"] * x["profit"]).sum() / x["w"].sum()) for (s,), x in g.group_by("side")}
        for tname, expr in tags().items():
            t = g.filter(expr.fill_null(False))
            for (side,), x in t.group_by("side"):
                if x.height < 30:
                    continue
                sw = float(x["w"].sum())
                wroi = float((x["w"] * x["profit"]).sum() / sw)
                per = {int(s): (float(y["profit"].mean()), y.height) for (s,), y in x.group_by("season")}
                rows.append(dict(group=gname, situation=tname, side=side, n=x.height, n_recent=int(x.filter(pl.col("season") >= 2025).height),
                                 hit=float(x["won"].mean()), roi=float(x["profit"].mean()), roi_recency=wroi,
                                 roi_shrunk=float(((x["w"] * x["profit"]).sum() + K * base[side]) / (sw + K)),
                                 roi_2025_26=float(x.filter(pl.col("season") >= 2025)["profit"].mean()) if x.filter(pl.col("season") >= 2025).height else None,
                                 seasons_up=sum(1 for v in per.values() if v[0] > 0 and v[1] >= 10), seasons=sum(1 for v in per.values() if v[1] >= 10),
                                 by_season={str(k): [round(v[0], 4), v[1]] for k, v in sorted(per.items())}, base_recency=base[side]))
    return pl.DataFrame(rows).sort("roi_shrunk", descending=True)


if __name__ == "__main__":
    d = frame()
    print("bets:", d.height, d.group_by("season").len().sort("season").rows())
    t = table(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(OUT.with_suffix(".parquet"))
    OUT.with_suffix(".json").write_text(json.dumps(t.to_dicts(), indent=1, default=str))
    pl.Config.set_tbl_rows(40)
    pl.Config.set_fmt_str_lengths(60)
    print(t.filter(pl.col("group") != "all").select("group", "situation", "side", "n", "hit", "roi", "roi_recency", "roi_shrunk", "roi_2025_26", "seasons_up", "seasons").head(30))
