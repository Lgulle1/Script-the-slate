"""Game state (Phase 4b): expected margin and total with a starting-QB adjustment (4b.2); state shares, dropback rate by
state and play accounting (4b.3, further down).

4b.2 -- expected points, margin and total
-----------------------------------------
For a team A playing B, everything known before the game (ratings from features/team_ratings.py for that week; history
before the week's first kickoff):

    x_rating = expected plays_A * [ pass_rate_A * (off_pass_A + def_pass_B) + (1 - pass_rate_A) * (off_rush_A + def_rush_B) ]
               (the EPA over a game that the two teams' ratings imply, relative to an average matchup)
    x_home   = +0.5 home, -0.5 away, 0 at a neutral site
    x_rest   = A's days of rest - B's (from the schedule, clipped to +-REST_CLIP)
    x_qb     = (expected starter's EPA per dropback - EPA per dropback of the QBs the ratings were built on) * expected dropbacks_A

    points_A = league average points + c + b * x_rating + h * x_home + r * x_rest + beta * x_qb

The scaling terms (c, b, h, r, beta) are fit by least squares on every team-game BEFORE the target week (walk-forward,
expanding window; at least MIN_FIT_ROWS rows, else no prediction). margin = points_home - points_away, total = the sum.
"league average points" is the recency-weighted mean points per team-game before the week.

  expected plays_A   0.5 * (A's recency-weighted plays per game + B's recency-weighted plays allowed per game)
  pass_rate_A        A's recency-weighted dropbacks / plays          (recency: features/weights.py half-life, offseason gap)

Starting QB (item 2). Each QB's EPA per dropback = (his EPA on all dropbacks before the week + QB_K * league mean) /
(his dropbacks + QB_K), k = 200 dropbacks. The QBs the ratings were built on = the team's dropbacks this season before the
week and last season, weighted exactly like the ratings (recency within season; current vs last season by n / (n + 6)).
The expected starter is the week's depth-chart QB1, unless he is likely out: P(QB1 does not start) = 1 for Out / Doubtful /
IR-PUP-suspended, else the 4a.1 status model's absence probability (2020+; 0 for the 2019 history season). The starter
quality is then the probability-weighted blend of QB1 and the chart's QB2. beta is the fitted points per EPA; reported as
points per 0.1 EPA/dropback over EXAMPLE_DROPBACKS dropbacks.

Uncertainty (item 4): the SD of the margin and of the total for week W are the root mean squared walk-forward errors of all
earlier predicted games (at least MIN_SD_GAMES), not a constant.

NO MARKET LINE ANYWHERE (item 3): the schedule loader selects SCHEDULE_COLUMNS explicitly; spread_line, total_line, the
moneylines and odds (team_ratings.FORBIDDEN_COLUMNS) are never read, and the schedules' actual starting-QB columns
(home_qb_id / away_qb_id, known only after kickoff) are not read either.

History: 2019 games get pre-game features and predictions (ratings start in 2018) so that 2020 week 1 already has fitted
scaling terms and prior errors; 2019 is history only, never scored. 2025+ is never loaded.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import polars as pl

import config
from features import team_ratings as tr
from features.weights import recency_weight

HISTORY_START = tr.HISTORY_START
FIRST_FEATURE_SEASON = HISTORY_START + 1          # 2019: first season with pre-game features (history)
QB_K = 200.0
MIN_FIT_ROWS = 256
MIN_SD_GAMES = 100
REST_CLIP = 10.0
EXAMPLE_DROPBACKS = 35.0
FEATURES = ("x_rating", "x_home", "x_rest", "x_qb")
SCHEDULE_COLUMNS = ("game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score", "location",
                    "home_rest", "away_rest")
BLOCKED = ("RES", "PUP", "SUS", "NWT", "RSN", "EXE")
# The play-by-play already uses today's abbreviations for every season; the schedules and depth charts still say OAK for 2018-2019.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def _team(col: str) -> pl.Expr:
    return pl.col(col).replace(TEAM_ALIASES)


# ====================================================================== inputs
@dataclass
class Inputs:
    games: pl.DataFrame        # SCHEDULE_COLUMNS, regular season
    plays: pl.DataFrame        # team_ratings.load_play_rows
    qb: pl.DataFrame           # season, week, game_id, team, qb_id, dropbacks, epa
    chart: pl.DataFrame        # season, week, team, qb1, qb2 (pre-game depth chart)
    status: pl.DataFrame       # season, week, gsis_id, report_status, practice_status, date_modified (UTC)
    roster: pl.DataFrame       # season, week, gsis_id, roster_status


def _cap(max_season):
    return config.cap_season(max_season)       # raises config.HoldoutError for 2025+


def load_games(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    cap = _cap(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        g = con.execute(
            f"SELECT {', '.join(SCHEDULE_COLUMNS)} FROM (SELECT DISTINCT ON (game_id) * FROM schedules ORDER BY game_id, pulled_at DESC) "
            f"WHERE game_type = 'REG' AND season >= {HISTORY_START} AND season <= {cap}").pl()
    finally:
        con.close()
    return g.with_columns(pl.col("gameday").str.to_date(), home_team=_team("home_team"), away_team=_team("away_team")).sort("season", "week", "game_id")


def load_qb_dropbacks(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """EPA on every QB dropback (attempts, sacks, scrambles) per (game, team, QB)."""
    cap = _cap(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        con.execute("SET threads TO 1")          # a fixed summation order: floating-point sums are identical run to run
        d = con.execute(
            "SELECT season, week, game_id, posteam AS team, coalesce(passer_player_id, rusher_player_id) AS qb_id, "
            "CAST(count(*) AS DOUBLE) AS dropbacks, sum(epa) AS epa "
            "FROM (SELECT DISTINCT ON (game_id, play_id) * FROM pbp ORDER BY game_id, play_id, pulled_at DESC) "
            f"WHERE season_type = 'REG' AND season >= {HISTORY_START} AND season <= {cap} AND qb_dropback = 1 AND epa IS NOT NULL "
            "AND coalesce(two_point_attempt, 0) = 0 AND posteam IS NOT NULL AND coalesce(passer_player_id, rusher_player_id) IS NOT NULL "
            "GROUP BY ALL").pl()
    finally:
        con.close()
    return d.sort("season", "week", "game_id", "team", "qb_id")


def load_qb_chart(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> pl.DataFrame:
    """The week's pre-game depth chart at QB: qb1, qb2 per (season, week, team) (ties on depth_team broken by gsis_id)."""
    cap = _cap(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute(
            "SELECT season, week, club_code AS team, gsis_id, depth_team FROM (SELECT DISTINCT ON (season, week, club_code, gsis_id, depth_team, position) * "
            f"FROM depth_charts WHERE season >= {HISTORY_START} AND season <= {cap} ORDER BY season, week, club_code, gsis_id, depth_team, position, pulled_at DESC) "
            "WHERE position = 'QB' AND formation = 'Offense' AND game_type = 'REG' AND gsis_id IS NOT NULL").pl()
    finally:
        con.close()
    d = d.with_columns(dt=pl.col("depth_team").cast(pl.Int32, strict=False), team=_team("team")).drop_nulls("dt")
    d = d.group_by("season", "week", "team", "gsis_id").agg(pl.col("dt").min()).sort("season", "week", "team", "dt", "gsis_id")
    d = d.with_columns(rank=pl.int_range(pl.len()).over("season", "week", "team"))
    q1 = d.filter(pl.col("rank") == 0).select("season", "week", "team", qb1="gsis_id")
    q2 = d.filter(pl.col("rank") == 1).select("season", "week", "team", qb2="gsis_id")
    return q1.join(q2, on=["season", "week", "team"], how="left")


def load_status(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(injury rows, roster statuses) for HISTORY_START..cap (the QB starter rule needs 2019 too)."""
    cap = _cap(max_season)
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        inj = con.execute(
            "SELECT CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS week, gsis_id, report_status, practice_status, date_modified "
            "FROM (SELECT DISTINCT ON (season, week, team, gsis_id) * FROM injuries ORDER BY season, week, team, gsis_id, pulled_at DESC) "
            f"WHERE game_type = 'REG' AND gsis_id IS NOT NULL AND season >= {HISTORY_START} AND season <= {cap}").pl()
        ros = con.execute(
            "SELECT season, week, gsis_id, status AS roster_status FROM (SELECT DISTINCT ON (season, week, team, gsis_id) * FROM rosters_weekly "
            f"ORDER BY season, week, team, gsis_id, pulled_at DESC) WHERE game_type = 'REG' AND gsis_id IS NOT NULL "
            f"AND season >= {HISTORY_START} AND season <= {cap}").pl()
    finally:
        con.close()
    return inj.with_columns(pl.col("date_modified").dt.convert_time_zone("UTC")), ros


def load_inputs(raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> Inputs:
    inj, ros = load_status(raw_db, max_season)
    return Inputs(load_games(raw_db, max_season), tr.load_play_rows(raw_db, max_season), load_qb_dropbacks(raw_db, max_season),
                  load_qb_chart(raw_db, max_season), inj, ros)


# ====================================================================== clock and history helpers
def _clock(games: pl.DataFrame) -> dict:
    """(season, week) -> position on a week clock where an offseason adds config.OFFSEASON_GAP_GAMES (as features/weights.py)."""
    weeks = dict(games.group_by("season").agg(pl.col("week").max()).iter_rows())
    base, pos = 0.0, {}
    for s in sorted(weeks):
        for w in range(1, weeks[s] + 2):
            pos[(s, w)] = base + w
        base += weeks[s] + config.OFFSEASON_GAP_GAMES
    return pos


def team_games(inp: Inputs) -> pl.DataFrame:
    """One row per team-game: points for/against, plays, dropbacks, the opponent's plays (= plays allowed), home flag, rest."""
    g = inp.games
    home = g.select("game_id", "season", "week", "gameday", team="home_team", opp="away_team", pf="home_score", pa="away_score",
                    rest="home_rest", opp_rest="away_rest", x_home=pl.when(pl.col("location") == "Neutral").then(0.0).otherwise(0.5))
    away = g.select("game_id", "season", "week", "gameday", team="away_team", opp="home_team", pf="away_score", pa="home_score",
                    rest="away_rest", opp_rest="home_rest", x_home=pl.when(pl.col("location") == "Neutral").then(0.0).otherwise(-0.5))
    tg = pl.concat([home, away])
    p = (inp.plays.group_by("game_id", "posteam").agg(plays=pl.col("n").sum(), dropbacks=pl.col("n").filter(pl.col("type") == "pass").sum())
         .rename({"posteam": "team"}))
    tg = tg.join(p, on=["game_id", "team"], how="left")
    tg = tg.join(p.rename({"team": "opp", "plays": "opp_plays", "dropbacks": "opp_dropbacks"}), on=["game_id", "opp"], how="left")
    return tg.with_columns(x_rest=(pl.col("rest") - pl.col("opp_rest")).cast(pl.Float64).clip(-REST_CLIP, REST_CLIP)).sort("season", "week", "game_id", "team")


# ====================================================================== QB quality
class QBBook:
    """EPA per dropback per QB as of any (season, week): shrunk toward the league mean with n / (n + QB_K)."""

    def __init__(self, qb: pl.DataFrame):
        a = (qb.sort("season", "week", "qb_id", "game_id").group_by("qb_id", "season", "week", maintain_order=True)
             .agg(n=pl.col("dropbacks").sum(), e=pl.col("epa").sum()).sort("season", "week", "qb_id"))
        self.key_of = lambda s, w: s * 100 + w
        self.by_qb = {}
        for (qid,), g in a.partition_by("qb_id", as_dict=True).items():
            k = (g["season"] * 100 + g["week"]).to_numpy()
            self.by_qb[qid] = (k, np.cumsum(g["n"].to_numpy()), np.cumsum(g["e"].to_numpy()))
        lg = a.group_by("season", "week", maintain_order=True).agg(n=pl.col("n").sum(), e=pl.col("e").sum()).sort("season", "week")
        self.lg = ((lg["season"] * 100 + lg["week"]).to_numpy(), np.cumsum(lg["n"].to_numpy()), np.cumsum(lg["e"].to_numpy()))

    def league_mean(self, season, week) -> float:
        k, n, e = self.lg
        i = int(np.searchsorted(k, self.key_of(season, week), "left"))
        return float(e[i - 1] / n[i - 1]) if i else 0.0

    def quality(self, qb_id, season, week) -> float:
        """(his EPA on dropbacks before the week + QB_K * league mean) / (his dropbacks + QB_K)."""
        L = self.league_mean(season, week)
        rec = self.by_qb.get(qb_id)
        if qb_id is None or rec is None:
            return L
        k, n, e = rec
        i = int(np.searchsorted(k, self.key_of(season, week), "left"))
        if not i:
            return L
        return float((e[i - 1] + QB_K * L) / (n[i - 1] + QB_K))


def built_on_quality(book: QBBook, qb: pl.DataFrame, team: str, season: int, week: int, last_week: dict, n_games: int) -> float:
    """EPA per dropback of the QBs the team's ratings were built on: this season's dropbacks before the week (recency weights
    as in the ratings) blended with last season's (weights relative to its end) by n / (n + PRIOR_GAMES)."""
    def mix(rows, target_week):
        if not rows.height:
            return None
        w = np.array([recency_weight(float(target_week - x)) for x in rows["week"].to_list()]) * rows["dropbacks"].to_numpy()
        q = np.array([book.quality(i, season, week) for i in rows["qb_id"].to_list()])
        return float((w * q).sum() / w.sum()) if w.sum() > 0 else None

    cur = mix(qb.filter((pl.col("team") == team) & (pl.col("season") == season) & (pl.col("week") < week)), week)
    prev_rows = qb.filter((pl.col("team") == team) & (pl.col("season") == season - 1))
    prev = mix(prev_rows, last_week.get(season - 1, 0) + 1) if prev_rows.height else None
    wt = n_games / (n_games + tr.PRIOR_GAMES)
    if cur is None and prev is None:
        return book.league_mean(season, week)
    if cur is None:
        return prev
    if prev is None:
        return cur
    return wt * cur + (1 - wt) * prev


# ====================================================================== pre-game features
def _status_q(report: str, blocked: bool) -> float | None:
    """The rule part of P(QB1 does not start): 1 for Out / Doubtful / IR-PUP-suspended, None = ask the status model."""
    if blocked or report in ("Out", "Doubtful"):
        return 1.0
    return None


def pregame_features(inp: Inputs, ratings: pl.DataFrame, status_models: dict | None = None) -> pl.DataFrame:
    """One row per team-game of FIRST_FEATURE_SEASON..cap with every pre-game input and the outcome columns (pf, pa).

    status_models: {(season, week): models.injuries.StatusModel fit before that week} for the probabilistic absence of a
    Questionable QB1 (missing weeks -> only the Out / Doubtful / IR rule)."""
    from models import injuries as ij

    tg = team_games(inp)
    clock = _clock(inp.games)
    book = QBBook(inp.qb)
    last_week = dict(inp.games.group_by("season").agg(pl.col("week").max()).iter_rows())
    status_fn = ij.make_status_lookup(inp.status.with_columns(team=pl.lit("")))
    blocked = {k[:2]: set(g["gsis_id"].to_list()) for k, g in
               inp.roster.filter(pl.col("roster_status").is_in(list(BLOCKED))).partition_by("season", "week", as_dict=True).items()}
    chart = {(r["season"], r["week"], r["team"]): (r["qb1"], r["qb2"]) for r in inp.chart.iter_rows(named=True)}
    # every ratings row is "the ratings in force before that week"; a season's final row (last week + 1) is also what a live
    # prediction of the next, not yet played week reads, so it is kept
    rmap = {(r["season"], r["week"], r["team"]): r for r in ratings.iter_rows(named=True)}
    hist = {}
    for (team,), g in tg.partition_by("team", as_dict=True).items():
        hist[team] = g.sort("season", "week")
    league = tg.sort("season", "week")
    lpos = np.array([clock[(s, w)] for s, w in zip(league["season"].to_list(), league["week"].to_list())])
    lpts = league["pf"].to_numpy().astype(float)
    rows = []
    targets = tg.filter(pl.col("season") >= FIRST_FEATURE_SEASON)
    for r in targets.iter_rows(named=True):
        s, w, team, opp = r["season"], r["week"], r["team"], r["opp"]
        here = clock[(s, w)]
        # recency-weighted history (all earlier games of the team / of the opponent / of the league)
        def wmean(h, cols):
            past = h.filter((pl.col("season") < s) | ((pl.col("season") == s) & (pl.col("week") < w)))
            if not past.height:
                return [None] * len(cols), 0
            pos = np.array([clock[(a, b)] for a, b in zip(past["season"].to_list(), past["week"].to_list())])
            wt = 0.5 ** ((here - pos) / config.RECENCY_HALF_LIFE_GAMES)
            return [float((wt * past[c].to_numpy().astype(float)).sum() / wt.sum()) for c in cols], past.height
        (pl_a, db_a), n_a = wmean(hist[team], ["plays", "dropbacks"])
        (pa_b,), _ = wmean(hist[opp], ["opp_plays"])
        m = lpos < here
        wt = 0.5 ** ((here - lpos[m]) / config.RECENCY_HALF_LIFE_GAMES)
        league_pts = float((wt * lpts[m]).sum() / wt.sum()) if m.any() else None
        ra, rb = rmap.get((s, w, team)), rmap.get((s, w, opp))
        rec = dict(season=s, week=w, game_id=r["game_id"], gameday=r["gameday"], team=team, opp=opp, pf=r["pf"], pa=r["pa"],
                   x_home=r["x_home"], x_rest=r["x_rest"], league_pts=league_pts)
        if pl_a is None or pa_b is None or ra is None or rb is None or db_a is None:
            rows.append(rec)
            continue
        exp_plays = 0.5 * (pl_a + pa_b)
        pass_rate = db_a / pl_a if pl_a else 0.0
        x_rating = exp_plays * (pass_rate * (ra["off_pass"] + rb["def_pass"]) + (1 - pass_rate) * (ra["off_rush"] + rb["def_rush"]))
        # starting QB
        qb1, qb2 = chart.get((s, w, team), (None, None))
        n_games = ra["n_games"]
        built = built_on_quality(book, inp.qb, team, s, w, last_week, n_games)
        q_out = 0.0
        if qb1 is not None:
            report, practice = status_fn(qb1, s, w, ij.main_run_as_of(r["gameday"]))
            rule = _status_q(report, qb1 in blocked.get((s, w), set()))
            if rule is not None:
                q_out = rule
            elif status_models and (s, w) in status_models:
                q_out, _ = ij.absence_inputs(status_models[(s, w)], "QB", report, practice)
            starter = (1 - q_out) * book.quality(qb1, s, w) + q_out * book.quality(qb2, s, w)
        else:
            starter = built
        exp_db = exp_plays * pass_rate
        rec.update(exp_plays=exp_plays, pass_rate=pass_rate, exp_dropbacks=exp_db, x_rating=x_rating, qb1=qb1, qb2=qb2, p_qb1_out=q_out,
                   qb_starter_epa=starter, qb_built_on_epa=built, x_qb=(starter - built) * exp_db)
        rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None).sort("season", "week", "game_id", "team")


# ====================================================================== walk-forward fit and prediction
def _design(df: pl.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(df.height)] + [df[c].to_numpy().astype(float) for c in FEATURES])


def walk_forward(features: pl.DataFrame, min_fit_rows: int = MIN_FIT_ROWS, min_sd_games: int = MIN_SD_GAMES) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(team-game predictions, per-week coefficients). For each (season, week) in order: fit the scaling terms on the earlier
    team-games, predict this week's points; margin / total SDs from the earlier games' errors."""
    f = features.with_columns(y=(pl.col("pf") - pl.col("league_pts")).cast(pl.Float64))
    ok = f.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in (*FEATURES, "league_pts")]))
    weeks = f.select("season", "week").unique().sort("season", "week").rows()
    preds, coefs = [], []
    ok_key = (ok["season"] * 100 + ok["week"]).to_numpy()
    X_all, y_all = _design(ok), ok["y"].to_numpy().astype(float)
    have_y = ~np.isnan(y_all)
    for s, w in weeks:
        key = s * 100 + w
        tr_m = (ok_key < key) & have_y
        te = ok.filter((pl.col("season") == s) & (pl.col("week") == w))
        if tr_m.sum() < min_fit_rows or not te.height:
            continue
        beta, *_ = np.linalg.lstsq(X_all[tr_m], y_all[tr_m], rcond=None)
        coefs.append(dict(season=s, week=w, n_fit=int(tr_m.sum()), **{n: float(b) for n, b in zip(("c", *FEATURES), beta)}))
        p = te.with_columns(pts_mean=pl.col("league_pts") + pl.Series(_design(te) @ beta))
        preds.append(p.drop("y"))
    return pl.concat(preds), pl.DataFrame(coefs)


def side_points(side: dict, coef: dict) -> float:
    """Expected points for one side: league average + c + sum(coefficient * feature) (the same formula walk_forward uses)."""
    return side["league_pts"] + coef["c"] + sum(coef[f] * side[f] for f in FEATURES)


def game_prediction(home: dict, away: dict, coef: dict) -> tuple[float, float]:
    """(margin = home - away, total) from the two sides' pre-game features."""
    h, a = side_points(home, coef), side_points(away, coef)
    return h - a, h + a


def game_expectations(team_preds: pl.DataFrame, games: pl.DataFrame, min_sd_games: int = MIN_SD_GAMES) -> pl.DataFrame:
    """One row per game: expected home / away points, margin (home - away) and total, with SDs = RMS walk-forward errors of
    all earlier predicted games (null until MIN_SD_GAMES exist)."""
    h = team_preds.filter(pl.col("x_home") >= 0).join(games.select("game_id", "home_team"), on="game_id").filter(pl.col("team") == pl.col("home_team"))
    a = team_preds.join(games.select("game_id", "away_team"), on="game_id").filter(pl.col("team") == pl.col("away_team"))
    cols = ["pts_mean", "pf", "x_rating", "x_qb", "qb1", "p_qb1_out", "qb_starter_epa", "qb_built_on_epa", "exp_plays", "pass_rate", "exp_dropbacks"]
    g = (h.select("season", "week", "game_id", "gameday", home_team="team", neutral=(pl.col("x_home") == 0), **{f"home_{c}": c for c in cols})
         .join(a.select("game_id", away_team="team", **{f"away_{c}": c for c in cols}), on="game_id", how="inner"))
    g = g.with_columns(margin_mean=pl.col("home_pts_mean") - pl.col("away_pts_mean"), total_mean=pl.col("home_pts_mean") + pl.col("away_pts_mean"),
                       margin=(pl.col("home_pf") - pl.col("away_pf")).cast(pl.Float64), total=(pl.col("home_pf") + pl.col("away_pf")).cast(pl.Float64))
    g = g.sort("season", "week", "game_id")
    key = (g["season"] * 100 + g["week"]).to_numpy()
    em = (g["margin"] - g["margin_mean"]).to_numpy()
    et = (g["total"] - g["total_mean"]).to_numpy()
    msd, tsd, nprior = [], [], []
    for k in key:
        m = (key < k) & ~np.isnan(em)
        n = int(m.sum())
        nprior.append(n)
        msd.append(float(np.sqrt(np.mean(em[m] ** 2))) if n >= min_sd_games else None)
        tsd.append(float(np.sqrt(np.mean(et[m] ** 2))) if n >= min_sd_games else None)
    return g.with_columns(margin_sd=pl.Series(msd, dtype=pl.Float64), total_sd=pl.Series(tsd, dtype=pl.Float64), n_prior_games=pl.Series(nprior),
                          is_history=pl.col("season") < config.FEATURE_HISTORY_START)


def qb_points_per_tenth(coefs: pl.DataFrame) -> pl.DataFrame:
    """beta expressed as points per 0.1 EPA/dropback over EXAMPLE_DROPBACKS dropbacks."""
    return coefs.with_columns(qb_pts_per_0_1_epa=pl.col("x_qb") * 0.1 * EXAMPLE_DROPBACKS)


def status_models_by_week(inp: Inputs, raw_db=config.RAW_DUCKDB_PATH, max_season=None) -> dict:
    """{(season, week): 4a.1 status model fit before that week's first kickoff} for the seasons the injury layer covers."""
    from models import injuries as ij
    pw = ij.build_player_weeks(raw_db, max_season)
    cut = inp.games.group_by("season", "week").agg(pl.col("gameday").min()).rows()
    return {(s, w): ij.fit_status_model(pw, c) for s, w, c in cut if s >= config.FEATURE_HISTORY_START}


def build_expectations(raw_db=config.RAW_DUCKDB_PATH, max_season=None, ratings: pl.DataFrame | None = None):
    """Everything 4b.2 needs, end to end: (ratings, pre-game features, team predictions, game expectations, coefficients)."""
    inp = load_inputs(raw_db, max_season)
    ratings = ratings if ratings is not None else tr.compute_ratings(inp.plays, n_boot=0)
    feats = pregame_features(inp, ratings, status_models_by_week(inp, raw_db, max_season))
    tp, coefs = walk_forward(feats)
    return ratings, feats, tp, game_expectations(tp, inp.games), coefs
