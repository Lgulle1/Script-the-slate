"""Opponent-adjusted team ratings (Phase 4b.1): offense pass, offense rush, defense pass, defense rush, in EPA per play.

Model, per play type (pass = every QB dropback: attempts, sacks, scrambles; rush = designed runs; two-point tries, kneels,
spikes and no-plays excluded), regular season only:

    EPA of a play = league mean + offense rating (team with the ball) + defense rating (the other team) + home-field term

fit by ridge regression on the target season's games BEFORE the target week (penalty RIDGE_LAMBDA on the team ratings, none
on the league mean or the home term). Plays are aggregated to (game, offense, type) rows, which gives exactly the play-level
fit. Games are weighted by the recency half-life of features/weights.py (0.5 ** (weeks back / 6)). A defense rating is the
EPA per play that defense ALLOWS relative to the league: positive = a worse defense. With an unpenalised mean, every week's
fitted offense ratings (and defense ratings) sum to exactly zero across the league; after the prior blend they are
re-centred to zero again.

Season start (item 3): each team starts the season at CARRYOVER x its final rating of the previous season (multiplying a
league-relative rating by a factor below 1 shrinks it toward league average), then moves toward the current season's fit with
n / (n + PRIOR_GAMES) where n is the games it has played this season before the target week:

    rating = w * current-season fit + (1 - w) * prior,      w = n / (n + PRIOR_GAMES)

The league mean and home term are blended the same way (n = the league's average games played). A season's FINAL ratings
(the prior for the next season) are the ratings for the week after its last regular-season week. CARRYOVER is tuned only on
2020-2024 (tune_carryover; the grid result is recorded next to the constant).

Uncertainty (item 2): BOOTSTRAP_N resamples of the current season's games (with replacement), each refit and blended with the
matching resample of the prior season's final ratings, so the SD carries the prior's uncertainty forward. The stored rating
is the full-data fit; the SD is the bootstrap standard deviation.

History: the ratings start at HISTORY_START (league average, no prior) so that 2020 week 1 has a real prior. Seasons before
config.FEATURE_HISTORY_START are history only -- a feature needing history, never scored. 2025+ is never loaded.

Coaching hook (item 4): `coaching_prior` maps (team, season) -> {rating name: (mean, strength in games)}. When given, the
season-start prior for that rating is (PRIOR_GAMES * carryover prior + strength * mean) / (PRIOR_GAMES + strength) and the blend
weight becomes n / (n + PRIOR_GAMES + strength). Defaults to nothing; 4d fills it. No coaching logic lives here.
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from features.weights import recency_weight

RATINGS = ("off_pass", "off_rush", "def_pass", "def_rush")
PLAY_TYPES = ("pass", "rush")
HISTORY_START = 2018
RIDGE_LAMBDA = 50.0            # penalty on each team rating, in recency-weighted plays (keeps early-season fits identifiable)
PRIOR_GAMES = 6.0              # k in n / (n + k)
CARRYOVER = 0.7                # tuned on 2020-2024 only (tune_carryover; start value was 0.6) -- see CARRYOVER_TUNING
# tune_carryover result (play-weighted mean squared error of predicted per-game mean EPA, every week of 2020-2024, no bootstrap):
#   carryover 0.0 0.066849 | 0.1 0.066580 | 0.2 0.066343 | 0.3 0.066142 | 0.4 0.065979 | 0.5 0.065858
#             0.6 0.065782 | 0.7 0.065756 | 0.8 0.065786 | 0.9 0.065879 | 1.0 0.066042      -> 0.7
# RIDGE_LAMBDA was fixed beforehand, not tuned; a check at carryover 0.7 gave 5: 0.067142, 20: 0.066331, 50: 0.065756,
# 150: 0.065792, 400: 0.066970.
CARRYOVER_TUNING = {0.0: 0.066849, 0.1: 0.066580, 0.2: 0.066343, 0.3: 0.066142, 0.4: 0.065979, 0.5: 0.065858, 0.6: 0.065782,
                    0.7: 0.065756, 0.8: 0.065786, 0.9: 0.065879, 1.0: 0.066042}
CARRYOVER_GRID = tuple(round(0.1 * i, 1) for i in range(11))
BOOTSTRAP_N = 200
FORBIDDEN_COLUMNS = ("spread_line", "total_line", "away_moneyline", "home_moneyline", "away_spread_odds", "home_spread_odds",
                     "over_odds", "under_odds")


# ------------------------------------------------------------------ data
def load_play_rows(raw_db=config.RAW_DUCKDB_PATH, max_season=None, first_season=HISTORY_START) -> pl.DataFrame:
    """One row per (game, offense, play type): n plays and mean EPA, with week, teams and the home flag
    (+0.5 home offense, -0.5 away offense, 0 at a neutral site)."""
    cap = config.cap_season(max_season)          # raises config.HoldoutError for 2025+
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute(
            "SELECT season, week, game_id, posteam, defteam, "
            "CASE WHEN location = 'Neutral' THEN 0.0 WHEN posteam = home_team THEN 0.5 ELSE -0.5 END AS home, "
            "CASE WHEN qb_dropback = 1 THEN 'pass' ELSE 'rush' END AS type, count(*) AS n, avg(epa) AS y "
            "FROM (SELECT DISTINCT ON (game_id, play_id) * FROM pbp ORDER BY game_id, play_id, pulled_at DESC) "
            f"WHERE season_type = 'REG' AND season >= {int(first_season)} AND season <= {cap} AND epa IS NOT NULL "
            "AND posteam IS NOT NULL AND defteam IS NOT NULL AND coalesce(two_point_attempt, 0) = 0 "
            "AND (qb_dropback = 1 OR (play_type = 'run' AND coalesce(qb_dropback, 0) = 0)) "
            "GROUP BY ALL").pl()
    finally:
        con.close()
    return d.with_columns(pl.col("n").cast(pl.Float64)).sort("season", "week", "game_id", "posteam", "type")


# ------------------------------------------------------------------ ridge
def _design(rows: pl.DataFrame, team_index: dict):
    off = np.array([team_index[t] for t in rows["posteam"].to_list()], dtype=int)
    de = np.array([team_index[t] for t in rows["defteam"].to_list()], dtype=int)
    return off, de, rows["home"].to_numpy().astype(float), rows["y"].to_numpy().astype(float), rows["n"].to_numpy().astype(float)


def design_matrix(off, de, home, n_teams: int) -> np.ndarray:
    """Columns: league mean, home term, offense team (n_teams), defense team (n_teams)."""
    m = len(off)
    X = np.zeros((m, 2 + 2 * n_teams))
    X[:, 0] = 1.0
    X[:, 1] = home
    X[np.arange(m), 2 + off] = 1.0
    X[np.arange(m), 2 + n_teams + de] = 1.0
    return X


def ridge_solve(X: np.ndarray, y: np.ndarray, wn: np.ndarray, n_teams: int, lam: float = RIDGE_LAMBDA):
    """Weighted ridge: minimise sum wn * (y - X b)^2 + lam * (sum of squared team ratings); the league mean and home term are
    not penalised. Returns (mu, h, offense ratings, defense ratings). The offense (and defense) ratings sum to zero exactly:
    the zero gradient for the unpenalised mean makes lam * sum(o) = 0."""
    Xw = X * wn[:, None]
    A = Xw.T @ X
    idx = np.arange(2, X.shape[1])
    A[idx, idx] += lam
    beta = np.linalg.solve(A, Xw.T @ y)
    return beta[0], beta[1], beta[2:2 + n_teams], beta[2 + n_teams:]


def ridge_fit(off, de, home, y, wn, n_teams: int, lam: float = RIDGE_LAMBDA):
    return ridge_solve(design_matrix(off, de, home, n_teams), y, wn, n_teams, lam)


# ------------------------------------------------------------------ the rating process
def compute_ratings(rows: pl.DataFrame, carryover: float = CARRYOVER, k: float = PRIOR_GAMES, lam: float = RIDGE_LAMBDA,
                    n_boot: int = BOOTSTRAP_N, coaching_prior: dict | None = None, seed: int = config.BOOTSTRAP_SEED,
                    last_season: int | None = None) -> pl.DataFrame:
    """Ratings for every (season, week) target from the first season in `rows` through `last_season`, plus each season's
    FINAL row (week = last week + 1, is_final = True). Uses only rows with (season, week) before the target.

    Columns: season, week, is_final, team, n_games, w_current, <rating>, <rating>_sd for each of RATINGS, and the league terms
    mu_pass, mu_rush, home_pass, home_rush.
    """
    teams = sorted(set(rows["posteam"].to_list()) | set(rows["defteam"].to_list()))
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    seasons = sorted(set(rows["season"].to_list()))
    if last_season is not None:
        seasons = [s for s in seasons if s <= last_season]
    out = []
    zero = {r: np.zeros(T) for r in RATINGS}
    prev_final = {"point": dict(zero), "boot": {r: np.zeros((n_boot, T)) for r in RATINGS},
                  "league": {"mu_pass": None, "mu_rush": None, "home_pass": 0.0, "home_rush": 0.0}}
    for season in seasons:
        srows = rows.filter(pl.col("season") == season)
        weeks = sorted(set(srows["week"].to_list()))
        targets = [(w, False) for w in weeks] + [(max(weeks) + 1, True)]
        # season-start prior (with the optional coaching hook)
        prior_pt, prior_bt, strength = {}, {}, {}
        for r in RATINGS:
            prior_pt[r] = carryover * prev_final["point"][r]
            prior_bt[r] = carryover * prev_final["boot"][r]
            strength[r] = np.zeros(T)
        for (team, s_), spec in (coaching_prior or {}).items():
            if s_ != season or team not in ti:
                continue
            for r, (mean, st) in spec.items():
                i = ti[team]
                prior_pt[r][i] = (k * prior_pt[r][i] + st * mean) / (k + st)
                prior_bt[r][:, i] = (k * prior_bt[r][:, i] + st * mean) / (k + st)
                strength[r][i] = st
        for week, is_final in targets:
            past = srows.filter(pl.col("week") < week)
            games = sorted(set(past["game_id"].to_list()))
            gidx = {g: i for i, g in enumerate(games)}
            n_games = np.zeros(T)
            for t, g in set(zip(past["posteam"].to_list(), past["game_id"].to_list())):
                n_games[ti[t]] += 1
            n_league = float(n_games[n_games > 0].mean()) if (n_games > 0).any() else 0.0
            cur_pt, cur_bt, league = {}, {}, {}
            for typ in PLAY_TYPES:
                sub = past.filter(pl.col("type") == typ)
                if not sub.height:
                    continue
                off, de, home, y, n = _design(sub, ti)
                X = design_matrix(off, de, home, T)
                w = np.array([recency_weight(float(week - x)) for x in sub["week"].to_list()])
                mu, h, o, d = ridge_solve(X, y, w * n, T, lam)
                cur_pt[f"off_{typ}"], cur_pt[f"def_{typ}"] = o, d
                league[f"mu_{typ}"], league[f"home_{typ}"] = mu, h
                if n_boot:
                    rng = np.random.default_rng([seed, season, week, 0 if typ == "pass" else 1])
                    g_of_row = np.array([gidx[g] for g in sub["game_id"].to_list()])
                    ob, db_ = np.zeros((n_boot, T)), np.zeros((n_boot, T))
                    for b in range(n_boot):
                        cnt = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))
                        wb = w * n * cnt[g_of_row]
                        if wb.sum() <= 0:
                            continue
                        _, _, ob[b], db_[b] = ridge_solve(X, y, wb, T, lam)
                    cur_bt[f"off_{typ}"], cur_bt[f"def_{typ}"] = ob, db_
            row = {}
            for r in RATINGS:
                wt = n_games / (n_games + k + strength[r])
                cp = cur_pt.get(r, np.zeros(T))
                pt = wt * cp + (1 - wt) * prior_pt[r]
                pt = pt - pt.mean()
                cb = cur_bt.get(r, np.zeros((n_boot, T)))
                bt = wt[None, :] * cb + (1 - wt[None, :]) * prior_bt[r] if n_boot else None
                if n_boot:
                    bt = bt - bt.mean(axis=1, keepdims=True)
                row[r] = (pt, bt)
            wl = n_league / (n_league + k)
            lg = {}
            for typ in PLAY_TYPES:
                prev_mu = prev_final["league"][f"mu_{typ}"]
                cur_mu = league.get(f"mu_{typ}")
                if cur_mu is None:
                    lg[f"mu_{typ}"] = prev_mu
                elif prev_mu is None:
                    lg[f"mu_{typ}"] = cur_mu
                else:
                    lg[f"mu_{typ}"] = wl * cur_mu + (1 - wl) * prev_mu
                cur_h = league.get(f"home_{typ}")
                prev_h = prev_final["league"][f"home_{typ}"]
                lg[f"home_{typ}"] = prev_h if cur_h is None else wl * cur_h + (1 - wl) * prev_h
            for i, t in enumerate(teams):
                rec = dict(season=season, week=int(week), is_final=is_final, team=t, n_games=int(n_games[i]),
                           w_current=float(n_games[i] / (n_games[i] + k)), **lg)
                for r in RATINGS:
                    pt, bt = row[r]
                    rec[r] = float(pt[i])
                    rec[f"{r}_sd"] = float(bt[:, i].std(ddof=1)) if n_boot > 1 else None
                out.append(rec)
            if is_final:
                prev_final = {"point": {r: row[r][0] for r in RATINGS},
                              "boot": {r: (row[r][1] if n_boot else np.zeros((0, T))) for r in RATINGS},
                              "league": lg}
    return pl.DataFrame(out, infer_schema_length=None).sort("season", "week", "team")



def build_team_ratings(raw_db=config.RAW_DUCKDB_PATH, max_season=None, carryover: float = CARRYOVER, n_boot: int = BOOTSTRAP_N,
                       coaching_prior: dict | None = None) -> pl.DataFrame:
    """Ratings for HISTORY_START..max_season (default: the last backtest season). Seasons before FEATURE_HISTORY_START are
    history (is_history = True): they exist so 2020 starts from a real prior, and are never scored."""
    rows = load_play_rows(raw_db, max_season)
    r = compute_ratings(rows, carryover=carryover, n_boot=n_boot, coaching_prior=coaching_prior)
    return r.with_columns(is_history=pl.col("season") < config.FEATURE_HISTORY_START)


# ------------------------------------------------------------------ prediction + tuning
def predicted_epa(ratings: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """Join each (game, offense, type) row to the ratings in force for its week and return the predicted mean EPA."""
    r = ratings.filter(~pl.col("is_final"))
    off = r.select("season", "week", posteam="team", off_pass="off_pass", off_rush="off_rush", mu_pass="mu_pass", mu_rush="mu_rush",
                   home_pass="home_pass", home_rush="home_rush")
    de = r.select("season", "week", defteam="team", def_pass="def_pass", def_rush="def_rush")
    d = rows.join(off, on=["season", "week", "posteam"], how="inner").join(de, on=["season", "week", "defteam"], how="inner")
    pred = (pl.when(pl.col("type") == "pass")
            .then(pl.col("mu_pass") + pl.col("home") * pl.col("home_pass") + pl.col("off_pass") + pl.col("def_pass"))
            .otherwise(pl.col("mu_rush") + pl.col("home") * pl.col("home_rush") + pl.col("off_rush") + pl.col("def_rush")))
    return d.with_columns(pred=pred)


def tune_carryover(rows: pl.DataFrame, grid=CARRYOVER_GRID, seasons=config.BACKTEST_SEASONS) -> pl.DataFrame:
    """Out-of-sample play-weighted squared error of the predicted mean EPA per (game, offense, type) for every week of
    `seasons` (2020-2024 only), for each carryover value. Ratings are fit with no bootstrap. Lower is better."""
    res = []
    for c in grid:
        r = compute_ratings(rows, carryover=c, n_boot=0, last_season=max(seasons))
        p = predicted_epa(r, rows.filter(pl.col("season").is_in(list(seasons))))
        p = p.filter(pl.col("mu_pass").is_not_null() & pl.col("mu_rush").is_not_null())
        early = p.filter(pl.col("week") <= 4)
        res.append(dict(carryover=c, sse=float((p["n"] * (p["y"] - p["pred"]) ** 2).sum() / p["n"].sum()),
                        sse_weeks_1_4=float((early["n"] * (early["y"] - early["pred"]) ** 2).sum() / early["n"].sum()), rows=p.height))
    return pl.DataFrame(res)


# ------------------------------------------------------------------ persistence
def persist(ratings: pl.DataFrame, db_path=config.DUCKDB_PATH, out_dir=config.PROCESSED_DIR) -> None:
    """data/processed/team_ratings.parquet and the DuckDB table team_ratings (CREATE OR REPLACE)."""
    config.ensure_data_dirs()
    ratings.write_parquet(out_dir / "team_ratings.parquet")
    con = duckdb.connect(str(db_path))
    try:
        con.register("_t", ratings.to_arrow())
        con.execute("CREATE OR REPLACE TABLE team_ratings AS SELECT * FROM _t")
        con.unregister("_t")
    finally:
        con.close()
