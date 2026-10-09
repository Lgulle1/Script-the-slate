"""The team-level sanity test (decision of 2026-10-09; fixed in docs/overnight_plan.md before it ran).

Do similar matchups predict a team's output, measured against the offense's own normal output, without the base model?
  (a)        the offense's own recent average: the recency-weighted mean of the team's earlier games (the comps' half-life), across seasons;
  (b)        (a) + the similarity-weighted mean, over the matched past team-games, of (that game's outcome - that offense's own recent average
             before it), shrunk by n_eff / (n_eff + SHRINK_K); matches from the market's per-unit S4 pair at THRESHOLD (geometric mean, 0.40 floor),
             MIN_NEFF 2, recency x continuity x quality weights, the 25% per-team-game cap, healthy vectors, recency_weighted window; no match: (b) = (a);
  (b-random) the same adjustment from past team-games drawn at random from the same pool before the target week, the real weights kept (so n_eff
             and the shrinkage are identical); N_DRAWS draws, fixed seed.
Markets: team rushing yards (run_offense x run_defense) and team pass attempts (pass_offense x pass_coverage). Targets: 2020-2024 team-games; past
games from 2016. Absolute error; primary rows the team-games where (b) has a match. PASS per market: (b) beats (a) AND (b-random), each with the
season-week bootstrap lower bound above zero. 2025 is never loaded.
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from eval import compare
from models import comps as C
from models import comps_spec as cs

THRESHOLD = 0.50
N_DRAWS = 10
SEED = 20261010
FIRST_SEASON = 2016
MARKETS = {"team_rush_yds": ("rush_yds", "S4:run_offense*run_defense"), "team_pass_att": ("pass_att", "S4:pass_offense*pass_coverage")}
SEARCH_MARKET = "spread"          # a team market whose S4 has both pairs; its continuity row weighs the matches


def team_outcomes(raw_db=config.RAW_DUCKDB_PATH, first: int = FIRST_SEASON, last: int = max(config.BACKTEST_SEASONS)) -> pl.DataFrame:
    """Per regular-season team-game, first..last (never the holdout): rushing yards and pass attempts summed over the team's players (player_stats,
    the latest pull of each player-week, as features/volume_features.build_team_volume)."""
    if last >= config.HOLDOUT_SEASON:
        raise config.HoldoutError(f"team outcomes through {last} would read the holdout season {config.HOLDOUT_SEASON}")
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        return con.execute(
            "SELECT game_id, team, CAST(sum(rushing_yards) AS DOUBLE) AS rush_yds, CAST(sum(attempts) AS DOUBLE) AS pass_att FROM "
            "(SELECT DISTINCT ON (player_id, season, week) * FROM player_stats ORDER BY player_id, season, week, pulled_at DESC) "
            "WHERE season_type = 'REG' AND season BETWEEN ? AND ? GROUP BY game_id, team ORDER BY game_id, team", [first, last]).pl()
    finally:
        con.close()


def recent_average(key: np.ndarray, clock: np.ndarray, team: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per pool row: the recency-weighted mean of the same team's outcomes in strictly earlier weeks (key < its key), weight
    0.5 ** (clock gap / RECENCY_HALF_LIFE_GAMES); NaN without an earlier game with an outcome."""
    out = np.full(len(key), np.nan)
    for tm in np.unique(team):
        rows = np.flatnonzero(team == tm)
        rows = rows[np.argsort(key[rows], kind="stable")]
        for j, r in enumerate(rows):
            past = rows[:j]
            past = past[(key[past] < key[r]) & np.isfinite(y[past])]
            if len(past):
                w = 0.5 ** (np.maximum(clock[r] - clock[past], 0.0) / config.RECENCY_HALF_LIFE_GAMES)
                out[r] = float((w * y[past]).sum() / w.sum())
    return out


def adjustment(w: np.ndarray, dev: np.ndarray) -> float:
    """sum(w dev) / sum(w) x n_eff / (n_eff + SHRINK_K) (one observation per past team-game)."""
    n = C.neff(w)
    return float((w * dev).sum() / w.sum() * n / (n + cs.SHRINK_K))


def test_rows(pool: C.Pool, outcomes: pl.DataFrame, threshold: float = THRESHOLD, n_draws: int = N_DRAWS, seed: int = SEED) -> pl.DataFrame:
    """One row per (2020-2024 target team-game, market): actual, (a), (b), the mean absolute error of (b-random), whether (b) has a match, n_eff."""
    gids, teams = pool._gids, pool.team
    key = np.asarray(pool.key, dtype=np.int64)
    clock = np.asarray(pool.clock, dtype=float)
    yk = dict(zip(zip(outcomes["game_id"].to_list(), outcomes["team"].to_list()), zip(outcomes["rush_yds"].to_list(), outcomes["pass_att"].to_list())))
    ys = {col: np.array([yk.get((g, t), (np.nan, np.nan))[i] for g, t in zip(gids, teams.tolist())], dtype=float) for i, col in enumerate(("rush_yds", "pass_att"))}
    norm = {col: recent_average(key, clock, teams, ys[col]) for col in ys}
    dev = {col: ys[col] - norm[col] for col in ys}                     # each past game's outcome against its offense's own normal output
    games = pool.games
    rows = []
    for i, r in enumerate(games.filter(pl.col("season").is_in(config.BACKTEST_SEASONS)).sort("season", "week", "game_id", "team").iter_rows(named=True)):
        g = pool.idx[(r["game_id"], r["team"])]
        if (r["game_id"], r["opponent"]) not in pool.idx:
            continue
        tg = C.Target(SEARCH_MARKET, r["game_id"], r["team"], r["opponent"], r["season"], r["week"])
        pool.clear_cache()
        res = pool.search(tg, ("S4",), keep=True, sim_threshold=threshold, min_neff=cs.MIN_NEFF, obs_positions=True)
        rng = np.random.default_rng((seed, i))
        for market, (col, uid) in MARKETS.items():
            y, a = ys[col][g], norm[col][g]
            if not (np.isfinite(y) and np.isfinite(a)):
                continue
            rr = res[uid]
            b, rand_loss, matched, n_eff = a, abs(a - y), False, 0.0
            m = rr.matches
            if not rr.summary["no_match"] and m is not None and m.height:
                orow = np.array([pool.idx[(gg, tt)] for gg, tt in zip(m["obs_game_id"].to_list(), m["obs_team"].to_list())])
                d = dev[col][orow]
                ok = np.isfinite(d)
                tgid = (m["obs_game_id"] + "|" + m["obs_team"]).to_numpy()[ok]
                w = C.cap_shares(m["final_weight"].to_numpy()[ok], tgid, cs.TEAM_CAP_PER_SEARCH)
                if ok.any() and C.meets_min_neff(C.neff(w), cs.MIN_NEFF):
                    matched, n_eff = True, C.neff(w)
                    b = a + adjustment(w, d[ok])
                    cand = dev[col][rr.summary["obs_pos"]]
                    cand = cand[np.isfinite(cand)]
                    rand_loss = float(np.mean([abs(a + adjustment(w, cand[rng.choice(len(cand), size=len(w), replace=False)]) - y)
                                               for _ in range(n_draws)]))
            rows.append(dict(market=market, season=r["season"], week=r["week"], game_id=r["game_id"], team=r["team"], actual=float(y), a=float(a),
                             b=float(b), loss_a=abs(a - y), loss_b=abs(b - y), loss_b_random=rand_loss, matched=matched, n_eff=n_eff))
    return pl.DataFrame(rows, schema={"market": pl.String, "season": pl.Int64, "week": pl.Int64, "game_id": pl.String, "team": pl.String,
                                      "actual": pl.Float64, "a": pl.Float64, "b": pl.Float64, "loss_a": pl.Float64, "loss_b": pl.Float64,
                                      "loss_b_random": pl.Float64, "matched": pl.Boolean, "n_eff": pl.Float64})


def verdicts(rows: pl.DataFrame) -> pl.DataFrame:
    """Per market and row set (matched = the primary rows; all): (b)'s gain over (a) and over (b-random) with season-week intervals, and PASS
    (primary rows) when both gains are above zero with lower bounds above zero."""
    out = []
    for market in MARKETS:
        for rowset in ("matched", "all"):
            r = rows.filter((pl.col("market") == market) & (pl.col("matched") if rowset == "matched" else pl.lit(True))).sort("season", "week", "game_id", "team")
            row = dict(market=market, rows=rowset, n=r.height, match_rate=float(rows.filter(pl.col("market") == market)["matched"].mean()))
            if r.height:
                cl = (r["season"] * 100 + r["week"]).to_numpy()
                for name, base in (("vs_a", "loss_a"), ("vs_random", "loss_b_random")):
                    d = (r[base] - r["loss_b"]).to_numpy()
                    lo, hi = compare.cluster_bootstrap(d, cl)
                    row.update({f"gain_{name}": float(d.mean()), f"ci_lo_{name}": float(lo), f"ci_hi_{name}": float(hi)})
                row.update(mae_a=float(r["loss_a"].mean()), mae_b=float(r["loss_b"].mean()), mae_b_random=float(r["loss_b_random"].mean()))
                row["passes"] = bool(rowset == "matched" and row["gain_vs_a"] > 0 and row["ci_lo_vs_a"] > 0 and row["gain_vs_random"] > 0 and row["ci_lo_vs_random"] > 0)
            out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)
