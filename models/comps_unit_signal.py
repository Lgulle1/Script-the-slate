"""The fingerprint diagnostic (decision of 2026-10-09; fixed in docs/overnight_plan.md before it ran): does each team unit's fingerprint carry signal
about its own unit's next game?

Per unit (the nine team units) and every 2020-2024 team-game with the unit's vector and outcome (the offense's row for offense units, the defense's
row for defense units; walk-forward):
  outcome  the unit's headline feature (its first comps_spec feature) on that single game, from the comps ledger;
  (a)      the unit's own trailing average: the recency-weighted mean of that outcome in the team's earlier games;
  (k)      the similarity-weighted mean of the single-game outcomes of the K earlier team-games most similar on that unit alone (Pool.unit_sims:
           the unit's sigma, the space as in the searches, the healthy target vector against the pool vectors);
  (r)      K earlier team-games at random with (k)'s weights, N_DRAWS draws (reported only).
Absolute error. PASS per unit: (k) beats (a), mean gain above zero with the season-week bootstrap lower bound above zero at level 1 - 0.05 / 9.
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config
from eval import compare
from models import comps as C
from models import comps_spec as cs
from models import comps_team_test as TT

K = 20
N_DRAWS = 10
SEED = 20261011
UNITS = cs.OFFENSE_UNITS + cs.DEFENSE_UNITS


def headline(unit: str) -> str:
    """The unit's headline feature: the first one comps_spec lists for it."""
    return C.feature_keys(unit)[0]


def single_game_outcomes(wide: pl.DataFrame) -> dict:
    """unit -> {(game_id, team): the headline's single-game value n / d} from the wide comps ledger (missing when d is 0)."""
    out = {}
    for u in UNITS:
        f = headline(u)
        n, d = wide[f"{f}|n"].to_numpy().astype(float), wide[f"{f}|d"].to_numpy().astype(float)
        v = np.divide(n, d, out=np.full(len(n), np.nan), where=d > 0)
        out[u] = dict(zip(zip(wide["game_id"].to_list(), wide["team"].to_list()), v.tolist()))
    return out


def load_ledger(raw_db=config.RAW_DUCKDB_PATH, max_season: int = max(config.BACKTEST_SEASONS)) -> pl.DataFrame:
    """The wide team ledger (team + structure features, '|n' / '|d' per (game_id, team)), as models.comps.load_inputs builds it; never the holdout."""
    from features import comps_ledger as L
    if max_season >= config.HOLDOUT_SEASON:
        raise config.HoldoutError(f"the ledger through {max_season} would read the holdout season {config.HOLDOUT_SEASON}")
    games, plays = L.load_games(raw_db, max_season), L.load_plays(raw_db, max_season)
    pos, snaps = L.load_positions(raw_db, max_season), L.load_snaps(raw_db, max_season)
    pfr_pass, pfr_rush = L.load_pfr(raw_db, max_season)
    team = L.team_ledger(plays, games, pfr_pass)
    player = L.player_ledger(plays, team, pos, snaps, pfr_rush, games)
    return (team.join(L.structure_ledger(player, team), on=["game_id", "team"], how="left")
            .join(L.coverage_position_ledger(plays, pos), on=["game_id", "team"], how="left"))


def knn(sim: np.ndarray, y: np.ndarray, k: int = K) -> tuple:
    """(prediction, positions, weights): the k most similar positions with a finite similarity and outcome (ties by position), similarity-weighted."""
    cand = np.flatnonzero(np.isfinite(sim) & np.isfinite(y))
    if not len(cand):
        return np.nan, cand, np.zeros(0)
    if len(cand) > k:
        kth = np.partition(sim[cand], len(cand) - k)[len(cand) - k]
        cand = cand[sim[cand] >= kth]
        cand = cand[np.lexsort((cand, -sim[cand]))][:k]
    w = sim[cand]
    pred = float((w * y[cand]).sum() / w.sum()) if w.sum() > 0 else float(y[cand].mean())
    return pred, cand, w


def unit_rows(pool: C.Pool, outcomes: dict, units=UNITS, k: int = K, n_draws: int = N_DRAWS, seed: int = SEED) -> pl.DataFrame:
    """One row per (unit, 2020-2024 target team-game): outcome, (a), (k), the mean absolute error of (r)."""
    gids, teams = pool._gids, pool.team
    key = np.asarray(pool.key, dtype=np.int64)
    clock = np.asarray(pool.clock, dtype=float)
    games = pool.games
    seasons = games["season"].to_numpy()
    weeks = games["week"].to_numpy()
    rows = []
    for ui, u in enumerate(units):
        y = np.array([outcomes[u].get((g, t), np.nan) for g, t in zip(gids, teams.tolist())], dtype=float)
        a_all = TT.recent_average(key, clock, teams, y)
        kind = "off" if u in cs.OFFENSE_UNITS else "def"
        rng = np.random.default_rng((seed, ui))
        for g in np.flatnonzero(np.isin(seasons, config.BACKTEST_SEASONS)):
            if not (np.isfinite(y[g]) and np.isfinite(a_all[g])):
                continue
            kk = pool._kmax(int(key[g]))
            pool.clear_cache()
            sim = pool.unit_sims(u, kind, int(g), kk, int(key[g]))[0]
            pred, cand, w = knn(sim, y[:kk], k)
            if not np.isfinite(pred):
                continue
            pool_ok = np.flatnonzero(np.isfinite(y[:kk]))
            rl = float(np.mean([abs(float((w * y[pool_ok[rng.choice(len(pool_ok), size=len(w), replace=False)]]).sum() / w.sum()) - y[g])
                                for _ in range(n_draws)])) if w.sum() > 0 else np.nan
            rows.append(dict(unit=u, season=int(seasons[g]), week=int(weeks[g]), game_id=gids[g], team=str(teams[g]), outcome=float(y[g]),
                             a=float(a_all[g]), k=pred, loss_a=abs(a_all[g] - y[g]), loss_k=abs(pred - y[g]), loss_r=rl, n_neighbours=len(cand)))
    return pl.DataFrame(rows, schema={"unit": pl.String, "season": pl.Int64, "week": pl.Int64, "game_id": pl.String, "team": pl.String,
                                      "outcome": pl.Float64, "a": pl.Float64, "k": pl.Float64, "loss_a": pl.Float64, "loss_k": pl.Float64,
                                      "loss_r": pl.Float64, "n_neighbours": pl.Int64})


def verdicts(rows: pl.DataFrame, units=UNITS) -> pl.DataFrame:
    """Per unit: MAEs, (k)'s gain over (a) (the pass test, Bonferroni level 1 - 0.05 / len(units)) and over (r) (reported), seasons won, PASS."""
    level = 1 - 0.05 / len(units)
    out = []
    for u in units:
        r = rows.filter(pl.col("unit") == u).sort("season", "week", "game_id", "team")
        row = dict(unit=u, headline=headline(u), n=r.height, interval_level=level)
        if r.height:
            cl = (r["season"] * 100 + r["week"]).to_numpy()
            for name, base, lv in (("vs_a", "loss_a", level), ("vs_random", "loss_r", config.BOOTSTRAP_CI)):
                d = (r[base] - r["loss_k"]).to_numpy()
                lo, hi = compare.cluster_bootstrap(d, cl, level=lv)
                row.update({f"gain_{name}": float(d.mean()), f"ci_lo_{name}": float(lo), f"ci_hi_{name}": float(hi)})
            s = r["season"].to_numpy()
            dd = (r["loss_a"] - r["loss_k"]).to_numpy()
            row.update(mae_a=float(r["loss_a"].mean()), mae_k=float(r["loss_k"].mean()), mae_r=float(r["loss_r"].mean()),
                       seasons_won_vs_a=int(sum(dd[s == x].mean() > 0 for x in sorted(set(s.tolist())))))
            row["passes"] = bool(row["gain_vs_a"] > 0 and row["ci_lo_vs_a"] > 0)
        else:
            row["passes"] = False
        out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)
