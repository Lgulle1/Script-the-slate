"""4c.6.1 window selection: for each unit and market, which window's fingerprints retrieve the most informative comparables (build plan 4c.6).

Criterion (walk-forward, 2020-2024 targets only): for a target and a unit, take the K_NEIGHBOURS past observations most similar to the target on
that unit alone (the observation set and the unit's sides exactly as in the searches: the target offense vs the past offense, tonight's defense vs
the defense that past game faced, the player's archetype vs the past player's), predict the target's comp-free standardized residual z (4c.4) by
their similarity-weighted mean z, and score the squared error against the target's own z. A market's error is that of its volume z plus, for a
market with an efficiency side, that of its efficiency z. The window with the lowest mean error over the targets the unit can score is chosen; ties go
to the earlier window in WINDOW_ORDER. The continuity window of a market is the continuity_weighted variant of that market's penalty key.

Only observations with their own expectation count (2020+ scored rows), so 2020 targets have none and the choice rests on 2021-2024.
No threshold is involved: the choice does not depend on SIM_THRESHOLD / MIN_NEFF.
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config
from models import comps as C
from models import comps_spec as cs

K_NEIGHBOURS = 20
WINDOW_ORDER = ("last_3", "last_6", "season_to_date", "recency_weighted", "continuity_weighted")


def window_name(window: str, market: str) -> str:
    """The stored window of a window family for a market (the continuity window is keyed by the market's penalty row)."""
    return f"continuity_weighted:{config.BASELINE_TO_PENALTY_MARKET[market]}" if window == "continuity_weighted" else window


def _team_z(pool: C.Pool, zl: dict, q: str | None) -> np.ndarray:
    d = zl.get(q, {}) if q else {}
    return np.array([d.get((g, t), np.nan) for g, t in zip(pool._gids, pool.team.tolist())], dtype=float)


def _player_z(pool: C.Pool, unit: str, zl: dict, q: str | None) -> np.ndarray:
    pop = pool.pl[unit]
    d = zl.get(q, {}) if q else {}
    gids = [pool._gids[r] for r in pop["row"].tolist()]
    return np.array([d.get((g, p), np.nan) for g, p in zip(gids, pop["pid"].tolist())], dtype=float)


def unit_retrieval_errors(pool: C.Pool, targets: list, zl: dict, k_neighbours: int = K_NEIGHBOURS) -> pl.DataFrame:
    """One row per (target, market, unit): the squared error of the unit-only retrieval of the volume z and of the efficiency z (null when the
    target has no z or no past observation with one). `targets` are representatives of their market families (backtest_targets)."""
    rows = []
    cache_z = {}

    def zarr(kind, unit, q):
        key = (kind, unit, q)
        if key not in cache_z:
            cache_z[key] = _team_z(pool, zl, q) if kind == "team" else _player_z(pool, unit, zl, q)
        return cache_z[key]

    last = None
    for tg in targets:
        if (tg.game_id, tg.team) != last:
            pool.clear_cache()
            last = (tg.game_id, tg.team)
        units = cs.MARKET_UNITS[tg.market]
        arch = [u for u in units if u in cs.ARCHETYPE_UNITS]
        is_player = tg.player_id is not None
        g, o = pool.idx[(tg.game_id, tg.team)], pool.idx[(tg.game_id, tg.opponent)]
        key = int(pool.key[g])
        k = pool._kmax(key)
        if is_player:
            pop = pool.pl[arch[0]]
            kp = int(np.searchsorted(pop["key"], key, side="left"))
            oi = np.flatnonzero(pop["in_pool"][:kp])
            obs_row = pop["row"][oi]
            tpos = np.flatnonzero((pop["pid"] == tg.player_id) & (pop["row"] == g))
        else:
            obs_row = np.arange(k)
        obs_opp = pool.opp_row[obs_row] if len(obs_row) else obs_row
        for market in C.MARKET_FAMILIES[C.FAMILY_OF_MARKET[tg.market]]:
            qv, qe = C.MARKET_QUANTITIES[market]
            tz = {}
            for side, q in (("vol", qv), ("eff", qe)):
                if q is None:
                    continue
                d = zl.get(q, {})
                tz[side] = (q, d.get((tg.game_id, tg.player_id if is_player else tg.team), np.nan))
            for unit in units:
                if unit in cs.ARCHETYPE_UNITS:
                    if not len(tpos):
                        continue
                    sims = []
                    for space in C.stored_spaces(unit):
                        sig = pool._sigma(unit, space, key)
                        if not np.isfinite(sig):
                            continue
                        dd = C.unit_distance(pop["blocks"][space][0][tpos[0]][None, :], pop["blocks"][space][0][oi], unit, space)
                        sims.append((space, C.similarity(dd.d2[0], sig)))
                    if not sims:
                        continue
                    sd = dict(sims)
                    use_ext = (pop["blocks"]["extended"][1][oi] & bool(pop["blocks"]["extended"][1][tpos[0]])) if "extended" in sd else np.zeros(len(oi), bool)
                    nan = np.full(len(oi), np.nan)
                    sim = np.where(use_ext, sd.get("extended", nan), sd.get("base", nan))
                else:
                    kind, row, at = ("off", g, obs_row) if unit in cs.OFFENSE_UNITS else ("def", o, obs_opp)
                    sim = pool.unit_sims(unit, kind, row, k, key)[0][at]
                for side, (q, actual) in tz.items():
                    if not np.isfinite(actual):
                        continue
                    z_obs = zarr("player", arch[0], q)[oi] if is_player else zarr("team", None, q)[obs_row]
                    ok = np.isfinite(sim) & np.isfinite(z_obs)
                    if not ok.any():
                        continue
                    cand = np.flatnonzero(ok)
                    if len(cand) > k_neighbours:              # the k most similar; ties by position, as everywhere
                        kth = np.partition(sim[cand], len(cand) - k_neighbours)[len(cand) - k_neighbours]
                        cand = cand[sim[cand] >= kth]
                        cand = cand[np.lexsort((cand, -sim[cand]))][:k_neighbours]
                    w = sim[cand]
                    pred = float((w * z_obs[cand]).sum() / w.sum()) if w.sum() > 0 else float(z_obs[cand].mean())
                    rows.append(dict(season=tg.season, week=tg.week, game_id=tg.game_id, team=tg.team, player_id=tg.player_id, market=market,
                                     unit=unit, side=side, sq_error=(pred - actual) ** 2, n_neighbours=len(cand)))
    return pl.DataFrame(rows, schema={"season": pl.Int64, "week": pl.Int64, "game_id": pl.String, "team": pl.String, "player_id": pl.String,
                                      "market": pl.String, "unit": pl.String, "side": pl.String, "sq_error": pl.Float64, "n_neighbours": pl.Int64})


def choose(errors: pl.DataFrame) -> tuple:
    """(summary, chosen): per (market, unit, window family) the mean error (volume + efficiency mean squared errors) over the targets every window
    could score, and the chosen window per (market, unit)."""
    per = errors.group_by("window_family", "market", "unit", "side", "game_id", "team", "player_id").agg(pl.col("sq_error").first())
    # compare windows on the same targets: keep (market, unit, side, target) scored by every window family
    nw = per["window_family"].n_unique()
    common = per.group_by("market", "unit", "side", "game_id", "team", "player_id").agg(pl.len().alias("n")).filter(pl.col("n") == nw).drop("n")
    per = per.join(common, on=["market", "unit", "side", "game_id", "team", "player_id"], how="inner")
    summ = (per.group_by("window_family", "market", "unit", "side").agg(mse=pl.col("sq_error").mean(), n_targets=pl.len())
            .group_by("window_family", "market", "unit").agg(error=pl.col("mse").sum(), n_targets=pl.col("n_targets").min()))
    order = {w: i for i, w in enumerate(WINDOW_ORDER)}
    summ = summ.with_columns(order=pl.col("window_family").replace_strict(order)).sort("market", "unit", "error", "order")
    chosen = summ.group_by("market", "unit", maintain_order=True).first().select("market", "unit", window_family="window_family", error="error",
                                                                                  n_targets="n_targets")
    return summ.drop("order").sort("market", "unit", "window_family"), chosen.sort("market", "unit")
