"""The SIM_THRESHOLD sweep (decision of 2026-10-09, rule fixed in docs/overnight_plan.md before any error was computed).

For every threshold in THRESHOLDS, every 2020-2024 target, search and quantity (volume, and efficiency where the market has one; markets that share a
quantity count once: the first market of the family, MARKET_FAMILIES order, that has it) where the search matches on that side at that threshold:
  real error    (shift - z)^2, the search's shift against the target's own comp-free standardized residual z;
  random error  the same with every match replaced by a past observation drawn at random from the same search's observation set as of the same
                target week, among those with an expectation on that side (the past games the search could have picked, ignoring similarity);
                the real weights are kept, so n_eff and the shrinkage are identical and only the z's change; averaged over N_DRAWS draws.
improvement = random error - real error, pooled over all searches, markets and sides, with the 3.3 season-week cluster bootstrap. The chosen
threshold is the lowest whose pooled improvement is above zero with the interval's lower bound above zero; if none passes, DEFAULT_THRESHOLD stays.

Everything else is as the searches run (MIN_NEFF, the 0.40 floor, the geometric-mean check, count weights, healthy vectors). One search at the lowest
threshold serves every threshold (`restrict`: a match at threshold t is a match at the lowest threshold whose checked similarity reaches t).
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config
from eval import compare
from models import comps as C
from models import comps_spec as cs

THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_THRESHOLD = 0.70
N_DRAWS = 10
SEED = 20261009


def restrict(res: dict, t: float, min_neff: float = cs.MIN_NEFF) -> dict:
    """The search results at threshold t from results computed at a lower threshold (keep=True): matches with checked similarity >= t, and the
    no-match rule re-applied (best similarity is over all observations, so it does not change)."""
    out = {}
    for s, r in res.items():
        summ = dict(r.summary)
        m = r.matches
        if not summ["applicable"] or summ["reason"] in ("no_target_vector", "missing_required_ftn_features", "no_similarity", "weaker_factor_below_floor"):
            out[s] = r
            continue
        m = m.filter(pl.col("sim_check") >= t) if m is not None else None
        n = 0 if m is None else m.height
        nef = C.cluster_neff(m["final_weight"].to_numpy(), (m["obs_game_id"] + "|" + m["obs_team"]).to_numpy()) if n else 0.0
        best = summ["best_similarity"]
        reason = "best_similarity_below_threshold" if best < t else (None if C.meets_min_neff(nef, min_neff) else "n_eff_below_minimum")
        summ.update(n_matches=n, n_eff=nef, no_match=reason is not None, widened_uncertainty=reason is not None,
                    shift=0.0 if reason is not None else float("nan"), reason=reason)
        out[s] = C.SearchResult(s, summ, m)
    return out


def side_markets() -> dict:
    """family -> [(market, side, quantity)]: each quantity of the family once, from the first market (MARKET_FAMILIES order) that has it."""
    out = {}
    for fam, ms in C.MARKET_FAMILIES.items():
        seen, rows = set(), []
        for m in ms:
            for side, q in zip(("vol", "eff"), C.MARKET_QUANTITIES[m]):
                if q is not None and q not in seen:
                    seen.add(q)
                    rows.append((m, side, q))
        out[fam] = rows
    return out


def random_shift(w: np.ndarray, n_eff: float, cand_z: np.ndarray, rng: np.random.Generator) -> float:
    """The shift with the real weights `w` (the matches that carry weight) and z's drawn without replacement from `cand_z` (the search's observation
    set with an expectation on this side): sum(w z) / sum(w) x n_eff / (n_eff + SHRINK_K), the real n_eff."""
    z = cand_z[rng.choice(len(cand_z), size=len(w), replace=False)]
    return float((w * z).sum() / w.sum() * n_eff / (n_eff + cs.SHRINK_K))


def _zarr(pool: C.Pool, unit: str | None, src: dict, q: str) -> np.ndarray:
    d = src.get(q, {})
    if unit is None:
        return np.array([d.get((g, t), np.nan) for g, t in zip(pool._gids, pool.team.tolist())], dtype=float)
    pop = pool.pl[unit]
    return np.array([d.get((pool._gids[r], p), np.nan) for r, p in zip(pop["row"].tolist(), pop["pid"].tolist())], dtype=float)


def sweep_rows(pool: C.Pool, targets: list, zl: dict, counts: dict, thresholds=THRESHOLDS, min_neff: float = cs.MIN_NEFF,
               n_draws: int = N_DRAWS, seed: int = SEED) -> tuple:
    """(rows, rates): rows -- one per threshold, target, search and side where the search matches on that side, with the real and the mean random
    squared error; rates -- per threshold, target, market and search, whether the search matched (volume side), for the match-rate report."""
    sm = side_markets()
    zcache = {}

    def zall(unit, q):
        if (unit, q) not in zcache:
            zcache[(unit, q)] = _zarr(pool, unit, zl, q)
        return zcache[(unit, q)]

    rows, rates = [], []
    last = None
    for i, tg in enumerate(targets):
        if (tg.game_id, tg.team) != last:
            pool.clear_cache()
            last = (tg.game_id, tg.team)
        fam = C.FAMILY_OF_MARKET[tg.market]
        is_player = tg.player_id is not None
        arch = [u for u in cs.MARKET_UNITS[tg.market] if u in cs.ARCHETYPE_UNITS]
        unit = arch[0] if is_player else None
        rng = np.random.default_rng((seed, i))
        base = None
        for tgr, markets in C._target_rows([tg]):
            res0 = pool.search(tgr, C.SEARCHES, keep=True, sim_threshold=min(thresholds), min_neff=min_neff, obs_positions=True)
            for t in thresholds:
                res = restrict(res0, t, min_neff)
                for m in markets:
                    vals, _, sets, caps = C._market_shifts(res, tgr, m, zl, min_neff, counts)
                    for s in C.SEARCHES:
                        if res[s].summary["applicable"]:
                            rates.append(dict(threshold=t, season=tg.season, week=tg.week, game_id=tg.game_id, team=tg.team, player_id=tg.player_id,
                                              market=m, search=s, matched=not vals[f"nomatch_{s}"]))
                    for (mk, side, q) in sm[fam]:
                        if mk != m:
                            continue
                        tz = zl.get(q, {}).get((tg.game_id, tg.player_id if is_player else tg.team))
                        if tz is None or not np.isfinite(tz):
                            continue
                        j = 1 if side == "vol" else 2
                        for s in C.SEARCHES:
                            nm = vals[f"nomatch_{s}"] if side == "vol" else vals[f"nomatch_eff_{s}"]
                            if nm or s not in sets:
                                continue
                            w = caps[side][s]
                            ok = (w > 0) & np.isfinite(sets[s][j])
                            n_eff = vals[f"n_eff_{s}"] if side == "vol" else vals[f"n_eff_eff_{s}"]
                            real = vals[f"shift_vol_{s}"] if side == "vol" else vals[f"shift_eff_{s}"]
                            cz = zall(unit, q)[res0[s].summary["obs_pos"]]
                            cz = cz[np.isfinite(cz)]
                            rand = [random_shift(w[ok], n_eff, cz, rng) for _ in range(n_draws)]
                            rows.append(dict(threshold=t, season=tg.season, week=tg.week, game_id=tg.game_id, team=tg.team, player_id=tg.player_id,
                                             market=m, search=s, side=side, quantity=q, n_matches=int(ok.sum()), n_eff=float(n_eff), z=float(tz),
                                             shift=float(real), sq_error_real=float((real - tz) ** 2),
                                             sq_error_random=float(np.mean([(r - tz) ** 2 for r in rand]))))
    schema = {"threshold": pl.Float64, "season": pl.Int64, "week": pl.Int64, "game_id": pl.String, "team": pl.String, "player_id": pl.String,
              "market": pl.String, "search": pl.String}
    rows_df = pl.DataFrame(rows, schema=schema | {"side": pl.String, "quantity": pl.String, "n_matches": pl.Int64, "n_eff": pl.Float64, "z": pl.Float64,
                                                  "shift": pl.Float64, "sq_error_real": pl.Float64, "sq_error_random": pl.Float64})
    return rows_df, pl.DataFrame(rates, schema=schema | {"matched": pl.Boolean})


def summarize(rows: pl.DataFrame, thresholds=THRESHOLDS) -> pl.DataFrame:
    """Per threshold: the pooled improvement (random - real squared error), its 95% season-week cluster bootstrap interval and the number of rows."""
    out = []
    for t in thresholds:
        r = rows.filter(pl.col("threshold") == t).sort("season", "week", "game_id", "team", "player_id", "market", "search", "side", nulls_last=True)
        if r.height == 0:
            out.append(dict(threshold=t, n_rows=0, improvement=None, ci_lo=None, ci_hi=None, real=None, random=None))
            continue
        d = (r["sq_error_random"] - r["sq_error_real"]).to_numpy()
        lo, hi = compare.cluster_bootstrap(d, (r["season"] * 100 + r["week"]).to_numpy())
        out.append(dict(threshold=t, n_rows=r.height, improvement=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                        real=float(r["sq_error_real"].mean()), random=float(r["sq_error_random"].mean())))
    return pl.DataFrame(out, schema={"threshold": pl.Float64, "n_rows": pl.Int64, "improvement": pl.Float64, "ci_lo": pl.Float64, "ci_hi": pl.Float64,
                                     "real": pl.Float64, "random": pl.Float64})


def choose_threshold(summary: pl.DataFrame, default: float = DEFAULT_THRESHOLD) -> float:
    """The lowest threshold whose pooled improvement is above zero with the interval's lower bound above zero; none: the default."""
    ok = summary.filter((pl.col("improvement") > 0) & (pl.col("ci_lo") > 0)).sort("threshold")
    return float(ok["threshold"][0]) if ok.height else default


def breakdown(rows: pl.DataFrame, rates: pl.DataFrame) -> pl.DataFrame:
    """For reading only: per threshold, search, market and side the mean improvement and rows, and per threshold, market and search the match rate."""
    imp = (rows.group_by("threshold", "search", "market", "side")
           .agg(n_rows=pl.len(), improvement=(pl.col("sq_error_random") - pl.col("sq_error_real")).mean(), mean_n_eff=pl.col("n_eff").mean()))
    rate = rates.group_by("threshold", "market", "search").agg(match_rate=pl.col("matched").mean(), n_targets=pl.len())
    return rate.join(imp, on=["threshold", "market", "search"], how="left").sort("threshold", "market", "search", "side", nulls_last=True)
