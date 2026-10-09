"""The SIM_THRESHOLD sweep (decision of 2026-10-09, rule fixed in docs/overnight_plan.md before any error was computed).

For every threshold in THRESHOLDS, every 2020-2024 target, unit search (comps.unit_searches) and quantity (volume, and efficiency where the market has one; markets that share a
quantity count once: the first market of the family, MARKET_FAMILIES order, that has it) where the search matches on that side at that threshold:
  real error    (shift - z)^2, the search's shift against the target's own comp-free standardized residual z;
  random error  the same with every match replaced by a past observation drawn at random from the same search's observation set as of the same
                target week, among those with an expectation on that side (the past games the search could have picked, ignoring similarity);
                the real weights are kept, so n_eff and the shrinkage are identical and only the z's change; averaged over N_DRAWS draws.
improvements = random error - real error and z^2 (a zero shift) - real error, pooled over all searches, markets and sides, with the 3.3 season-week
cluster bootstrap. The chosen threshold is the lowest where BOTH are above zero with their intervals' lower bounds above zero (tightened on
2026-10-09 after the all-units baseline, before the per-unit sweep was read); if none passes, DEFAULT_THRESHOLD stays.

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
    """(rows, rates): rows -- one per threshold, target, unit search and side where the unit search matches on that side, with the real and the mean random
    squared error; rates -- per threshold, target, market, search and side, whether the search matched on that side, for the match-rate report."""
    sm = side_markets()
    zcache = {}

    def zall(market, unit, q):
        """The z of every observation of the search's index space: pool rows (team markets) or the archetype population of the pool the search read
        the archetype from (a WindowedPool reads it from the archetype's chosen window)."""
        zp = pool.pools[pool.choice[market][unit]] if (unit is not None and isinstance(pool, C.WindowedPool)) else (pool.base if isinstance(pool, C.WindowedPool) else pool)
        if (id(zp), unit, q) not in zcache:
            zcache[(id(zp), unit, q)] = _zarr(zp, unit, zl, q)
        return zcache[(id(zp), unit, q)]

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
        for tgr, markets in C._target_rows([tg]):
            res0 = pool.search(tgr, C.SEARCHES, keep=True, sim_threshold=min(thresholds), min_neff=min_neff, obs_positions=True)
            for t in thresholds:
                res = restrict(res0, t, min_neff)
                for m in markets:
                    _, det, sets, caps = C._market_shifts(res, tgr, m, zl, min_neff, counts)
                    ud = {d["search"]: d for d in det}                  # per unit search
                    for s, d in ud.items():
                        if d["applicable"]:
                            for side, q in zip(("vol", "eff"), C.MARKET_QUANTITIES[m]):
                                if q is not None:
                                    rates.append(dict(threshold=t, season=tg.season, week=tg.week, game_id=tg.game_id, team=tg.team,
                                                      player_id=tg.player_id, market=m, search=s, side=side, quantity=q,
                                                      matched=not (d["nomatch"] if side == "vol" else d["nomatch_eff"])))
                    for (mk, side, q) in sm[fam]:
                        if mk != m:
                            continue
                        tz = zl.get(q, {}).get((tg.game_id, tg.player_id if is_player else tg.team))
                        if tz is None or not np.isfinite(tz):
                            continue
                        j = 1 if side == "vol" else 2
                        for s, d in ud.items():
                            nm = d["nomatch"] if side == "vol" else d["nomatch_eff"]
                            if nm or s not in sets:
                                continue
                            w = caps[side][s]
                            ok = (w > 0) & np.isfinite(sets[s][j])
                            n_eff = d["n_eff_vol"] if side == "vol" else d["n_eff_eff"]
                            real = d["shift_vol"] if side == "vol" else d["shift_eff"]
                            cz = zall(tgr.market, unit, q)[res0[s].summary["obs_pos"]]
                            cz = cz[np.isfinite(cz)]
                            rand = [random_shift(w[ok], n_eff, cz, rng) for _ in range(n_draws)]
                            rows.append(dict(threshold=t, season=tg.season, week=tg.week, game_id=tg.game_id, team=tg.team, player_id=tg.player_id,
                                             market=m, search=s, parent=C.parent(s), side=side, quantity=q, n_matches=int(ok.sum()), n_eff=float(n_eff), z=float(tz),
                                             shift=float(real), sq_error_real=float((real - tz) ** 2), sq_error_zero=float(tz ** 2),
                                             sq_error_random=float(np.mean([(r - tz) ** 2 for r in rand]))))
    schema = {"threshold": pl.Float64, "season": pl.Int64, "week": pl.Int64, "game_id": pl.String, "team": pl.String, "player_id": pl.String,
              "market": pl.String, "search": pl.String}
    rows_df = pl.DataFrame(rows, schema=schema | {"parent": pl.String, "side": pl.String, "quantity": pl.String, "n_matches": pl.Int64, "n_eff": pl.Float64, "z": pl.Float64,
                                                  "shift": pl.Float64, "sq_error_real": pl.Float64, "sq_error_zero": pl.Float64,
                                                  "sq_error_random": pl.Float64})
    return rows_df, pl.DataFrame(rates, schema=schema | {"side": pl.String, "quantity": pl.String, "matched": pl.Boolean})


def summarize(rows: pl.DataFrame, thresholds=THRESHOLDS) -> pl.DataFrame:
    """Per threshold: the pooled improvement over random pairing (random - real squared error) and over a zero shift (z^2 - real), each with its 95%
    season-week cluster bootstrap interval, the mean errors and the number of rows."""
    out = []
    for t in thresholds:
        r = rows.filter(pl.col("threshold") == t).sort("season", "week", "game_id", "team", "player_id", "market", "search", "side", nulls_last=True)
        row = dict(threshold=t, n_rows=r.height)
        if r.height == 0:
            out.append(row)
            continue
        cl = (r["season"] * 100 + r["week"]).to_numpy()
        for suffix, col in (("", "sq_error_random"), ("_zero", "sq_error_zero")):
            d = (r[col] - r["sq_error_real"]).to_numpy()
            lo, hi = compare.cluster_bootstrap(d, cl)
            row.update({f"improvement{suffix}": float(d.mean()), f"ci_lo{suffix}": float(lo), f"ci_hi{suffix}": float(hi)})
        row.update(real=float(r["sq_error_real"].mean()), random=float(r["sq_error_random"].mean()), zero=float(r["sq_error_zero"].mean()))
        out.append(row)
    cols = ["improvement", "ci_lo", "ci_hi", "improvement_zero", "ci_lo_zero", "ci_hi_zero", "real", "random", "zero"]
    return pl.DataFrame(out, schema={"threshold": pl.Float64, "n_rows": pl.Int64, **{c: pl.Float64 for c in cols}})


def choose_threshold(summary: pl.DataFrame, default: float = DEFAULT_THRESHOLD) -> float:
    """The lowest threshold whose pooled comps beat BOTH random pairing and a zero shift, each improvement above zero with its interval's lower bound
    above zero (rule tightened on 2026-10-09, before the per-unit sweep was read); none: the default (and the comps get weight 0 for V1)."""
    ok = summary.filter((pl.col("improvement") > 0) & (pl.col("ci_lo") > 0) & (pl.col("improvement_zero") > 0) & (pl.col("ci_lo_zero") > 0)).sort("threshold")
    return float(ok["threshold"][0]) if ok.height else default


def breakdown(rows: pl.DataFrame, rates: pl.DataFrame) -> pl.DataFrame:
    """For reading only: per threshold, market, search and side the match rate, the mean improvement and the scored rows."""
    imp = (rows.group_by("threshold", "search", "market", "side")
           .agg(n_rows=pl.len(), improvement=(pl.col("sq_error_random") - pl.col("sq_error_real")).mean(), mean_n_eff=pl.col("n_eff").mean()))
    rate = rates.group_by("threshold", "market", "search", "side").agg(match_rate=pl.col("matched").mean(), n_targets=pl.len())
    return rate.join(imp, on=["threshold", "market", "search", "side"], how="left").sort("threshold", "market", "search", "side")


# ---------------------------------------------------------------------------------------------------------------- the unit diagnostic (reported only)
def diagnostic_cells() -> list:
    """Every (unit search, side, quantity) the diagnostic tests, from the spec alone (fixed before any result): per family, each quantity once (side_markets)
    with the unit searches of its market. Its length is the Bonferroni m."""
    out = []
    for fam, rows in side_markets().items():
        for m, side, q in rows:
            for uid, _, _ in C.unit_searches(m, fam != "game"):
                out.append((uid, side, q))
    return sorted(set(out))


def unit_diagnostic(rows: pl.DataFrame, rates: pl.DataFrame, m_cells: int | None = None) -> pl.DataFrame:
    """Per (unit search, side, quantity) cell, over the rows of one threshold: the match rate, the improvement over a zero shift (z^2 - (shift - z)^2)
    and over random pairing, each with a season-week interval at level 1 - 0.05 / m (Bonferroni, m = the number of cells, counted from the spec) and
    the seasons it is positive in. WORKS: both improvements positive, both intervals above zero, both positive in >= MIN_SEASONS_WON seasons; EDGE:
    both positive but a condition fails; NO: otherwise. Reported, not used to drop units (docs/overnight_plan.md)."""
    cells = diagnostic_cells()
    m_cells = m_cells or len(cells)
    level = 1 - 0.05 / m_cells
    # each quantity once, from the same market as the scored rows (side_markets)
    first = {(q, fam): m for fam, rs in side_markets().items() for m, _, q in rs}
    keep = pl.DataFrame([dict(market=m, quantity=q) for (q, _), m in first.items()])
    rate = rates.join(keep, on=["market", "quantity"], how="semi").group_by("search", "side", "quantity").agg(match_rate=pl.col("matched").mean(), n_targets=pl.len())
    out = []
    for uid, side, q in cells:
        r = rows.filter((pl.col("search") == uid) & (pl.col("side") == side) & (pl.col("quantity") == q)).sort(
            "season", "week", "game_id", "team", "player_id", "market", nulls_last=True)
        row = dict(search=uid, parent=C.parent(uid), side=side, quantity=q, n_rows=r.height)
        ok_all, pos_all = r.height > 0, r.height > 0
        for name, col in (("zero", "sq_error_zero"), ("random", "sq_error_random")):
            if r.height == 0:
                row.update({f"improvement_{name}": None, f"ci_lo_{name}": None, f"ci_hi_{name}": None, f"seasons_won_{name}": 0})
                continue
            d = (r[col] - r["sq_error_real"]).to_numpy()
            lo, hi = compare.cluster_bootstrap(d, (r["season"] * 100 + r["week"]).to_numpy(), level=level)
            seasons = r["season"].to_numpy()
            won = sum(float(d[seasons == s].mean()) > 0 for s in sorted(set(seasons.tolist())))
            row.update({f"improvement_{name}": float(d.mean()), f"ci_lo_{name}": float(lo), f"ci_hi_{name}": float(hi), f"seasons_won_{name}": int(won)})
            pos_all &= d.mean() > 0
            ok_all &= d.mean() > 0 and lo > 0 and won >= config.MIN_SEASONS_WON
        row["label"] = "WORKS" if ok_all else ("EDGE" if pos_all else "NO")
        out.append(row)
    diag = pl.DataFrame(out, schema={"search": pl.String, "parent": pl.String, "side": pl.String, "quantity": pl.String, "n_rows": pl.Int64,
                                     "improvement_zero": pl.Float64, "ci_lo_zero": pl.Float64, "ci_hi_zero": pl.Float64, "seasons_won_zero": pl.Int64,
                                     "improvement_random": pl.Float64, "ci_lo_random": pl.Float64, "ci_hi_random": pl.Float64,
                                     "seasons_won_random": pl.Int64, "label": pl.String})
    return diag.join(rate, on=["search", "side", "quantity"], how="left").with_columns(bonferroni_m=pl.lit(m_cells), interval_level=pl.lit(level))
