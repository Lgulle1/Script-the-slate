"""Print real comparable searches as readable cards (build plan 4c, manual step): read them like a human.

  python print_comp_cards.py [--window recency_weighted] [--top 5] [--auto]
  python print_comp_cards.py --target MARKET GAME_ID TEAM [PLAYER_ID] ...

Each card shows the target, then for every search its live verdict (the no-match rule at SIM_THRESHOLD / MIN_NEFF), the closest past games whatever
their similarity (per-unit similarity, combined similarity, recency, continuity, quality, final weight, and the comp-free z of the outcome when the
walk-forward expectation exists), how the retrieval changes on the healthy target vector (overlap of the top 10, change in the live shifts) and the
live volume / efficiency shifts. --auto picks five 2022-2024 targets: a game side, a back whose team's lineup-adjusted rushing vector moved most from
the healthy one, a receiver, a quarterback, and a target whose S1 has nothing above the threshold. Reads only stored vectors and 2020-2024 results;
2025 is never loaded.
"""
import argparse

import duckdb
import numpy as np
import polars as pl

import config
from models import comps as C
from models import comps_spec as cs


def names(raw_db=config.RAW_DUCKDB_PATH) -> dict:
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        d = con.execute("SELECT gsis_id, full_name FROM rosters_weekly WHERE gsis_id IS NOT NULL AND season <= ? ORDER BY season, week, pulled_at",
                        [max(config.BACKTEST_SEASONS)]).pl()
    finally:
        con.close()
    return dict(zip(d["gsis_id"].to_list(), d["full_name"].to_list()))


def auto_targets(pool: C.Pool, targets: list, live: pl.DataFrame | None) -> list:
    """Five illustrative 2022-2024 targets (see the module doc), chosen by fixed rules, not by their shifts."""
    late = [t for t in targets if t.season >= 2022]
    pick = []
    team = [t for t in late if t.player_id is None]
    pick.append(team[len(team) // 2])
    rush = [t for t in late if t.market == "rush_att"]
    def moved(t):
        g = pool.idx[(t.game_id, t.team)]
        a, h = pool.T["run_offense"]["base"][0][g], pool.TH["run_offense"]["base"][0][g]
        return float(np.nansum(np.abs(a - h)))
    pick.append(max(rush, key=lambda t: (moved(t), t.game_id, t.player_id)))
    rec = [t for t in late if t.market == "targets"]
    pick.append(rec[len(rec) // 3])
    qb = [t for t in late if t.market == "pass_att"]
    pick.append(qb[len(qb) // 2])
    if live is not None:
        nm = live.filter((pl.col("search") == "S1") & (pl.col("reason") == "best_similarity_below_threshold") & (pl.col("season") >= 2022)
                         & pl.col("player_id").is_not_null()).sort("best_similarity").row(0, named=True)
        pick.append(next(t for t in late if (t.game_id, t.team, t.player_id) == (nm["game_id"], nm["team"], nm["player_id"])))
    return pick


def card(pool: C.Pool, tg: C.Target, zl: dict, who: dict, top: int):
    nm = lambda p: who.get(p, p) if p else ""
    print("=" * 150)
    print(f"{tg.market} | {tg.season} week {tg.week} | {tg.game_id} | {tg.team} vs {tg.opponent}" + (f" | {nm(tg.player_id)} ({tg.player_id})" if tg.player_id else ""))
    print(f"units: {', '.join(cs.MARKET_UNITS[tg.market])}")
    pool.clear_cache()
    feats, _, detail, retr = C.comp_shifts(pool, [tg], zl, keep_matches=False)
    disp = pool.search(tg, C.SEARCHES, keep=True, sim_threshold=0.0, min_neff=0.0)
    qv, qe = C.MARKET_QUANTITIES[tg.market]
    f = feats.filter(pl.col("market") == tg.market).row(0, named=True)
    for s in C.SEARCHES:
        d = detail.filter((pl.col("market") == tg.market) & (pl.col("search") == s)).row(0, named=True)
        rc = retr.filter((pl.col("market") == tg.market) & (pl.col("search") == s)).row(0, named=True)
        verdict = "not applicable" if not d["applicable"] else ("NO MATCH (" + str(d["reason"]) + ")" if d["nomatch"] else "match")
        print(f"\n  {s}: {verdict} | best similarity {f[f'best_sim_{s}']:.3f} | matches {d['n_matches']} | n_eff {d['n_eff_similarity']:.2f} "
              f"| shift vol {f[f'shift_vol_{s}']:+.3f}" + (f" eff {f[f'shift_eff_{s}']:+.3f}" if qe else "")
              + f" | healthy vs adjusted: differs={rc['adjusted_differs']} overlap top-10 {rc['overlap_top_k']}, shift change {rc['shift_vol_change']:+.3f}")
        m = disp[s].matches
        if not d["applicable"] or m is None or m.height == 0:
            continue
        ent = m["obs_player_id"].to_list() if tg.player_id else m["obs_team"].to_list()
        m = m.head(top).with_columns(
            z_vol=pl.Series([zl.get(qv, {}).get(k) for k in zip(m["obs_game_id"].to_list()[:top], ent[:top])], dtype=pl.Float64),
            z_eff=pl.Series([zl.get(qe, {}).get(k) if qe else None for k in zip(m["obs_game_id"].to_list()[:top], ent[:top])], dtype=pl.Float64),
            who=pl.Series([nm(p) for p in m["obs_player_id"].to_list()[:top]], dtype=pl.String))
        sims = [c for c in m.columns if c.startswith("sim_") and c not in ("sim_combined", "sim_offense", "sim_defense")]
        with pl.Config(tbl_rows=top, tbl_cols=30, tbl_width_chars=250, fmt_str_lengths=22, tbl_hide_dataframe_shape=True, float_precision=3):
            print(m.select("obs_game_id", "obs_team", "who", *sims, "sim_combined", "recency_weight", "continuity_weight", "quality_weight", "final_weight",
                           "z_vol", "z_eff"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default=C.LOG_WINDOW)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--target", nargs="+", action="append", default=[], help="MARKET GAME_ID TEAM [PLAYER_ID]")
    a = ap.parse_args()
    pool = C.load_pool(a.window)
    zl = C.z_lookup(C.standardized_residuals(pl.read_parquet(config.PROCESSED_DIR / "walkforward_predictions.parquet")))
    who = names()
    targets = C.backtest_targets()
    chosen = []
    if a.auto:
        p = config.PROCESSED_DIR / "comp_search_summary.parquet"
        chosen += auto_targets(pool, targets, pl.read_parquet(p) if p.exists() else None)
    for spec in a.target:
        market, gid, team, *pid = spec
        g = pool.games.filter((pl.col("game_id") == gid) & (pl.col("team") == team)).row(0, named=True)
        chosen.append(C.Target(market, gid, team, g["opponent"], g["season"], g["week"], pid[0] if pid else None))
    for tg in chosen:
        card(pool, tg, zl, who, a.top)
    print(f"\nholdout {config.HOLDOUT_SEASON} untouched")
