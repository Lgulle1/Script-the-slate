"""Comp chains (comps v3): for every player-game prop, which past situations are relevant tonight, and how did they go.

Similarity follows the spec literally: cosine similarity of standardized, weighted unit vectors; only comps with similarity >= MIN_SIM; the TOP_K best by
(similarity x recency x context); fewer than MIN_COMPS -> "no reliable comp" (NaN). The chain's number is the weighted mean of the comps' deltas:
    matchup_effect = sum(w * delta) / sum(w),  w = similarity * recency * context
delta is how each comp game went against ITS OWN baseline (never the raw stat): vs our projection (relc) or vs the book's line (rline).

Vectors per market (sub-unit weights; inside a sub-unit every stat counts the same):
  rushing: offense run 0.75 + pace 0.25 / defense run;  receiving WR: pass 0.6 + wr 0.4 (both sides);  TE: pass 0.6 + te 0.4;  RB receiving: pass 0.6 + rb_rec 0.4;
  passing: offense pass 0.8 + pace 0.2 / defense pass.
Profiles are lineup-aware (comps3.profile): tonight's offense and defense are built from the games tonight's players played.

Chains for player p (team T, opponent D, game g):
  own   p's own past games vs defenses like D tonight; context = same QB as tonight (passing / receiving 1.0 vs 0.35, rushing 1.0 vs 0.85) x same team (1.0 vs 0.7)
  dfn   D's past games vs players like p in offenses like T; context = how many of TONIGHT's defenders played that game (lineup overlap)
  team  T's past games (same market and family, any player) vs defenses like D; context = how many of TONIGHT's T players played that game
  int   league-wide: offense like T, defense like D and player like p all at once
  sit   role change / injury (comps2.situations)
"""
from __future__ import annotations

import numpy as np
import polars as pl

import config
from comps2 import archetypes as AR
from comps2 import situations as SI
from comps3 import profile as PR
from comps3 import units as U

MIN_SIM, TOP_K, MIN_COMPS = 0.70, 8, 3
FLOOR_Q = 0.90   # a comp must be more similar than this share of all possible pairings (same chain, market, family): "similar relative to the other comps"
WINDOW_DAYS, HL_DAYS = 730, 365.0
REL_LO, REL_HI = -1.0, 2.0
OUT = config.PROCESSED_DIR / "comps3_chains.parquet"

GROUP = dict(rush_att="rush", rush_yds="rush", qb_rush_att="rush", qb_rush_yds="rush", targets="rec", rec="rec", rec_yds="rec",
             pass_att="pass", pass_cmp="pass", pass_yds="pass")


def vec_spec(market, family):
    g = GROUP[market]
    if g == "rush":
        return [("run", 0.75), ("pace", 0.25)], [("run", 1.0)], "run"
    if g == "pass":
        return [("pass", 0.8), ("pace", 0.2)], [("pass", 1.0)], "pass"
    sub = {"WR": "wr", "TE": "te", "RB": "rb_rec"}.get(family, "wr")
    return [("pass", 0.6), (sub, 0.4)], [("pass", 0.6), (sub, 0.4)], "pass"


def frame() -> pl.DataFrame:
    e = pl.read_parquet(config.PROCESSED_DIR / "comps2_expectations.parquet").filter(pl.col("market").is_in(list(GROUP)))
    e = e.with_columns(relc=pl.col("rel").clip(REL_LO, REL_HI), gd=pl.col("gameday").cast(pl.Date).cast(pl.Int32))
    sel = pl.read_parquet(config.PROCESSED_DIR / "line_study_sel.parquet").filter((pl.col("side") == "over") & (pl.col("tier") != "late"))
    sel = sel.unique(["game_id", "player_id", "market"], keep="first").select("game_id", "player_id", "market", "line")
    e = e.join(sel, on=["game_id", "player_id", "market"], how="left").with_columns(rline=(pl.col("actual") / pl.col("line") - 1).clip(REL_LO, REL_HI))
    prof = pl.read_parquet(PR.OUT)
    off = prof.filter(pl.col("side") == "off").select("game_id", "team", *[pl.col(s).alias(f"o_{s}") for s in U.STATS])
    dfn = prof.filter(pl.col("side") == "def").select("game_id", pl.col("team").alias("opponent"), *[pl.col(s).alias(f"d_{s}") for s in U.STATS])
    arc = pl.read_parquet(config.PROCESSED_DIR / "comps2_archetypes.parquet").select("game_id", "player_id", *[pl.col(s).alias(f"a_{s}") for s in AR.ALL])
    from live import history as H
    log = H.read_played_logs(max(config.BACKTEST_SEASONS) + 2)[0]
    # the announced starting QB of each team-game (schedules; known before kickoff), not who ended up throwing the most
    import duckdb
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sq = con.execute("""SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, home_team, away_team, home_qb_id, away_qb_id FROM schedules
                        WHERE game_type = 'REG' ORDER BY game_id, pulled_at DESC)""").pl()
    con.close()
    qb = pl.concat([sq.select("game_id", team="home_team", qb_id="home_qb_id"), sq.select("game_id", team="away_team", qb_id="away_qb_id")])
    e = e.join(off, on=["game_id", "team"], how="left").join(dfn, on=["game_id", "opponent"], how="left").join(arc, on=["game_id", "player_id"], how="left")
    e = e.join(qb, on=["game_id", "team"], how="left")
    return e.sort("gd", "game_id", "player_id", "market").with_row_index("rid")


def _vecs(x: pl.DataFrame, spec, prefix):
    cols, wts = [], []
    for unit, w in spec:
        st = U.UNITS[unit]
        for s in st:
            c = f"{prefix}_{s}"
            if c in cols:          # a stat shared by two sub-units (neutral_pass, tgt_wr) counts once, with the larger weight
                i = cols.index(c)
                wts[i] = max(wts[i], np.sqrt(w / len(st)))
                continue
            cols.append(c)
            wts.append(np.sqrt(w / len(st)))
    V = np.nan_to_num(x.select(cols).to_numpy().astype(float)) * np.array(wts)[None, :]
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.where(n > 0, n, 1.0)


def _arch(x: pl.DataFrame, family):
    cols = [f"a_{s}" for s in AR.FEATS.get(family, ())]
    if not cols:
        return np.zeros((x.height, 1))
    V = np.nan_to_num(x.select(cols).to_numpy().astype(float))
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.where(n > 0, n, 1.0)


def floors(VO, VD, VA, seed=7, n=3000):
    """The FLOOR_Q quantile of each chain's similarity over random pairs of rows (the bar a comp must clear), capped at MIN_SIM."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(VO), n), rng.integers(0, len(VO), n)
    so, sd, sp = (VO[i] * VO[j]).sum(1), (VD[i] * VD[j]).sum(1), (VA[i] * VA[j]).sum(1)
    q = lambda v: float(min(MIN_SIM, np.quantile(v, FLOOR_Q)))
    return dict(own=q(sd), team=q(sd), dfn=q(np.sqrt(np.clip(so, 0, None) * np.clip(sp, 0, None))),
                int=q(np.cbrt(np.clip(so, 0, None) * np.clip(sd, 0, None) * np.clip(sp, 0, None))))


def _pick(sim, w, delta, floor=MIN_SIM):
    """Similarity floor, TOP_K by weight, MIN_COMPS or NaN. Returns (effect, n, top indices)."""
    ok = (sim >= floor) & ~np.isnan(delta) & (w > 0)
    idx = np.flatnonzero(ok)
    if len(idx) < MIN_COMPS:
        return np.nan, len(idx), idx
    idx = idx[np.argsort(-w[idx])[:TOP_K]]
    return float((w[idx] * delta[idx]).sum() / w[idx].sum()), len(idx), idx


def run(x: pl.DataFrame, targets: np.ndarray, outcome: str = "relc", overlaps: pl.DataFrame | None = None) -> pl.DataFrame:
    overlaps = overlaps if overlaps is not None else pl.read_parquet(PR.OVERLAP_OUT)
    ovd = {}
    for g, t, sd, pg, r, p in overlaps.iter_rows():
        ovd[(g, t, sd, pg)] = (r, p)
    n = x.height
    res = {f"{c}_{k}": np.full(n, np.nan) for c in ("own", "dfn", "team", "int") for k in ("eff", "n")}
    delta = x[outcome].to_numpy().astype(float)
    gd, mk, fam = x["gd"].to_numpy(), x["market"].to_numpy(), x["family"].to_numpy()
    pid, team, opp, gid, qb = (x[c].to_numpy() for c in ("player_id", "team", "opponent", "game_id", "qb_id"))
    tset = np.zeros(n, bool)
    tset[targets] = True
    for m in np.unique(mk):
        for f_ in np.unique(fam[mk == m]):
            ix = np.flatnonzero((mk == m) & (fam == f_))
            sub = x[ix]
            ospec, dspec, kind = vec_spec(m, f_)
            VO, VD, VA = _vecs(sub, ospec, "o"), _vecs(sub, dspec, "d"), _arch(sub, f_)
            FL = floors(VO, VD, VA)
            k_ov = 0 if kind == "run" else 1
            qb_miss = 0.85 if GROUP[m] == "rush" else 0.35
            g_ = gd[ix]
            by_player, by_opp, by_team = _index(pid[ix]), _index(opp[ix]), _index(team[ix])
            for a in np.flatnonzero(tset[ix]):
                r = ix[a]
                live = (g_ < g_[a]) & (g_ >= g_[a] - WINDOW_DAYS)
                rec_all = 0.5 ** ((g_[a] - g_) / HL_DAYS)
                # own
                c = by_player.get(pid[r], np.array([], int))
                c = c[live[c]]
                if len(c):
                    sim = VD[c] @ VD[a]
                    ctx = np.where(qb[ix[c]] == qb[r], 1.0, qb_miss) * np.where(team[ix[c]] == team[r], 1.0, 0.7)
                    res["own_eff"][r], res["own_n"][r], _ = _pick(sim, sim * rec_all[c] * ctx, delta[ix[c]], FL["own"])
                # dfn: tonight's defense, weighted by how many of tonight's defenders played that game
                c = by_opp.get(opp[r], np.array([], int))
                c = c[live[c]]
                if len(c):
                    so, sp = VO[c] @ VO[a], VA[c] @ VA[a]
                    sim = np.sqrt(np.clip(so, 0, None) * np.clip(sp, 0, None))
                    ov = np.array([ovd.get((gid[r], opp[r], "def", gid[ix[j]]), (0.0, 0.0))[k_ov] for j in c])
                    res["dfn_eff"][r], res["dfn_n"][r], _ = _pick(sim, sim * rec_all[c] * ov, delta[ix[c]], FL["dfn"])
                # team: tonight's offense (tonight's players) vs similar defenses
                c = by_team.get(team[r], np.array([], int))
                c = c[live[c]]
                if len(c):
                    sim = VD[c] @ VD[a]
                    ov = np.array([ovd.get((gid[r], team[r], "off", gid[ix[j]]), (0.0, 0.0))[k_ov] for j in c])
                    res["team_eff"][r], res["team_n"][r], _ = _pick(sim, sim * rec_all[c] * ov, delta[ix[c]], FL["team"])
            # int: league-wide triples, one block per gameday
            for day in np.unique(g_[tset[ix]]):
                tr = np.flatnonzero((g_ == day) & tset[ix])
                c = np.flatnonzero((g_ < day) & (g_ >= day - WINDOW_DAYS) & ~np.isnan(delta[ix]))
                if not len(c):
                    continue
                so, sd_, sp = VO[tr] @ VO[c].T, VD[tr] @ VD[c].T, VA[tr] @ VA[c].T
                sim = np.cbrt(np.clip(so, 0, None) * np.clip(sd_, 0, None) * np.clip(sp, 0, None))
                rec = 0.5 ** ((day - g_[c]) / HL_DAYS)
                for t_, a in enumerate(tr):
                    res["int_eff"][ix[a]], res["int_n"][ix[a]], _ = _pick(sim[t_], sim[t_] * rec, delta[ix[c]], FL["int"])
    return x.select("rid", "season", "week", "game_id", "player_id", "team", "opponent", "family", "market", "expected", "actual", "relc", "line", "rline").with_columns(
        [pl.Series(k, v) for k, v in res.items()])


def _index(keys):
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    cuts = np.flatnonzero(k[1:] != k[:-1]) + 1
    return {keys[s[0]]: s for s in np.split(order, cuts)} if len(k) else {}


def with_situation(c: pl.DataFrame, x: pl.DataFrame, targets, outcome="relc") -> pl.DataFrame:
    xx = x.with_columns(relc=pl.col(outcome)) if outcome != "relc" else x
    xx = xx.filter(pl.col("relc").is_not_null())
    s = SI.search(xx, pl.read_parquet(SI.OUT), np.flatnonzero(np.isin(xx["rid"].to_numpy(), targets)), h=0.5)
    c = c.join(s.select("rid", sit_eff="sit_shift"), on="rid", how="left")
    eff = c.select("own_eff", "dfn_eff", "team_eff", "int_eff", "sit_eff").to_numpy()
    mean = np.nanmean(np.where(np.isnan(eff), np.nan, eff), axis=1) if eff.size else eff
    with np.errstate(invalid="ignore"):
        agree = np.nansum(np.sign(eff) == np.sign(mean)[:, None], 1)
    nch = (~np.isnan(eff)).sum(1)
    return c.with_columns(chains_mean=pl.Series(mean), agree=pl.Series(agree.astype(float)), n_chains=pl.Series(nch.astype(float)))
