"""Lineup-aware unit profiles: tonight's unit, built from the games its players actually played together (continuity + injury-adjusted fingerprint).

For team T and game g, the LINEUP is the players on the field for g (snap share >= ON_FIELD; the inactive list and the starters are known at kickoff).
Each player's EXPECTED share is his average snap share over his previous games for T (EXPECT_GAMES of them; a newcomer counts DEFAULT_SHARE), so the
size of tonight's role never leaks in. Every earlier game h of T gets a lineup overlap per unit:
    overlap(h) = sum_p w(pos p) * min(share_h(p), expected_p) / sum_p w(pos p) * expected_p      (p in tonight's lineup)
with position weights per unit (POS_W: for the pass offense the QB weighs 4, for the run defense the front seven, ...). A Penix pass-offense profile is
then built almost only from Penix games; a Saints run defense missing three front-seven starters is built from the games without them.
The profile is sum_h r(h) * overlap(h)^GAMMA * num_h / sum_h r(h) * overlap(h)^GAMMA * den_h over the unit's stats, r(h) the recency weight
(half-life HALF_LIFE team games, SEASON_GAP extra games per season boundary, LOOKBACK seasons). Then z per season and week (scale only).
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from comps3 import units as U

OUT = config.PROCESSED_DIR / "comps3_profiles.parquet"
OVERLAP_OUT = config.PROCESSED_DIR / "comps3_overlaps.parquet"     # per (game, team, side, earlier game): lineup overlap, run and pass kinds
ON_FIELD = 0.20
EXPECT_GAMES = 3
DEFAULT_SHARE = 0.5
STARTER, STARTER_SHARE = 0.60, 0.80
HALF_LIFE, SEASON_GAP, LOOKBACK, GAMMA = 8.0, 4.0, 3, 1.5
GROUP = {**{p: "OL" for p in ("T", "G", "C", "OL", "OT", "C/G", "G/T", "G/C")}, "QB": "QB", **{p: "RB" for p in ("RB", "FB", "HB")}, "WR": "WR", "TE": "TE",
         **{p: "DL" for p in ("DE", "DT", "NT", "DL", "DT/D")}, **{p: "LB" for p in ("LB", "OLB", "ILB")}, **{p: "DB" for p in ("CB", "S", "FS", "SS", "DB")}}
POS_W = {  # (side, unit kind) -> weight per position group
    ("off", "run"): dict(OL=1.0, RB=1.0, TE=0.6, QB=0.6, WR=0.2),
    ("off", "pass"): dict(QB=4.0, OL=0.6, WR=1.0, TE=0.8, RB=0.4),
    ("def", "run"): dict(DL=1.0, LB=1.0, DB=0.4),
    ("def", "pass"): dict(DB=1.0, DL=0.7, LB=0.6),
}


def snaps() -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    s = con.execute("""SELECT * FROM (SELECT DISTINCT ON (game_id, pfr_player_id) game_id, season, week, pfr_player_id AS pid, position, team, offense_pct, defense_pct
                       FROM snap_counts WHERE game_type = 'REG' ORDER BY game_id, pfr_player_id, pulled_at DESC)""").pl()
    con.close()
    fix = lambda c: pl.when(pl.col(c) > 1.0).then(pl.col(c) / 100).otherwise(pl.col(c)).fill_null(0.0)
    s = s.with_columns(off=fix("offense_pct"), dfn=fix("defense_pct"), grp=pl.col("position").replace_strict(GROUP, default=None))
    return s.filter(pl.col("grp").is_not_null())


def profiles(g: pl.DataFrame, s: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    g = g.join(sched.select("game_id", "gameday", "season"), on="game_id", how="left")
    rows, ovl = [], []
    for (team, side), t in g.sort("gameday").group_by("team", "side", maintain_order=True):
        games = t["game_id"].to_list()
        seasons = t["season"].to_numpy().astype(int)
        pos = np.arange(len(games)) + SEASON_GAP * (seasons - seasons.min())
        col = "off" if side == "off" else "dfn"
        st = s.filter(pl.col("team") == team).select("game_id", "pid", "grp", col)
        pids = st["pid"].unique().to_list()
        pix = {p: i for i, p in enumerate(pids)}
        gix = {gm: i for i, gm in enumerate(games)}
        M = np.zeros((len(games), len(pids)))
        grp = {}
        for gm, pid, gr, v in st.iter_rows():
            if gm in gix:
                M[gix[gm], pix[pid]] = v
                grp[pid] = gr
        gw = {k: np.array([POS_W[(side, k)].get(grp.get(p), 0.0) for p in pids]) for k in ("run", "pass")}
        N = {k: t.select([f"{x}_n" for x in U.STATS]).to_numpy() for k in ("x",)}["x"]
        D = t.select([f"{x}_d" for x in U.STATS]).to_numpy()
        kind_of_stat = np.array([("run" if x in U.UNITS["run"] and x not in U.UNITS["pass"] else "pass") for x in U.STATS])
        for i in range(len(games)):
            lo = np.flatnonzero(seasons >= seasons[i] - LOOKBACK)
            prev = lo[lo < i]
            if not len(prev):
                rows.append((games[i], team, side, 0.0, 0.0, *[np.nan] * len(U.STATS)))
                continue
            on = M[i] >= ON_FIELD
            played = M[:i] > 0
            # expected share: mean over his last EXPECT_GAMES appearances before g
            exp = np.full(M.shape[1], DEFAULT_SHARE)
            for j in np.flatnonzero(on):
                hist = M[:i, j][played[:, j]]
                if len(hist):
                    exp[j] = hist[-EXPECT_GAMES:].mean()
                # a starter tonight (known at kickoff from the depth chart / inactives) counts as a starter even if his history is backup snaps:
                # Penix's first start, Allen with Hall out
                if M[i, j] >= STARTER:
                    exp[j] = max(exp[j], STARTER_SHARE)
            rec = 0.5 ** ((pos[i] - pos[prev]) / HALF_LIFE)
            num = np.zeros(len(U.STATS))
            den = np.zeros(len(U.STATS))
            ov_out, ov_k = {}, {}
            for k in ("run", "pass"):
                w = gw[k] * on * exp
                tot = w.sum()
                ov = (np.minimum(M[prev], exp[None, :]) * (gw[k] * on)[None, :]).sum(1) / tot if tot > 0 else np.ones(len(prev))
                ov_out[k] = float((rec * ov).sum() / rec.sum())
                ov_k[k] = ov
                wt = rec * np.clip(ov, 0, 1) ** GAMMA
                cols = kind_of_stat == k
                num[cols] = (wt[:, None] * N[prev][:, cols]).sum(0)
                den[cols] = (wt[:, None] * D[prev][:, cols]).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                fp = np.where(den > 0, num / den, np.nan)
            rows.append((games[i], team, side, ov_out["run"], ov_out["pass"], *fp))
            ovl.extend((games[i], team, side, games[h], float(ov_k["run"][a]), float(ov_k["pass"][a])) for a, h in enumerate(prev))
    a = pl.DataFrame(rows, schema=["game_id", "team", "side", "continuity_run", "continuity_pass", *U.STATS], orient="row", infer_schema_length=None)
    a = a.join(sched, on="game_id", how="left")
    pl.DataFrame(ovl, schema=["game_id", "team", "side", "prev_game_id", "ov_run", "ov_pass"], orient="row").write_parquet(OVERLAP_OUT)
    a = a.with_columns([pl.col(x).fill_nan(None) for x in U.STATS])
    return a.with_columns([((pl.col(x) - pl.col(x).mean().over("season", "week")) / pl.col(x).std().over("season", "week")).alias(x) for x in U.STATS])


def build() -> pl.DataFrame:
    g = pl.read_parquet(U.OUT)
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, season, week, gameday FROM schedules WHERE game_type = 'REG' ORDER BY game_id, pulled_at DESC)").pl()
    con.close()
    sched = sched.with_columns(pl.col("gameday").cast(pl.Utf8).str.slice(0, 10).str.to_date(strict=False), season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32))
    return profiles(g.drop("season", "week"), snaps(), sched)


if __name__ == "__main__":
    import time
    t0 = time.time()
    a = build()
    a.write_parquet(OUT)
    print(a.shape, f"{time.time() - t0:.0f}s")
    print(a.group_by("side").agg(pl.col("continuity_run", "continuity_pass").mean(), pl.col("continuity_pass").quantile(0.1).alias("pass_p10")))
