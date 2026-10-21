"""Role-change engine (2026-10-20): when a player's role changes, estimate his NEW role before kickoff from the cases most like it, and find where
FanDuel's line hasn't caught up (the Coker / Braelon Allen idea).

Per player-game, from information known before kickoff only (the week's roster and final injury report via comps2.pregame, the announced starting QBs
from the schedule, and usage in EARLIER games):
  own_car / own_tgt      his carry share / target share in the games he played among the team's last ROLE_GAMES games
  played_recent          how many of the team's last 4 games he played;  since_last  team games since he last played (a return from injury)
  vac_car / vac_tgt      share held by teammates who were regulars and are OUT for this game (same role)
  rank_car / rank_tgt    his rank by own share among the AVAILABLE players of the role; comp_car / comp_tgt  the share held by available others
  back_car / back_tgt    share held by teammates who are BACK this game after missing the last one (demotion pressure)
  vac_nth                consecutive team games with a vacancy of 15%+ in his role (1 = first game of the change)
  qb_change              his team's announced starter differs from last game's starter;  new_team  first games for this team
Target: his actual share this game. Model: deterministic gradient boosting, each 6-week block fitted only on player-games before it (2018 on).
Then a role-change projection: predicted share x the team's expected volume x his efficiency, per market, compared with FanDuel's line.

  python -m mark4.rolechange      # writes data/processed/rolechange.parquet and prints the forward tests
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

import config

OUT = config.PROCESSED_DIR / "rolechange.parquet"
ROLE_GAMES = 6
REG_CAR, REG_TGT = 0.15, 0.10           # a regular holds at least this share when he plays
VAC_EVENT = 0.15
BLOCKS = (1, 7, 13)
FIRST_SEASON = 2018
FEATS = ["own_car", "own_tgt", "played_recent", "since_last", "vac_car", "vac_tgt", "rank_car", "rank_tgt", "comp_car", "comp_tgt",
         "back_car", "back_tgt", "vac_nth", "qb_change", "new_team", "is_rb", "is_wr", "is_te", "team_car_pg", "team_tgt_pg"]


def usage() -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    ps = con.execute("""SELECT * FROM (SELECT DISTINCT ON (player_id, season, week) player_id, season, week, team, position, carries, targets, receptions,
                        rushing_yards, receiving_yards FROM player_stats WHERE season_type = 'REG' AND season >= 2016 ORDER BY player_id, season, week, pulled_at DESC)""").pl()
    sc = con.execute("""SELECT * FROM (SELECT DISTINCT ON (game_id) game_id, season, week, gameday, home_team, away_team, home_qb_id, away_qb_id
                        FROM schedules WHERE game_type = 'REG' ORDER BY game_id, pulled_at DESC)""").pl()
    con.close()
    sc = sc.with_columns(pl.col("season", "week").cast(pl.Int32), gameday=pl.col("gameday").cast(pl.Utf8).str.slice(0, 10))
    tg = pl.concat([sc.select("game_id", "season", "week", "gameday", team="home_team", qb="home_qb_id"),
                    sc.select("game_id", "season", "week", "gameday", team="away_team", qb="away_qb_id")])
    ps = ps.with_columns(pl.col("season", "week").cast(pl.Int32), *[pl.col(c).fill_null(0).cast(pl.Float64) for c in ("carries", "targets", "receptions", "rushing_yards", "receiving_yards")])
    ps = ps.join(tg, on=["season", "week", "team"], how="inner")
    tot = ps.group_by("game_id", "team").agg(team_car=pl.col("carries").sum(), team_tgt=pl.col("targets").sum())
    ps = ps.join(tot, on=["game_id", "team"]).with_columns(car_sh=pl.col("carries") / pl.col("team_car"), tgt_sh=pl.col("targets") / pl.col("team_tgt"))
    return ps, tg


def roles(ps: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """Pre-game role features for every player-game (players who played), built team by team in game order. For the next unplayed week (live,
    2026-10-21) a row per player with a recent role on the team, usage left empty, so the engine can score upcoming games."""
    played_games = set(ps["game_id"].unique().to_list())
    nxt = tg.filter(~pl.col("game_id").is_in(list(played_games)) & (pl.col("season") == tg["season"].max()))
    upcoming = set(nxt.filter(pl.col("week") == nxt["week"].min())["game_id"].to_list()) if nxt.height else set()
    from comps2 import pregame as PG
    av = {}
    for (se, wk, tm), x in PG.load().group_by("season", "week", "team"):
        av[(int(se), int(wk), tm)] = (set(x.filter(pl.col("available"))["gsis_id"].to_list()), set(x.filter(~pl.col("available"))["gsis_id"].to_list()))
    pos = dict(ps.select("player_id", "position").unique("player_id").iter_rows())
    rows = []
    for (team,), t in tg.sort("gameday").group_by("team", maintain_order=True):
        games = t.select("game_id", "season", "week", "gameday", "qb").unique("game_id").sort("gameday")
        u = ps.filter(pl.col("team") == team)
        by_game = {g: x for (g,), x in u.group_by("game_id")}
        hist: list[dict] = []          # per earlier team game: {player: (car_sh, tgt_sh)}
        team_vol: list[tuple] = []
        prev_qb, vac_streak = None, 0
        first_game = {}
        for gi, (gid, se, wk, gd, qb) in enumerate(games.iter_rows()):
            cur = by_game.get(gid)
            recent = hist[-ROLE_GAMES:]
            last4 = hist[-4:]
            own = {}
            for h in recent:
                for p, (c, tt) in h.items():
                    own.setdefault(p, []).append((c, tt))
            role = {p: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))) for p, v in own.items()}
            played4 = {p: sum(p in h for h in last4) for p in role}
            avail, out = av.get((int(se), int(wk), team), (None, set()))
            regular = {p: r for p, r in role.items() if played4.get(p, 0) > 0}
            vac_car = sum(r[0] for p, r in regular.items() if p in out and r[0] >= REG_CAR)
            vac_tgt = sum(r[1] for p, r in regular.items() if p in out and r[1] >= REG_TGT)
            last = hist[-1] if hist else {}
            back = {p: r for p, r in role.items() if p not in last and played4.get(p, 0) > 0 and (avail is None or p in avail)}
            vac_streak = vac_streak + 1 if max(vac_car, vac_tgt) >= VAC_EVENT else 0
            qb_change = float(prev_qb is not None and qb is not None and qb != prev_qb)
            tv = team_vol[-4:]
            car_pg = float(np.mean([a for a, _ in tv])) if tv else np.nan
            tgt_pg = float(np.mean([b for _, b in tv])) if tv else np.nan
            if cur is not None:
                avail_role = {p: r for p, r in role.items() if (avail is None or p in avail) and p not in out}
                for r in cur.iter_rows(named=True):
                    p = r["player_id"]
                    oc, ot = role.get(p, (0.0, 0.0))
                    others_c = sorted([v[0] for q, v in avail_role.items() if q != p], reverse=True)
                    others_t = sorted([v[1] for q, v in avail_role.items() if q != p], reverse=True)
                    since = next((k for k, h in enumerate(reversed(hist), 1) if p in h), 99)
                    first_game.setdefault(p, gi)
                    ps_ = pos.get(p, "")
                    rows.append(dict(game_id=gid, player_id=p, team=team, season=int(se), week=int(wk), gameday=gd, own_car=oc, own_tgt=ot,
                                     played_recent=float(played4.get(p, 0)), since_last=float(since), vac_car=vac_car, vac_tgt=vac_tgt,
                                     rank_car=float(1 + sum(v > oc for v in others_c)), rank_tgt=float(1 + sum(v > ot for v in others_t)),
                                     comp_car=float(sum(others_c[:2])), comp_tgt=float(sum(others_t[:3])),
                                     back_car=float(sum(v[0] for q, v in back.items() if q != p)), back_tgt=float(sum(v[1] for q, v in back.items() if q != p)),
                                     vac_nth=float(vac_streak), qb_change=qb_change, new_team=float(gi - first_game[p] < 2 and gi > 2),
                                     is_rb=float(ps_ in ("RB", "FB")), is_wr=float(ps_ == "WR"), is_te=float(ps_ == "TE"),
                                     team_car_pg=car_pg, team_tgt_pg=tgt_pg, car_sh=r["car_sh"], tgt_sh=r["tgt_sh"],
                                     carries=r["carries"], targets=r["targets"], receptions=r["receptions"], rushing_yards=r["rushing_yards"],
                                     receiving_yards=r["receiving_yards"], team_car=r["team_car"], team_tgt=r["team_tgt"]))
                hist.append({r["player_id"]: (r["car_sh"], r["tgt_sh"]) for r in cur.iter_rows(named=True)})
                team_vol.append((float(cur["team_car"][0]), float(cur["team_tgt"][0])))
            elif gid in upcoming:
                avail_role = {p: r for p, r in role.items() if (avail is None or p in avail) and p not in out}
                for p in [q for q in role if played4.get(q, 0) > 0 and (avail is None or q in avail) and q not in out]:
                    oc, ot = role[p]
                    others_c = sorted([v[0] for q, v in avail_role.items() if q != p], reverse=True)
                    others_t = sorted([v[1] for q, v in avail_role.items() if q != p], reverse=True)
                    since = next((k for k, h in enumerate(reversed(hist), 1) if p in h), 99)
                    ps_ = pos.get(p, "")
                    rows.append(dict(game_id=gid, player_id=p, team=team, season=int(se), week=int(wk), gameday=gd, own_car=oc, own_tgt=ot,
                                     played_recent=float(played4.get(p, 0)), since_last=float(since), vac_car=vac_car, vac_tgt=vac_tgt,
                                     rank_car=float(1 + sum(v > oc for v in others_c)), rank_tgt=float(1 + sum(v > ot for v in others_t)),
                                     comp_car=float(sum(others_c[:2])), comp_tgt=float(sum(others_t[:3])),
                                     back_car=float(sum(v[0] for q, v in back.items() if q != p)), back_tgt=float(sum(v[1] for q, v in back.items() if q != p)),
                                     vac_nth=float(vac_streak), qb_change=qb_change, new_team=float(gi - first_game.get(p, gi) < 2 and gi > 2),
                                     is_rb=float(ps_ in ("RB", "FB")), is_wr=float(ps_ == "WR"), is_te=float(ps_ == "TE"),
                                     team_car_pg=car_pg, team_tgt_pg=tgt_pg, car_sh=None, tgt_sh=None, carries=None, targets=None, receptions=None,
                                     rushing_yards=None, receiving_yards=None, team_car=None, team_tgt=None))
            prev_qb = qb if qb is not None else prev_qb
    return pl.DataFrame(rows, infer_schema_length=None)


def efficiency(R: pl.DataFrame) -> pl.DataFrame:
    """Per player, pre-game yards per carry, catch rate, yards per target over his last 16 games, shrunk toward his position group."""
    R = R.sort("player_id", "gameday")
    cs = lambda c: pl.col(c).shift(1).rolling_sum(16, min_samples=1).over("player_id").fill_null(0)
    R = R.with_columns(c16=cs("carries"), ry16=cs("rushing_yards"), t16=cs("targets"), rc16=cs("receptions"), wy16=cs("receiving_yards"))
    return R.with_columns(ypc=(pl.col("ry16") + 20 * 4.2) / (pl.col("c16") + 20), ctch=(pl.col("rc16") + 15 * 0.65) / (pl.col("t16") + 15),
                          ypt=(pl.col("wy16") + 15 * 7.5) / (pl.col("t16") + 15))


def forward_share(R: pl.DataFrame) -> pl.DataFrame:
    X = R.select(FEATS).to_numpy().astype(float)
    season, week = R["season"].to_numpy(), R["week"].to_numpy()
    out = {}
    for tgt in ("car_sh", "tgt_sh"):
        y = R[tgt].to_numpy().astype(float)
        pred = np.full(len(y), np.nan)
        for S in range(FIRST_SEASON + 1, int(season.max()) + 1):
            for i, b0 in enumerate(BLOCKS):
                b1 = BLOCKS[i + 1] if i + 1 < len(BLOCKS) else 99
                blk = (season == S) & (week >= b0) & (week < b1)
                if not blk.any():
                    continue
                tr = ((season < S) | ((season == S) & (week < b0))) & (season >= FIRST_SEASON) & ~np.isnan(y)
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=100, early_stopping=False)
                m.fit(X[tr], y[tr])
                pred[blk] = np.clip(m.predict(X[blk]), 0, 1)
        out[f"pred_{tgt}"] = pred
    return R.with_columns(**{k: pl.Series(v) for k, v in out.items()})


def build() -> pl.DataFrame:
    ps, tg = usage()
    R = roles(ps, tg)
    R = efficiency(R)
    R = forward_share(R)
    vol_c, vol_t = pl.col("team_car_pg"), pl.col("team_tgt_pg")
    return R.with_columns(rc_rush_att=pl.col("pred_car_sh") * vol_c, rc_rush_yds=pl.col("pred_car_sh") * vol_c * pl.col("ypc"),
                          rc_targets=pl.col("pred_tgt_sh") * vol_t, rc_rec=pl.col("pred_tgt_sh") * vol_t * pl.col("ctch"),
                          rc_rec_yds=pl.col("pred_tgt_sh") * vol_t * pl.col("ypt"))


def report(R: pl.DataFrame):
    ev = (pl.col("vac_car") >= VAC_EVENT) | (pl.col("vac_tgt") >= VAC_EVENT) | (pl.col("since_last") >= 2) & (pl.col("since_last") < 99) | (pl.col("back_car") + pl.col("back_tgt") >= 0.15)
    x = R.filter(pl.col("season") >= FIRST_SEASON + 1)
    for tgt, own, role in (("car_sh", "own_car", "carries (backs)"), ("tgt_sh", "own_tgt", "targets (receivers)")):
        y = x.filter(pl.col("is_rb") == 1) if tgt == "car_sh" else x.filter(pl.col(own) + pl.col(tgt) > 0.02)
        for lab, z in (("all player-games", y), ("role-change games", y.filter(ev))):
            e_old = (z[tgt] - z[own]).abs().mean()
            e_new = (z[tgt] - z[f"pred_{tgt}"]).abs().mean()
            print(f"  share of {role:20s} {lab:18s} n {z.height:6d}: miss using his old share {e_old:.4f}, engine {e_new:.4f} ({(e_new / e_old - 1):+.1%})")


if __name__ == "__main__":
    import time
    t0 = time.time()
    R = build()
    R.write_parquet(OUT)
    print(R.shape, f"{time.time() - t0:.0f}s")
    report(R)


ROLE_COLS = ("vac", "vac_nth", "own", "pred", "role_jump", "line_jump", "over_react", "back", "since_last", "qb_change")


def role_block(d: pl.DataFrame) -> np.ndarray:
    """Role-change inputs aligned with the picker's rows (mark4.forward.frame): for rushing props the carry role, for receiving props the target role,
    for passing props only the QB change and the line jump. line_jump = this line / the player's previous line in the same market (both pre-game)."""
    R = pl.read_parquet(OUT).select("game_id", "player_id", "vac_car", "vac_tgt", "vac_nth", "own_car", "own_tgt", "pred_car_sh", "pred_tgt_sh",
                                    "back_car", "back_tgt", "since_last", "qb_change")
    x = d.with_row_index("i").select("i", "game_id", "player_id", "market", "line", "gameday")
    x = x.sort("player_id", "market", "gameday").with_columns(prev_line=pl.col("line").shift(1).over("player_id", "market")).sort("i")
    x = x.join(R, on=["game_id", "player_id"], how="left").sort("i")
    rush, recv = pl.col("market").str.starts_with("rush"), pl.col("market").is_in(["rec", "rec_yds", "targets"])
    pick = lambda c, t: pl.when(rush).then(pl.col(c)).when(recv).then(pl.col(t)).otherwise(None)
    x = x.with_columns(vac=pick("vac_car", "vac_tgt"), own=pick("own_car", "own_tgt"), pred=pick("pred_car_sh", "pred_tgt_sh"), back=pick("back_car", "back_tgt"),
                       line_jump=(pl.col("line") / pl.col("prev_line")).log())
    x = x.with_columns(role_jump=pl.when(pl.col("own") >= 0.05).then((pl.col("pred") / pl.col("own")).log()))
    x = x.with_columns(over_react=pl.col("line_jump") - pl.col("role_jump"))
    X = x.select(ROLE_COLS).cast(pl.Float64).to_numpy()
    return np.where(np.isfinite(X), X, np.nan)


OVERREACT = 0.35          # the line moved this much (log) more than the role: FanDuel overreacting to a promotion
PRIOR_HIT, PRIOR_N = 0.5, 100.0


def mark4l(d: pl.DataFrame, A, p4i: np.ndarray, verbose=False) -> np.ndarray:
    """Mark 4l (2026-10-20): Mark 4i, except on promotion overreactions (vacancy 15%+ in his role, games 1-3 of it, line moved OVERREACT more than the
    engine's role estimate): there the under's chance is the hit rate of such lines in EARLIER seasons only, shrunk toward 50% with PRIOR_N lines."""
    X = role_block(d)
    c = {n: X[:, i] for i, n in enumerate(ROLE_COLS)}
    flag = (np.nan_to_num(c["vac"]) >= VAC_EVENT) & (np.nan_to_num(c["vac_nth"]) >= 1) & (np.nan_to_num(c["vac_nth"]) <= 3) & (np.nan_to_num(c["over_react"], nan=-9) >= OVERREACT)
    y = A.over_won.astype(float)
    season = d["season"].to_numpy()
    p = p4i.copy()
    for S in np.unique(season):
        past = flag & (season < S) & ~np.isnan(y)
        hit = ((1 - y[past]).sum() + PRIOR_HIT * PRIOR_N) / (past.sum() + PRIOR_N)
        cur = flag & (season == S)
        p[cur] = 1 - hit
        if verbose:
            print(f"   {S}: {int(cur.sum())} overreaction lines, under chance from earlier seasons {hit:.3f} (on {int(past.sum())} past lines)")
    return p
