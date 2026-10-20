"""Situation comps (yours, 2026-10-19): the Breece Hall / Braelon Allen case and the Jalen Coker case.

Per player-game row, the SITUATION it was played in, from information known before kickoff:
  vac_car   carry share of the team's backs (recency-weighted over the team's previous VAC_WINDOW games) held by RBs who are NOT playing this game
  vac_tgt   target share held by receivers (WR / TE / RB) who are not playing
  own_car / own_tgt   the player's own recency-weighted share before this game
  status    this week's game status: 0 none / full practice, 1 limited or questionable, 2 doubtful or did not practice (no status) [Coker: DNP = at least doubtful]
  hurt_last 1 if his previous game was played with a questionable / doubtful status or a DNP / limited week
Then the search: every earlier player-game of the same market (league-wide), weighted by how close its situation vector is (kernel), and recency;
the shrunk weighted mean of their rel is the shift. Rows with no vacancy and no injury flag get no situation shift (nothing unusual to compare).

Note: the projection stack already moves volume when a starter is OUT (layer 4a). The residuals here are what 4a got wrong in such games, which is the
point: if backs promoted by an injury beat their (4a-adjusted) expectation by 30% on average, that shows up here.
"""
from __future__ import annotations

import duckdb
import numpy as np
import polars as pl

import config
from live import history as H

VAC_WINDOW_HL = 3.0      # half-life in team games for the share a player "owns"
MIN_SHARE = 0.10         # a player owns a role from this share up
OUT = config.PROCESSED_DIR / "comps2_situations.parquet"


def team_shares(pregame: bool = True) -> pl.DataFrame:
    """pregame (default since 2026-10-20): a teammate's role counts as vacated when, before kickoff, he is not on the active roster or is listed Out /
    Doubtful (comps2.pregame); before, it was "did not play", which is only known after the game."""
    avail = {}
    if pregame:
        from comps2 import pregame as PG
        for (se, wk, tm), g in PG.load().filter(pl.col("available")).group_by("season", "week", "team"):
            avail[(int(se), int(wk), tm)] = set(g["gsis_id"].to_list())
    log = H.read_played_logs(max(config.BACKTEST_SEASONS) + 2)[0].select("player_id", "game_id", "team", "season", "week", "gameday", "family",
                                                                          "carries", "targets")
    log = log.with_columns(pl.col("carries").fill_null(0).cast(pl.Float64), pl.col("targets").fill_null(0).cast(pl.Float64))
    rows = []
    for (team,), t in log.sort("gameday").group_by("team", maintain_order=True):
        games = t.select("game_id", "gameday", "season").unique().sort("gameday")
        gl = games["game_id"].to_list()
        own = {}  # player -> (car_share, tgt_share) as of the current game, EW over team games
        last_seen = {}
        for gi, g in enumerate(gl):
            tg = t.filter(pl.col("game_id") == g)
            players = set(tg["player_id"].to_list())
            present = avail.get((int(tg["season"][0]), int(tg["week"][0]), team), players) if pregame else players
            fam = dict(zip(tg["player_id"].to_list(), tg["family"].to_list()))
            vac_car = sum(v[0] for p, v in own.items() if p not in present and v[2] == "RB" and v[0] >= MIN_SHARE and gi - last_seen.get(p, -99) <= 4)
            vac_tgt = sum(v[1] for p, v in own.items() if p not in present and v[2] in ("WR", "TE", "RB") and v[1] >= MIN_SHARE and gi - last_seen.get(p, -99) <= 4)
            for p in players:
                o = own.get(p, (0.0, 0.0, fam[p]))
                rows.append((g, team, p, float(vac_car), float(vac_tgt), float(o[0]), float(o[1]), int(gi - last_seen[p]) if p in last_seen else 99))
            # update ownership with this game's shares
            tc, tt = tg["carries"].sum(), tg["targets"].sum()
            a = 1 - 0.5 ** (1 / VAC_WINDOW_HL)
            car = dict(zip(tg["player_id"].to_list(), tg["carries"].to_list()))
            tgs = dict(zip(tg["player_id"].to_list(), tg["targets"].to_list()))
            for p in set(own) | players:
                o = own.get(p, (0.0, 0.0, fam.get(p)))
                c = car.get(p, 0.0) / tc if tc else 0.0
                x = tgs.get(p, 0.0) / tt if tt else 0.0
                if p in players:
                    own[p] = ((1 - a) * o[0] + a * c, (1 - a) * o[1] + a * x, fam.get(p, o[2]))
                    last_seen[p] = gi
            # a season change resets nobody: roles carry over, recency does the decay
    return pl.DataFrame(rows, schema=["game_id", "team", "player_id", "vac_car", "vac_tgt", "own_car", "own_tgt", "games_since_last"], orient="row", infer_schema_length=None)


def injury_flags() -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    d = con.execute("""SELECT * FROM (SELECT DISTINCT ON (season, week, gsis_id) CAST(season AS INTEGER) AS season, CAST(week AS INTEGER) AS "week", gsis_id AS player_id,
                       report_status, practice_status FROM injuries WHERE gsis_id IS NOT NULL ORDER BY season, week, gsis_id, pulled_at DESC)""").pl()
    con.close()
    rs, ps = pl.col("report_status").fill_null(""), pl.col("practice_status").fill_null("")
    st = (pl.when(rs.is_in(["Doubtful"]) | ((rs == "") & ps.str.starts_with("Did Not"))).then(2)
          .when((rs == "Questionable") | ps.str.starts_with("Limited")).then(1).otherwise(0))
    return d.with_columns(status=st).select("season", "week", "player_id", "status")


def build() -> pl.DataFrame:
    sh = team_shares()
    log = H.read_played_logs(max(config.BACKTEST_SEASONS) + 2)[0].select("player_id", "game_id", "season", "week", "gameday").sort("player_id", "gameday")
    inj = injury_flags()
    log = log.join(inj, on=["season", "week", "player_id"], how="left").with_columns(pl.col("status").fill_null(0))
    log = log.with_columns(hurt_last=(pl.col("status").shift(1).over("player_id") >= 1).fill_null(False).cast(pl.Int32))
    return sh.join(log.select("player_id", "game_id", "status", "hurt_last"), on=["player_id", "game_id"], how="left").with_columns(
        pl.col("status").fill_null(0), pl.col("hurt_last").fill_null(0))


SIT_FEATS = {
    "rush_att": ("vac_car", "own_car", "status", "hurt_last"), "rush_yds": ("vac_car", "own_car", "status", "hurt_last"),
    "targets": ("vac_tgt", "own_tgt", "status", "hurt_last"), "rec": ("vac_tgt", "own_tgt", "status", "hurt_last"),
    "rec_yds": ("vac_tgt", "own_tgt", "status", "hurt_last"),
    "qb_rush_att": ("status", "hurt_last"), "qb_rush_yds": ("status", "hurt_last"),
    "pass_att": ("vac_tgt", "status", "hurt_last"), "pass_cmp": ("vac_tgt", "status", "hurt_last"), "pass_yds": ("vac_tgt", "status", "hurt_last"),
}
SCALE = dict(vac_car=0.15, vac_tgt=0.10, own_car=0.15, own_tgt=0.08, status=0.5, hurt_last=0.5)


def search(e: pl.DataFrame, s: pl.DataFrame, targets, h=1.0, hl_days=540.0, k0=5.0, window_days=1460) -> pl.DataFrame:
    """sit_shift per target row: league-wide earlier rows of the same market, kernel on the situation vector (scaled by SCALE), recency in days.
    Only rows whose situation is unusual (a vacancy >= MIN_SHARE, or an injury flag) are searched, and only unusual rows serve as comps."""
    e = e.join(s, on=["game_id", "player_id"], how="left").with_columns([pl.col(c).fill_null(0) for c in SCALE])
    rel, gd, mk = e["relc"].to_numpy(), e["gd"].to_numpy(), e["market"].to_numpy()
    shift, neff = np.full(len(e), np.nan), np.full(len(e), np.nan)
    tset = np.zeros(len(e), bool)
    tset[targets] = True
    for m, feats in SIT_FEATS.items():
        mi = np.flatnonzero(mk == m)
        X = e[mi].select([pl.col(f) / SCALE[f] for f in feats]).to_numpy().astype(float)
        raw = e[mi].select(feats).to_numpy().astype(float)
        vac_cols = [i for i, f in enumerate(feats) if f.startswith("vac_")]
        flag_cols = [i for i, f in enumerate(feats) if f in ("status", "hurt_last")]
        unusual = np.zeros(len(mi), bool)
        for i in vac_cols:
            unusual |= raw[:, i] >= MIN_SHARE
        for i in flag_cols:
            unusual |= raw[:, i] >= 1
        g = gd[mi]
        for t in np.flatnonzero(unusual & tset[mi]):
            past = np.flatnonzero(unusual & (g < g[t]) & (g >= g[t] - window_days))
            if not len(past):
                continue
            d2 = ((X[past] - X[t]) ** 2).mean(1)
            w = np.exp(-d2 / (2 * h * h)) * 0.5 ** ((g[t] - g[past]) / hl_days)
            sw = w.sum()
            shift[mi[t]] = (w * rel[mi[past]]).sum() / sw * sw / (sw + k0)
            neff[mi[t]] = sw * sw / (w * w).sum()
    return e.select("rid").with_columns(sit_shift=shift, sit_neff=neff, **{c: e[c] for c in SCALE})


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = build()
    d.write_parquet(OUT)
    print(d.shape, f"{time.time() - t0:.0f}s")
    print(d.select((pl.col("vac_car") >= MIN_SHARE).mean(), (pl.col("vac_tgt") >= MIN_SHARE).mean(), (pl.col("status") >= 1).mean(), pl.col("hurt_last").mean()))
