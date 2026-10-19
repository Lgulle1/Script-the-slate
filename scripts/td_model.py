"""Anytime touchdown model and hidden-season test (yours, 2026-10-19).

Target: the player scored a rushing, receiving or special-teams touchdown (records/lines/td_lines_graded.parquet, from scripts/td_import.py).
Inputs known before the game:
  book      FanDuel's price (break-even chance 1 / decimal odds, overround included)
  usage     recency-weighted (half-life 6 games) per-game rates over his earlier games: touchdowns, carries inside the 10 and inside the 5, targets inside the 20 and the 10,
            and his share of the team's red-zone plays; games played; position, depth slot
  projection  the stat model's projected carries, rushing yards, catches, receiving yards, pass attempts for this game (walk-forward 2020-2024, the stored live ones 2025-26)
  game      the team's implied points (total and spread), home, dome
Each season 2021-2026 is hidden in turn: the model is trained on the other five and bets the hidden one (three seeds averaged).
Bets: "yes" at FanDuel's price when model chance x decimal odds - 1 (expected value per unit) clears a cut-off; the top of each week by expected value, at most 2 per game;
as straight bets and as 2-leg parlays in rank order (different games, 30% profit boost). 1 unit each.

  python scripts/td_model.py      # writes records/tuning/td_model.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
BOOST = 0.30
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
MKTS = ["rush_att", "rush_yds", "rec", "rec_yds", "pass_att", "pass_yds"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def red_zone_table() -> pl.DataFrame:
    """One row per (game, player) with red-zone involvement, and one per (game, team) with the team's red-zone plays (pbp, regular season)."""
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    p = con.execute("SELECT DISTINCT ON (game_id, play_id) game_id, posteam, play_type, yardline_100, rusher_player_id, receiver_player_id, rush_touchdown, pass_touchdown "
                    "FROM pbp WHERE season_type = 'REG' AND play_type IN ('run', 'pass') AND posteam IS NOT NULL ORDER BY game_id, play_id, pulled_at DESC").pl()
    con.close()
    p = p.with_columns(pl.col("yardline_100").cast(pl.Float64))
    run_ = p.filter((pl.col("play_type") == "run") & pl.col("rusher_player_id").is_not_null()).select(
        "game_id", player_id="rusher_player_id", rz_c=(pl.col("yardline_100") <= 10).cast(pl.Float64), gl_c=(pl.col("yardline_100") <= 5).cast(pl.Float64),
        td_r=pl.col("rush_touchdown").cast(pl.Float64).fill_null(0))
    rec = p.filter((pl.col("play_type") == "pass") & pl.col("receiver_player_id").is_not_null()).select(
        "game_id", player_id="receiver_player_id", rz_t=(pl.col("yardline_100") <= 20).cast(pl.Float64), ez_t=(pl.col("yardline_100") <= 10).cast(pl.Float64),
        td_c=pl.col("pass_touchdown").cast(pl.Float64).fill_null(0))
    a = run_.group_by("game_id", "player_id").agg(pl.col("rz_c").sum(), pl.col("gl_c").sum(), pl.col("td_r").sum())
    b = rec.group_by("game_id", "player_id").agg(pl.col("rz_t").sum(), pl.col("ez_t").sum(), pl.col("td_c").sum())
    pg = a.join(b, on=["game_id", "player_id"], how="full", coalesce=True).fill_null(0)
    team = p.filter(pl.col("yardline_100") <= 20).group_by("game_id", "posteam").agg(team_rz=pl.len())
    return pg, team


def usage_features(lines: pl.DataFrame, plog: pl.DataFrame, rz: pl.DataFrame, team_rz: pl.DataFrame) -> pl.DataFrame:
    """Per line: recency-weighted rates over the player's earlier games."""
    base = plog.select("player_id", "game_id", "team", "gameday", "family", "slot").join(rz, on=["game_id", "player_id"], how="left").join(
        team_rz, left_on=["game_id", "team"], right_on=["game_id", "posteam"], how="left").fill_null(0)
    base = base.with_columns(day=pl.col("gameday").cast(pl.String).str.slice(0, 10).str.to_date(), td=((pl.col("td_r") + pl.col("td_c")) >= 1).cast(pl.Float64),
                             rz_share=pl.when(pl.col("team_rz") > 0).then((pl.col("rz_c") + pl.col("rz_t")) / pl.col("team_rz")).otherwise(0.0)).sort("player_id", "day")
    cols = ["td", "rz_c", "gl_c", "rz_t", "ez_t", "rz_share", "td_r", "td_c"]
    by = {k[0]: g for k, g in base.partition_by("player_id", as_dict=True).items()}
    want = lines.select("player_id", "game_id").unique()
    gdate = base.select("game_id", "day").unique().to_dict(as_series=False)
    day_of = dict(zip(gdate["game_id"], gdate["day"]))
    rows = []
    for pid, gid in zip(want["player_id"].to_list(), want["game_id"].to_list()):
        g = by.get(pid)
        d = day_of.get(gid)
        r = dict(player_id=pid, game_id=gid)
        if g is None or d is None:
            rows.append(r)
            continue
        days = g["day"].to_list()
        j = sum(1 for x in days if x < d)
        r["n_games"] = float(j)
        if j:
            lo = max(0, j - 40)
            w = 0.5 ** ((j - 1 - np.arange(lo, j)) / 6.0)
            for c in cols:
                v = g[c].to_numpy()[lo:j]
                r[f"u_{c}"] = float((w * v).sum() / w.sum())
            r["u_td_season"] = float(g["td"].to_numpy()[max(0, j - 17):j].mean())
        rows.append(r)
    return pl.DataFrame(rows)


def projections(plog: pl.DataFrame) -> pl.DataFrame:
    """(game_id, player_id) -> projected stat means: walk-forward base model for 2020-2024, stored live-stack predictions for 2025-2026."""
    LS = _load("line_study")
    wf = pl.read_parquet(config.PROCESSED_DIR / "vol_x_eff_2024_walkforward.parquet").filter((pl.col("kind") == "player") & pl.col("market").is_in(MKTS))
    gid = plog.select("player_id", "season", "week", "game_id").unique(["player_id", "season", "week"])
    a = wf.select(player_id="entity", season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32), market="market", mean="prediction").join(
        gid.with_columns(season=pl.col("season").cast(pl.Int32), week=pl.col("week").cast(pl.Int32)), on=["player_id", "season", "week"], how="inner").select("game_id", "player_id", "market", "mean")
    parts = [a]
    for s in (2025, 2026):
        p = LS.predictions(s).filter(pl.col("market").is_in(MKTS) & pl.col("player_id").is_not_null())
        if "game_id" not in p.columns:
            continue
        parts.append(p.select("game_id", "player_id", "market", "mean"))
    allp = pl.concat(parts).unique(["game_id", "player_id", "market"], keep="last")
    return allp.pivot(on="market", index=["game_id", "player_id"], values="mean").rename({m: f"proj_{m}" for m in MKTS if m in allp["market"].unique().to_list()})


def game_context() -> pl.DataFrame:
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    s = con.execute("SELECT DISTINCT ON (game_id) game_id, home_team, away_team, spread_line, total_line, roof FROM schedules ORDER BY game_id, pulled_at DESC").pl()
    con.close()
    return s


def build() -> pl.DataFrame:
    H = _load("td_import").H
    d = pl.read_parquet(config.ROOT / "records" / "lines" / "td_lines_graded.parquet").filter(pl.col("scored").is_not_null() & pl.col("price").is_not_null())
    plog = H.read_played_logs(2026)[0]
    rz, team_rz = red_zone_table()
    U = usage_features(d, plog, rz, team_rz)
    P = projections(plog)
    G = game_context()
    x = d.join(U, on=["player_id", "game_id"], how="left").join(P, on=["game_id", "player_id"], how="left").join(G, on="game_id", how="left")
    slot = plog.select("player_id", "game_id", "slot").unique(["player_id", "game_id"])
    x = x.join(slot, on=["player_id", "game_id"], how="left")
    home = (pl.col("team") == pl.col("home_team")).cast(pl.Float64)
    tot, spr = pl.col("total_line").fill_null(44.0), pl.col("spread_line").fill_null(0.0)
    x = x.with_columns(is_home=home, implied_pts=pl.when(home == 1).then(tot / 2 + spr / 2).otherwise(tot / 2 - spr / 2), dome=pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64),
                       pos=pl.col("position").replace_strict({"QB": 0, "RB": 1, "WR": 2, "TE": 3}, default=-1).cast(pl.Float64),
                       logit_implied=(pl.col("implied") / (1 - pl.col("implied"))).log())
    return x


FEATS = ["implied", "logit_implied", "price", "n_games", "u_td", "u_rz_c", "u_gl_c", "u_rz_t", "u_ez_t", "u_rz_share", "u_td_r", "u_td_c", "u_td_season",
         "proj_rush_att", "proj_rush_yds", "proj_rec", "proj_rec_yds", "proj_pass_att", "proj_pass_yds", "total_line", "spread_line", "implied_pts", "is_home", "dome", "pos", "slot", "week"]


def hidden_predictions(x: pl.DataFrame, feats, seeds=(1, 2, 3)) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    cols = [c for c in feats if c in x.columns]
    X = x.select(cols).to_numpy().astype(float)
    y = x["scored"].to_numpy()
    season = x["season"].to_numpy()
    p = np.full(x.height, np.nan)
    for S in SEASONS:
        tr, te = season != S, season == S
        p[te] = np.mean([HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=80, l2_regularization=2.0, random_state=s).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                         for s in seeds], axis=0)
    return p


def logloss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def bet(rows, n, theta, kind, cap=2):
    """rows: dicts for one week sorted by expected value desc."""
    legs, g = [], {}
    for r in rows:
        if r["ev"] < theta or g.get(r["game_id"], 0) >= cap:
            continue
        legs.append(r)
        g[r["game_id"]] = g.get(r["game_id"], 0) + 1
        if len(legs) == n:
            break
    if kind == "straight":
        return [((r["dec"] - 1) if r["scored"] == 1 else -1.0) for r in legs]
    out, used = [], set()
    for i, a in enumerate(legs):
        if i in used:
            continue
        for j in range(i + 1, len(legs)):
            if j not in used and legs[j]["game_id"] != a["game_id"]:
                used |= {i, j}
                out.append((a["dec"] * legs[j]["dec"] - 1) * (1 + BOOST) if a["scored"] == 1 and legs[j]["scored"] == 1 else -1.0)
                break
    return out


def evaluate(x: pl.DataFrame, p: np.ndarray, label: str) -> dict:
    y = x["scored"].to_numpy()
    x = x.with_columns(p=pl.Series(p)).with_columns(ev=pl.col("p") * pl.col("dec") - 1)
    out = dict(label=label, logloss_model=logloss(y, p), logloss_book_raw=logloss(y, x["implied"].to_numpy()), bets={})
    W = {}
    for r in x.sort("ev", descending=True).iter_rows(named=True):
        W.setdefault((r["season"], r["week"]), []).append(r)
    for kind, n in (("straight", 3), ("straight", 6), ("parlay", 4), ("parlay", 6)):
        for theta in (0.0, 0.10, 0.20, 0.30):
            per = {}
            for (s, w), rows in W.items():
                res = bet(rows, n, theta, kind)
                u, b = per.setdefault(s, [0.0, 0])
                per[s] = [u + sum(res), b + len(res)]
            U, B = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
            out["bets"][f"{kind} top {n}, EV >= {int(theta * 100)}%"] = dict(roi=U / B if B else None, bets=B, seasons={str(s): (v[0] / v[1] if v[1] else None) for s, v in sorted(per.items())})
    return out


def main():
    x = build()
    print("lines", x.height, "scored", round(x["scored"].mean(), 3), "with usage history", int(x["u_td"].is_not_null().sum()), "with projections", int(x["proj_rec"].is_not_null().sum() + x["proj_rush_att"].is_not_null().sum()))
    from sklearn.linear_model import LogisticRegression
    y, season = x["scored"].to_numpy(), x["season"].to_numpy()
    # baseline: the book's price alone, recalibrated on the other seasons
    li = x["logit_implied"].to_numpy().reshape(-1, 1)
    pb = np.full(x.height, np.nan)
    for S in SEASONS:
        pb[season == S] = LogisticRegression(C=100).fit(li[season != S], y[season != S]).predict_proba(li[season == S])[:, 1]
    results = {"book price only (recalibrated)": evaluate(x, pb, "book price only")}
    pm = hidden_predictions(x, FEATS)
    results["full model"] = evaluate(x, pm, "full model")
    no_book = [f for f in FEATS if f not in ("implied", "logit_implied", "price")]
    results["model without the book's price"] = evaluate(x, hidden_predictions(x, no_book), "model without the book's price")
    for k, v in results.items():
        print(f"\n{k}: logloss {v['logloss_model']:.4f} (book raw {v['logloss_book_raw']:.4f})")
        for b, r in v["bets"].items():
            if r["roi"] is not None and r["bets"] >= 30:
                print(f"   {b:<28} {r['roi']:+.1%} on {r['bets']}  " + " ".join(f"{s}:{(v2 if v2 is not None else float('nan')):+.0%}" for s, v2 in r["seasons"].items()))
    (config.ROOT / "records" / "tuning" / "td_model.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
