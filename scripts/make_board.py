"""Probability board data: for every stored prediction of a run, the chance the real number reaches each bar (a survival curve), plus the 90/80/70% floors.

Display only (not a stored record, not a model input; no line is read). Curves come from the committed calibrator's pool: the actuals of the 600 past predictions nearest to the
prediction (250 for the QB rushing markets), 2020-2024. Usage: python scripts/make_board.py <store dir> <out.json>"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from app import data as D
from eval import calibrate as C

STEP = {"pass_att": 1, "pass_cmp": 1, "pass_yds": 10, "qb_rush_att": 1, "qb_rush_yds": 5, "rush_att": 1, "rush_yds": 5, "targets": 1, "rec": 1, "rec_yds": 5}
PLAYER_MARKETS = list(STEP)


def curves(cal: C.Calibrator, market: str, p: float):
    v = cal.markets.get(market)
    if v is None or len(v["pred"]) < cal.cfg.min_pool[market]:
        return None
    idx = C._window(v["pred"], np.array([p]), cal.cfg.k[market])
    s = v["act"][idx][0]
    top = float(np.quantile(v["act"], 0.995))
    step = STEP[market]
    ts = list(range(step, int(np.ceil(top / step)) * step + step, step))
    surv = [round(float((s >= t).mean()), 3) for t in ts]
    fl = {f"f{int(l * 100)}": float(np.floor(np.quantile(s, 1 - l, method="lower"))) for l in (0.9, 0.8, 0.7)}
    return dict(step=step, t0=ts[0], surv=surv, **fl)


def build(store: Path) -> dict:
    p = D.latest(ST_load(store))
    cal = C.Calibrator.load(config.PROCESSED_DIR / "calibrator_v1")
    pr = p.filter(pl.col("player_id").is_not_null() & pl.col("market").is_in(PLAYER_MARKETS))
    names = D.player_names(pr["player_id"].unique().to_list())
    teams = {}
    try:
        import duckdb
        con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
        teams = dict(con.execute("SELECT DISTINCT gsis_id, latest_team FROM players").fetchall())
        con.close()
    except Exception:
        pass
    rows = []
    for r in pr.iter_rows(named=True):
        c = curves(cal, r["market"], r["mean"])
        if c is None:
            continue
        gp = D.game_parts(r["game_id"])
        team = teams.get(r["player_id"])
        rows.append(dict(game=r["game_id"], away=gp["away"], home=gp["home"], kickoff=str(r["kickoff_time"])[:16], player=names.get(r["player_id"], r["player_id"]),
                         team=team if team in (gp["away"], gp["home"]) else "", market=r["market"], mean=round(r["mean"], 1), stack=r["stack"], **c))
    games = {}
    for r in p.filter(pl.col("player_id").is_null()).iter_rows(named=True):
        games.setdefault(r["game_id"], {})[r["market"]] = round(r["mean"], 2)
    return dict(rows=rows, games=games, week=sorted({(D.game_parts(g)["season"], D.game_parts(g)["week"]) for g in p["game_id"].unique().to_list()})[-1])


def ST_load(store: Path) -> pl.DataFrame:
    from live import store as ST
    return ST.PredictionStore(store).load()


if __name__ == "__main__":
    out = build(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(out))
    print(len(out["rows"]), "rows;", len(out["games"]), "games; week", out["week"])
