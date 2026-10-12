"""The weekly live run (plan "Phase 5 gates"): schedule -> targets -> features from input history -> Phase 3 predictions -> the prediction store.

  run(now, raw_db, store_dir, models, ...)   one run at the clock `now` (UTC). Everything it reads is as of `now`: schedule rows and charts stamped after it are
                                              ignored, game logs stop at the target week's cutoff, and the store refuses any game that has kicked off.

Layers: this version scores the Phase 3 base for every market (model_version "phase3"); the weight-1 layers of layer_weights.csv (4a for five markets, 4b spread) join
as new model versions when wired, never silently. Models are fit on 2020-2024 rows only (plan P5.4); the live path only builds feature rows for the targets.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

import config
from eval import backtest as bt
from features import volume_features as vf
from live import derive as DV
from live import history as H
from live import lines as LN
from live import predict as LP
from live import store as ST
from live import targets as TG
from models import efficiency as ef

TRAIN_SEASON_CAP = max(config.BACKTEST_SEASONS)


def code_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True, cwd=config.ROOT, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git must not stop a run; the version then says so
        return "nogit"


def model_version(layer_weights_csv: Path | None = None, layers: str = "phase3") -> str:
    p = layer_weights_csv or (config.ROOT / "layer_weights.csv")
    wh = hashlib.sha256(p.read_bytes()).hexdigest()[:8] if p.exists() else "noweights"
    return f"{layers}:{code_revision()}:{wh}"


def train_models() -> LP.LiveModels:
    """Fit once on the 2020-2024 feature tables (the capped loaders: nothing from 2025 on can reach them)."""
    data = bt.load_backtest_data()
    return LP.fit_models(vf.load_feature_tables(data), ef.load_efficiency_tables(data))


def read_schedule(raw_db, season: int, now: datetime) -> pl.DataFrame:
    """Latest pull of each game's schedule row stamped at or before `now`."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        return con.execute("SELECT * FROM (SELECT DISTINCT ON (game_id) * FROM schedules WHERE season = ? AND pulled_at <= ? ORDER BY game_id, pulled_at DESC)",
                           [season, now.replace(tzinfo=None)]).pl()
    finally:
        con.close()


def read_charts(raw_db, now: datetime) -> pl.DataFrame:
    """New-layout depth chart rows (season NULL) stamped (dt and pulled_at) at or before `now`."""
    con = duckdb.connect(str(raw_db), read_only=True)
    try:
        return con.execute("SELECT DISTINCT dt, team, pos_abb, pos_rank, gsis_id, pos_grp, season, club_code, week, game_type, depth_team, position, formation FROM depth_charts "
                           "WHERE season IS NULL AND dt IS NOT NULL AND pulled_at <= ?", [now.replace(tzinfo=None)]).pl()
    finally:
        con.close()


def target_week(sched: pl.DataFrame, now: datetime) -> tuple:
    """(season, week) of the earliest regular-season week with a game that has not kicked off at `now`; None when the season is over."""
    s = TG.schedule_frame(sched).filter(pl.col("kickoff_time") > now)
    if s.is_empty():
        return None
    r = s.sort("kickoff_time").row(0, named=True)
    return r["season"], r["week"]


def _align(ph: pl.DataFrame, like: pl.DataFrame) -> pl.DataFrame:
    return ph.select([(pl.col(c).cast(t) if c in ph.columns else pl.lit(None, dtype=t).alias(c)) for c, t in like.schema.items()])


def _row_hash(df: pl.DataFrame, keys: list, drop=("label", "den")) -> pl.DataFrame:
    cols = [c for c in df.columns if c not in keys and c not in drop]
    return df.select(*keys, h=pl.struct(cols).hash(seed=0).cast(pl.String))


def run(now: datetime, raw_db=config.RAW_DUCKDB_PATH, store_dir=None, models: LP.LiveModels | None = None, run_id: str | None = None,
        lines_path=None, version: str | None = None, work_dir=None) -> dict:
    now = now.astimezone(timezone.utc)
    store_dir = Path(store_dir or config.ROOT / "records" / "predictions")
    season = ST_season = H.season_of(now.date())
    sched = read_schedule(raw_db, season, now)
    tw = target_week(sched, now)
    if tw is None:
        return dict(status="no upcoming game", written=0)
    season, week = tw
    live_db = DV.build_live_db(raw_db, Path(work_dir or config.PROCESSED_DIR / "live") / "live.duckdb", now, season, week)
    games = TG.target_games(sched, season, week)
    cutoff = games["gameday"].min()
    trows = TG.team_rows(sched, games)
    charts = read_charts(live_db, now)
    from features.depth_live import chart_as_of
    chart = chart_as_of(charts, now.strftime("%Y-%m-%dT%H:%M:%SZ"), week).rename({"team": "team"})
    hist = H.read_input_history(cutoff, live_db)
    prows = TG.player_rows(games, chart, hist.player_log, hist.team_log, trows)
    P = pl.concat([hist.player_log, _align(prows, hist.player_log)])
    T = pl.concat([hist.team_log, _align(trows, hist.team_log)])
    vol = vf.build_feature_tables(P, T, hist.team_volume, hist.lineups)
    eff = ef.build_efficiency_tables(P, T, hist.team_volume, hist.lineups)
    gids = games["game_id"].to_list()
    sel = lambda t: vf.FeatureTables({q: d.filter(pl.col("game_id").is_in(gids)) for q, d in t.players.items()}, t.teams.filter(pl.col("game_id").is_in(gids)))
    vt, et = sel(vol), sel(eff)
    models = models or train_models()
    elig_of = {(r["game_id"], r["player_id"]): r["elig"] for r in prows.filter(pl.col("elig") != "").iter_rows(named=True)}
    pp = LP.predict_players(models, vt, et, elig_of)
    gp = LP.predict_games(models, vt, et, games.select("game_id", "home_team", "away_team"))
    # one hash per prediction: the model-input rows behind it (all quantities of the player, or both teams of the game)
    ph = pl.concat([_row_hash(d, ["game_id", "player_id"]) for d in list(vt.players.values()) + list(et.players.values()) if d.height]).group_by("game_id", "player_id").agg(
        input_hash=pl.col("h").sort().str.join("|")).with_columns(pl.col("input_hash").map_elements(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16], return_dtype=pl.String))
    gh = pl.concat([_row_hash(vt.teams, ["game_id", "team"]), _row_hash(et.teams, ["game_id", "team"])]).group_by("game_id").agg(
        input_hash=pl.col("h").sort().str.join("|")).with_columns(pl.col("input_hash").map_elements(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16], return_dtype=pl.String))
    k = games.select("game_id", "kickoff_time")
    mv = version or model_version()
    preds = pl.concat([
        pp.join(ph, on=["game_id", "player_id"], how="left").join(k, on="game_id").with_columns(model_version=pl.lit(mv)),
        gp.join(gh, on="game_id", how="left").join(k, on="game_id").with_columns(player_id=pl.lit(None, dtype=pl.String), model_version=pl.lit(mv))], how="diagonal_relaxed")
    preds = preds.with_columns(input_hash=pl.col("input_hash").fill_null("none"))
    rid = run_id or now.strftime("run_%Y%m%dT%H%M%SZ")
    res = ST.PredictionStore(store_dir).write(preds.select("game_id", "player_id", "market", "kickoff_time", "model_version", "mean", "input_hash"), now=now, run_id=rid)
    if lines_path is not None:
        LN.log_lines(games.select("game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline"), captured_at=now, path=lines_path)
    manifest = dict(run_id=rid, run_timestamp=now.isoformat(), season=season, week=week, model_version=mv, n_games=len(gids), cutoff=str(cutoff),
                    history=dict(player_rows=hist.player_log.height, team_rows=hist.team_log.height, newest_game=str(hist.team_log["gameday"].max()),
                                 through_season=hist.through_season),
                    chart=dict(rows=chart.height, newest_dt=chart["dt"].max() if chart.height else None, teams=chart["team"].n_unique()),
                    schedule=dict(rows=sched.height, newest_pull=str(sched["pulled_at"].max())),
                    written=res.written, refused=res.refused, unchanged=res.unchanged,
                    train_rows=models.n_train)
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / f"manifest_{rid}.json").write_text(json.dumps(manifest, indent=1, default=str))
    return dict(status="ok", written=res.written, refused=res.refused, unchanged=res.unchanged, season=season, week=week, manifest=manifest, predictions=preds)
