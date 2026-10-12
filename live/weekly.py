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
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

import config
from eval import backtest as bt
from features import volume_features as vf
from features import injury_features as inf
from live import derive as DV
from live import explain as EX
from live import history as H
from live import layer_sim as LS
from live import layers as LY
from live import lines as LN
from live import predict as LP
from live import store as ST
from live import targets as TG
from models import efficiency as ef

EXPLAIN_DIR = "explain"             # attribution files live in a subdirectory: PredictionStore.load reads every *.parquet in the store root
FIT_FLOOR = date(config.HOLDOUT_SEASON, 3, 1)       # models are fit on games before this date at most: 2020-2024 rows only (plan P5.4)


def code_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True, cwd=config.ROOT, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing git must not stop a run; the version then says so
        return "nogit"


def model_version(layer_weights_csv: Path | None = None) -> str:
    p = layer_weights_csv or (config.ROOT / "layer_weights.csv")
    wh = hashlib.sha256(p.read_bytes()).hexdigest()[:8] if p.exists() else "noweights"
    return f"{code_revision()}:{wh}"


@dataclass
class Trained:
    base: LP.LiveModels
    injury: LP.LiveModels | None = None


def train_models(active: dict | None = None, train_cutoff: date | None = None) -> Trained:
    """Fit once on the 2020-2024 feature tables (the capped loaders: nothing from 2025 on can reach them); `train_cutoff` only for replays of a 2020-2024 week."""
    active = LY.read_active() if active is None else active
    data = bt.load_backtest_data()
    pk, tk = data.player_log, data.team_log
    if train_cutoff is not None:
        pk, tk = pk.filter(pl.col("gameday") < train_cutoff), tk.filter(pl.col("gameday") < train_cutoff)
    base = LP.fit_models(vf.load_feature_tables(data), ef.load_efficiency_tables(data), pk, tk)
    inj = None
    if "4a_injuries" in active:
        feats = inf.build_injury_features(data)
        inj = LP.fit_models(vf.load_feature_tables(data, injury=feats), ef.load_efficiency_tables(data, injury=feats), pk, tk)
    return Trained(base, inj)


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


def _hashes(vt, et) -> tuple:
    """Per-prediction input hashes: the model-input rows behind a player prediction (all quantities of the player) or a game prediction (both teams)."""
    h = lambda df: hashlib.sha256(df.encode()).hexdigest()[:16]
    ph = pl.concat([_row_hash(d, ["game_id", "player_id"]) for d in list(vt.players.values()) + list(et.players.values()) if d.height]).group_by("game_id", "player_id").agg(
        input_hash=pl.col("h").sort().str.join("|")).with_columns(pl.col("input_hash").map_elements(h, return_dtype=pl.String))
    gh = pl.concat([_row_hash(vt.teams, ["game_id", "team"]), _row_hash(et.teams, ["game_id", "team"])]).group_by("game_id").agg(
        input_hash=pl.col("h").sort().str.join("|")).with_columns(pl.col("input_hash").map_elements(h, return_dtype=pl.String))
    return ph, gh


def _score(models: LP.LiveModels, vt, et, elig_of, games) -> pl.DataFrame:
    """(game_id, player_id, market, mean, input_hash) for one model set: player markets and game markets."""
    pp = LP.predict_players(models, vt, et, elig_of)
    gp = LP.predict_games(models, vt, et, games.select("game_id", "home_team", "away_team"))
    ph, gh = _hashes(vt, et)
    return pl.concat([pp.join(ph, on=["game_id", "player_id"], how="left"),
                      gp.join(gh, on="game_id", how="left").with_columns(player_id=pl.lit(None, dtype=pl.String))], how="diagonal_relaxed").select(
        "game_id", "player_id", "market", "mean", "input_hash")


def run(now: datetime, raw_db=config.RAW_DUCKDB_PATH, store_dir=None, models: Trained | None = None, run_id: str | None = None,
        lines_path=None, version: str | None = None, work_dir=None, layer_weights_csv=None, replay: dict | None = None) -> dict:
    """One run at the clock `now`. `replay` (tests only): {"hide_outcomes_from": date, "status_clock": fn, "fit_cutoff": date} scores a past week as if unplayed."""
    now = now.astimezone(timezone.utc)
    replay = replay or {}
    store_dir = Path(store_dir or config.ROOT / "records" / "predictions")
    active = LY.read_active(layer_weights_csv)
    LY.validate(active)
    season = H.season_of(now.date())
    sched = read_schedule(raw_db, season, now) if not replay else read_schedule(raw_db, season, datetime(2999, 1, 1, tzinfo=timezone.utc))
    tw = target_week(sched, now)
    if tw is None:
        return dict(status="no upcoming game", written=0)
    season, week = tw
    live_db = DV.build_live_db(raw_db, Path(work_dir or config.PROCESSED_DIR / "live") / "live.duckdb", now, season, week,
                               apply_pulled_at_clock=not replay, hide_outcomes_from=replay.get("hide_outcomes_from"))
    games = TG.target_games(sched, season, week)
    cutoff = games["gameday"].min()
    trows = TG.team_rows(sched, games)
    con = duckdb.connect(str(live_db), read_only=True)
    try:
        chart = con.execute("SELECT club_code AS team, gsis_id, position, depth_team FROM depth_charts WHERE season = ? AND week = ? AND formation = 'Offense' AND game_type = 'REG' "
                            "AND gsis_id IS NOT NULL", [season, week]).pl()
    finally:
        con.close()
    hist = H.read_input_history(cutoff, live_db)
    prows = TG.player_rows(games, chart, hist.player_log, hist.team_log, trows)
    P = pl.concat([hist.player_log, _align(prows, hist.player_log)])
    T = pl.concat([hist.team_log, _align(trows, hist.team_log)])
    vol = vf.build_feature_tables(P, T, hist.team_volume, hist.lineups)
    eff = ef.build_efficiency_tables(P, T, hist.team_volume, hist.lineups)
    gids = games["game_id"].to_list()
    sel = lambda t: vf.FeatureTables({q: d.filter(pl.col("game_id").is_in(gids)) for q, d in t.players.items()}, t.teams.filter(pl.col("game_id").is_in(gids)))
    vt, et = sel(vol), sel(eff)
    models = models or train_models(active, train_cutoff=replay.get("fit_cutoff"))
    elig_of = {(r["game_id"], r["player_id"]): r["elig"] for r in prows.filter(pl.col("elig") != "").iter_rows(named=True)}
    debug: dict = {}
    scored = {None: _score(models.base, vt, et, elig_of, games)}
    explained = [EX.explain_stack(models.base, vt, et, elig_of, LY.stack_label(None))]
    margin_terms = pl.DataFrame()
    if "4a_injuries" in active:
        future = TG.team_rows(sched, TG.schedule_frame(sched).filter((pl.col("season") == season) & (pl.col("week") > week))).select("season", "week", "team")
        inp = H.read_input_injury_inputs(cutoff, live_db, trows.select("season", "week", "team", "game_id", "gameday"), future_games=future)
        fit_cutoff = replay.get("fit_cutoff") or min(cutoff, FIT_FLOOR)
        feats = inf.build_live_injury_features(inp.player_weeks, inp.frame, inp.injuries, inp.rosters, trows.select("game_id", "team", "gameday"), season, week, cutoff, fit_cutoff,
                                               now, status_clock=replay.get("status_clock"))
        opp = trows.select("game_id", "team", "opponent")
        vt_i, et_i = inf.attach(vt, feats, opp), inf.attach(et, feats, opp)
        debug["injury"] = dict(feats=feats, vt=vt_i, et=et_i)
        scored["4a_injuries"] = _score(models.injury, vt_i, et_i, elig_of, games)
        explained.append(EX.explain_stack(models.injury, vt_i, et_i, elig_of, LY.stack_label("4a_injuries")))
    if "4b_simulation" in active:
        floor_key = season * 100 + week if replay else min(season * 100 + week, config.HOLDOUT_SEASON * 100 + 1)       # scaling terms fit on 2020-2024 games only
        gsin = H.read_input_game_state(cutoff, live_db, fit_floor=replay.get("fit_cutoff") or min(cutoff, FIT_FLOOR))
        ge = LS.game_expectations_live(gsin.features, gsin.games, gids, season, week, floor_key)
        hh = ge.with_columns(input_hash=pl.struct("margin_mean", "margin_sd").hash(seed=0).cast(pl.String).str.slice(0, 16))
        sp = LS.spread_rows(ge).join(hh.select("game_id", "input_hash"), on="game_id").with_columns(player_id=pl.lit(None, dtype=pl.String), market=pl.lit("spread"))
        scored["4b_simulation"] = sp.select("game_id", "player_id", "market", "mean", "input_hash")
        debug["sim"] = dict(expectations=ge)
        margin_terms = LS.margin_terms_live(gsin.features, gsin.games, gids, season, week, floor_key).with_columns(stack=pl.lit(LY.stack_label("4b_simulation")))
    k = games.select("game_id", "kickoff_time")
    mv = version or model_version(layer_weights_csv)
    parts = []
    for layer, df in scored.items():
        keep = df if layer is None else df.filter(pl.col("market").is_in(sorted(active[layer])))
        if layer is None:                                              # base predictions for every market a layer does not own
            owned = {m for ms in active.values() for m in ms}
            keep = df.filter(~pl.col("market").is_in(sorted(owned)))
        parts.append(keep.with_columns(stack=pl.lit(LY.stack_label(layer))))
    preds = pl.concat(parts).join(k, on="game_id").with_columns(model_version=pl.lit(mv), input_hash=pl.col("input_hash").fill_null("none"))
    rid = run_id or now.strftime("run_%Y%m%dT%H%M%SZ")
    res = ST.PredictionStore(store_dir).write(preds.select("game_id", "player_id", "market", "kickoff_time", "model_version", "stack", "mean", "input_hash"), now=now, run_id=rid)
    expl = pl.concat([e for e in explained if e.height], how="diagonal_relaxed") if any(e.height for e in explained) else pl.DataFrame()
    if expl.height:
        (store_dir / EXPLAIN_DIR).mkdir(parents=True, exist_ok=True)
        expl.with_columns(run_id=pl.lit(rid)).write_parquet(store_dir / EXPLAIN_DIR / f"{rid}.parquet")
        if margin_terms.height:
            margin_terms.with_columns(run_id=pl.lit(rid)).write_parquet(store_dir / EXPLAIN_DIR / f"{rid}_margin.parquet")
    if lines_path is not None:
        LN.log_lines(games.select("game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline"), captured_at=now, path=lines_path)
    manifest = dict(run_id=rid, run_timestamp=now.isoformat(), season=season, week=week, model_version=mv, n_games=len(gids), cutoff=str(cutoff),
                    layers={k_ or "phase3": sorted(active.get(k_, [])) for k_ in scored}, stacks=preds["stack"].value_counts().sort("stack").rows(),
                    history=dict(player_rows=hist.player_log.height, team_rows=hist.team_log.height, newest_game=str(hist.team_log["gameday"].max()),
                                 through_season=hist.through_season),
                    chart=dict(rows=chart.height, teams=chart["team"].n_unique()),
                    schedule=dict(rows=sched.height, newest_pull=str(sched["pulled_at"].max())),
                    written=res.written, refused=res.refused, unchanged=res.unchanged,
                    train_rows=models.base.n_train)
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / f"manifest_{rid}.json").write_text(json.dumps(manifest, indent=1, default=str))
    return dict(status="ok", written=res.written, refused=res.refused, unchanged=res.unchanged, season=season, week=week, manifest=manifest, predictions=preds, debug=debug, explanations=expl, margin_terms=margin_terms)
