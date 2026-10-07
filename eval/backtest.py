"""Walk-forward backtest harness.

For each regular-season week W of the backtest seasons the harness gives a predictor
ONLY the rows from games before W's first kickoff, asks it to predict every target
game in W, and stores prediction + actual. The harness never loads, filters in, or
scores a season outside config.BACKTEST_SEASONS (the locked 2025 holdout and later
are excluded in SQL, then re-checked).

Plugging in a predictor
-----------------------
A predictor is any callable
        predictor(history: History, targets: list, cutoff: date) -> list[dict]
returning, for each target in order, a dict {market: value-or-None}. `history` holds
only games with gameday < cutoff. A fitted model fits on `history` inside the call.
Player targets are PlayerTarget, game targets GameTarget (eval/baselines.py); pass one
predictor for each kind (either may be None). Baselines are adapted by
`baseline_predictors(method)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

import config
from eval import baselines as bl

PLAYER, GAME = "player", "game"
MAX_BACKTEST_SEASON = max(config.BACKTEST_SEASONS)


@dataclass(frozen=True)
class History:
    player_log: pl.DataFrame
    team_log: pl.DataFrame


@dataclass
class BacktestData:
    player_log: pl.DataFrame   # every player-game in the backtest seasons
    team_log: pl.DataFrame     # every team-game in the backtest seasons
    weeks: list                # [(season, week, cutoff_date)] in chronological order
    game_extras: pl.DataFrame | None = None  # optional extra per-game actuals: game_id + numeric columns


def assert_no_holdout(*frames: pl.DataFrame):
    for f in frames:
        if f.height and f["season"].max() >= config.HOLDOUT_SEASON:
            raise RuntimeError(f"data from season {f['season'].max()} reached the harness; "
                               f"{config.HOLDOUT_SEASON}+ is the locked holdout")


def load_backtest_data(raw_db=config.RAW_DUCKDB_PATH) -> BacktestData:
    tl = bl.build_team_game_log(raw_db, max_season=MAX_BACKTEST_SEASON)
    pl_log = bl.build_player_game_log(raw_db, max_season=MAX_BACKTEST_SEASON)
    assert_no_holdout(tl, pl_log)
    return _make_data(pl_log, tl)


def _make_data(player_log, team_log) -> BacktestData:
    wk = (team_log.group_by("season", "week").agg(cutoff=pl.col("gameday").min()).sort("season", "week"))
    return BacktestData(player_log, team_log, [(r["season"], r["week"], r["cutoff"]) for r in wk.iter_rows(named=True)])


# --- who gets scored (pregame-known criteria, identical for every predictor) -----------------
# QB markets: the team's QB1. Rushing: RB slots 1-2. Receiving: WR slots 1-3, TE and RB slots 1-2.
# A player who does not appear in the game's stats has no row, so "didn't play" games are not scored.
_PASS = ("pass_att", "pass_cmp", "pass_yds")
_RUSH = ("rush_att", "rush_yds")
_RECV = ("targets", "rec", "rec_yds")


def eligible_markets(family: str, slot: int) -> tuple:
    if family == "QB":
        return _PASS if slot == 1 else ()
    if family == "RB":
        return (_RUSH + _RECV) if slot in (1, 2) else ()
    if family == "WR":
        return _RECV if slot in (1, 2, 3) else ()
    if family == "TE":
        return _RECV if slot in (1, 2) else ()
    return ()


def _week_targets(data: BacktestData, season: int, week: int):
    extras = {}
    if data.game_extras is not None:
        extras = {r["game_id"]: {k: v for k, v in r.items() if k != "game_id"}
                  for r in data.game_extras.iter_rows(named=True)}
    games = data.player_log.filter((pl.col("season") == season) & (pl.col("week") == week))
    ptargets, pactual = [], []
    for r in games.iter_rows(named=True):
        mk = eligible_markets(r["family"], r["slot"])
        if not mk:
            continue
        ptargets.append(bl.PlayerTarget(r["player_id"], r["gameday"], season, r["team_game_num"], r["opponent"],
                                        r["family"], r["slot"], r["team"], week))
        pactual.append({m: r[bl.PLAYER_MARKETS[m]] for m in mk})
    tl = data.team_log.filter((pl.col("season") == season) & (pl.col("week") == week))
    gtargets, gactual, gids = [], [], []
    home = tl.filter(pl.col("is_home"))
    away = tl.filter(~pl.col("is_home")).select("game_id", a_num="team_game_num")
    for r in home.join(away, on="game_id").iter_rows(named=True):
        gtargets.append(bl.GameTarget(r["gameday"], season, r["team"], r["opponent"], r["team_game_num"], r["a_num"]))
        margin = r["pf"] - r["pa"]
        gactual.append({"spread": margin, "total": r["pf"] + r["pa"],
                        "moneyline": None if margin == 0 else float(margin > 0),  # ties dropped
                        **extras.get(r["game_id"], {})})
        gids.append(r["game_id"])
    return ptargets, pactual, gtargets, gactual, gids


def walk_forward(name: str, data: BacktestData, player_predictor=None, game_predictor=None,
                 seasons=config.BACKTEST_SEASONS) -> pl.DataFrame:
    """Run one predictor through every week; returns one row per (target, market).

    Columns: method, kind, market, season, week, gameday, entity, prediction, actual.
    """
    if not set(seasons) <= set(config.BACKTEST_SEASONS):
        raise ValueError(f"seasons {sorted(set(seasons) - set(config.BACKTEST_SEASONS))} are outside the backtest "
                         f"seasons {config.BACKTEST_SEASONS}; the {config.HOLDOUT_SEASON} holdout is locked")
    assert_no_holdout(data.player_log, data.team_log)
    rows = []
    for season, week, cutoff in data.weeks:
        if season not in seasons:
            continue
        history = History(data.player_log.filter(pl.col("gameday") < cutoff),
                          data.team_log.filter(pl.col("gameday") < cutoff))
        ptargets, pactual, gtargets, gactual, gids = _week_targets(data, season, week)
        if player_predictor and ptargets:
            _check(ptargets, cutoff)
            preds = player_predictor(history, ptargets, cutoff)
            for t, p, a in zip(ptargets, preds, pactual):
                for m, y in a.items():
                    if m in p:  # a predictor only gets scored on the markets it returns
                        rows.append((name, PLAYER, m, season, week, t.gameday, t.player_id, p.get(m), float(y)))
        if game_predictor and gtargets:
            _check(gtargets, cutoff)
            preds = game_predictor(history, gtargets, cutoff)
            for t, gid, p, a in zip(gtargets, gids, preds, gactual):
                for m, y in a.items():
                    if y is not None and m in p:
                        rows.append((name, GAME, m, season, week, t.gameday, gid, p.get(m), float(y)))
    schema = {"method": pl.String, "kind": pl.String, "market": pl.String, "season": pl.Int32, "week": pl.Int32,
              "gameday": pl.Date, "entity": pl.String, "prediction": pl.Float64, "actual": pl.Float64}
    return pl.DataFrame(rows, schema=schema, orient="row")


def _check(targets, cutoff: date):
    if any(t.gameday < cutoff for t in targets):
        raise RuntimeError("a target game is dated before the week's cutoff")


# --- baseline adapters ---------------------------------------------------------
def baseline_predictors(method: str):
    """(player_predictor, game_predictor) for one of eval.baselines.METHODS."""
    def player(history, targets, cutoff):
        return [bl.predict_player(method, history.player_log, t, cutoff) for t in targets]

    def game(history, targets, cutoff):
        return [bl.predict_game(method, history.team_log, t, cutoff) for t in targets]

    return player, game
