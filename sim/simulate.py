"""The game simulation (Phase 4b.4): 20,000 simulated games per real game (2,000 in backtest mode).

For each simulated game, in this order (every step vectorised over the simulated games of one real game):
 1. margin ~ Normal(4b.2 mean, 4b.2 SD) and total ~ Normal(4b.2 mean, 4b.2 SD); final scores follow: home = (total + margin) / 2.
 2. margin -> the five state shares of plays, through the FINAL-MARGIN curve of 4b.3 (a drawn margin stands in for a realised one).
 3. total plays per team ~ Normal(expected plays, walk-forward residual SD), rounded.
 4. plays are split over the states (multinomial of the shares); dropbacks in each state ~ Binomial(plays in the state, the team's
    shrunk dropback rate in that state); sacks ~ Binomial(dropbacks, sack rate); scrambles ~ Binomial(dropbacks - sacks, scramble rate
    given no sack). The accounting is the exact one of 4b.3: pass attempts = dropbacks - sacks - scrambles, rush attempts =
    plays - dropbacks + scrambles.
 5. ONE shared passing-efficiency shock and ONE shared rushing-efficiency shock per team (standard normal), loaded onto every
    efficiency quantity with that quantity's RELATIVE shock SD (sim/inputs.py: set from the historical covariance of same-team
    residuals), so players on the same team move together.
 6. each player's volume. VOLUME_SOURCE "phase3" (the default): his expected volume IS the Phase 3 volume model's walk-forward prediction
    for that player-game (pass attempts, carries, targets, QB kneel-free carries -- it is conditional on his playing, like the scored
    rows). The simulation adds variance and correlation around that anchor: his anchor share of the team total (prediction / the
    simulated team's mean total) times the simulated team total, drawn Dirichlet-multinomial (kappa estimated from how far actual counts
    sat from the Phase 3 anchors), so counts add up to the team totals exactly and move with the game script; his early exit (4a.3)
    scales his share. The 4a.2 redistribution is a SECONDARY signal in exactly two places: (a) the injury ratio -- in a simulated game in
    which a teammate is out, an anchored player's share is multiplied by (his 4a.2 share with that teammate out) / (his 4a.2 share with
    everyone in), clipped to [0, 3] and used only where his baseline share is at least 2%; Phase 3 has no injury information, so it
    cannot reflect the change itself; with nobody likely out the ratio is 1 and the anchor stands untouched -- and (b) players with no
    Phase 3 volume prediction for the stat (RB3, WR4, a QB's receptions...) keep their 4a.2 shares. Reconciliation: anchored players
    keep their anchor shares and the un-anchored players and the unlisted `rest` fill what is left of the team total; if the anchors
    alone exceed the team total they are scaled down proportionally (so counts always sum to the team total exactly).
    VOLUME_SOURCE "shares" is the previous design: each player's share of the team total comes from 4a.2 alone.
    Either way: his early exit (4a.3: P(exit) from his history, share of the game completed drawn from his position's empirical
    distribution) scales his expected share; shares are then drawn Dirichlet-multinomial (kappa estimated from history) over the team's
    carries (rush attempts x the league carries-per-rush ratio), targets (pass attempts x the league targets-per-attempt ratio) and
    dropbacks, so counts add up to the team totals exactly. Efficiency = the Phase 3 efficiency prediction x (1 + shock) + a player
    residual drawn from the empirical walk-forward residuals of that quantity and position family, rescaled by the simulated
    denominator. Outcomes: pass attempts (his dropbacks x team attempts / team dropbacks), completions, passing yards, rush
    attempts and yards, targets, receptions, receiving yards, and the QB rushing pair (kneel-free carries).
 7. a fixed random stream per game: np.random.default_rng([RUN_SEED, crc32(game_id)]); a rerun is identical.
 8. storage is once per game (store_game): sim_games (one row per simulated game) and sim_players (one row per simulated game and
    player), keyed simulation_run_id, game_id, simulation_id. Every prop for a game reads off that one stored set.
 9. SimMode: FULL (20,000) and BACKTEST (2,000, flagged backtest = True in every output).

Known simplifications (stated, not hidden): the walk-forward constants (dropback rates, sack / scramble rates, state curve) are used
as given, so their own estimation noise is not propagated; the two teams' plays and the margin / total draws are independent; the
4a.2 expected shares are a probability-weighted mixture (a Questionable player is partly present in every simulated game, not
absent in a share of them); completions are not forced to equal the receivers' receptions; the home / away points are real-valued.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

import config
from models import game_state as gs
from sim import inputs as si

RUN_SEED = 20260101
PLAYER_STATS = ("dropbacks", "pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds", "targets", "rec", "rec_yds", "qb_rush_att", "qb_rush_yds")
STAT_INDEX = {s: i for i, s in enumerate(PLAYER_STATS)}
COUNT_STATS = ("dropbacks", "pass_att", "rush_att", "targets", "qb_rush_att")
MARKET_STAT = {m: m for m in ("pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds", "targets", "rec", "rec_yds", "qb_rush_att", "qb_rush_yds")}
VOLUME_SOURCE = "phase3"          # "phase3": player volume anchored on the Phase 3 volume model; "shares": 4a.2 shares alone (the previous design)
INJURY_RATIO_MIN_BASE = 0.02      # the 4a.2 injury ratio is only applied to a player whose in-play baseline share is at least this
INJURY_RATIO_CAP = 3.0
MIN_PLAYS, MAX_PLAYS = 30, 110
MIN_TOTAL = 10.0
MIN_PRESENT = 100        # simulated games a player must be in before a prediction conditional on his playing is made


@dataclass(frozen=True)
class SimMode:
    name: str
    n_sim: int
    backtest: bool


FULL = SimMode("full_20000", 20_000, False)
BACKTEST = SimMode("backtest_2000", 2_000, True)


def run_id(mode: SimMode, seed: int = RUN_SEED, model_version: str = "phase4b_v1") -> str:
    """Deterministic simulation_run_id: the same mode, seed and model version always give the same id."""
    h = hashlib.sha256(json.dumps([mode.name, mode.n_sim, seed, model_version]).encode()).hexdigest()[:10]
    return f"sim_{mode.name}_{h}"


def game_rng(game_id: str, seed: int = RUN_SEED) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(game_id.encode())])


@dataclass
class TeamSim:
    team: str
    present: np.ndarray                 # (n_sim, K) False in the simulated games where that player is out
    plays: np.ndarray
    dropbacks: np.ndarray
    sacks: np.ndarray
    scrambles: np.ndarray
    pass_att: np.ndarray
    rush_att: np.ndarray
    targets: np.ndarray
    carries: np.ndarray
    points: np.ndarray


@dataclass
class GameSim:
    game_id: str
    mode: SimMode
    margin: np.ndarray
    total: np.ndarray
    home: TeamSim
    away: TeamSim
    player_ids: list
    teams: list
    roles: list
    stats: np.ndarray                   # (n_sim, n_players, len(PLAYER_STATS)) float32
    present: np.ndarray                 # (n_sim, n_players) bool: False where the player is out of that simulated game


# ====================================================================== one team
def _team_volume(t: si.TeamInput, tm: np.ndarray, cal: si.WeekCalib, rng: np.random.Generator) -> dict:
    n = len(tm)
    shares5 = gs.state_shares(cal.state_model, tm)
    plays = np.clip(np.rint(rng.normal(t.exp_plays, cal.plays_sd, n)), MIN_PLAYS, MAX_PLAYS).astype(np.int64)
    in_state = rng.multinomial(plays, shares5)
    db_state = rng.binomial(in_state, np.clip(t.db_rate, 0.0, 1.0))
    D = db_state.sum(axis=1)
    S = rng.binomial(D, min(max(t.sack_rate, 0.0), 1.0))
    p_scr = min(max(t.scramble_rate / max(1.0 - t.sack_rate, 1e-9), 0.0), 1.0)
    C = rng.binomial(D - S, p_scr)
    P = D - S - C
    R = plays - D + C
    return dict(plays=plays, D=D, S=S, C=C, P=P, R=R, T=np.rint(P * cal.target_ratio).astype(np.int64),
                Rc=np.rint(R * cal.carry_ratio).astype(np.int64), zp=rng.standard_normal(n), zr=rng.standard_normal(n))


def _exit_factors(t: si.TeamInput, exit_pools: dict, exit_all: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, K) factor on each player's share: 1, or the share of the game he completes when he exits early (4a.3)."""
    K = len(t.player_ids)
    f = np.ones((n, K))
    if not K:
        return f
    e = rng.random((n, K)) < t.p_exit[None, :]
    grp = np.array([g if g in exit_pools and len(exit_pools[g]) >= 20 else "_all" for g in t.groups])
    for g in sorted(set(grp.tolist())):
        idx = np.flatnonzero(grp == g)
        pool = exit_all if g == "_all" else exit_pools[g]
        draws = pool[rng.integers(0, len(pool), (n, len(idx)))]
        f[:, idx] = np.where(e[:, idx], draws, 1.0)
    return f


def _dirichlet_multinomial(shares: np.ndarray, factor: np.ndarray, kappa: float, total: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Counts (n, K + 1) of `total` draws over K players + `rest`: shares (K + 1,) or per-simulation (n, K + 1) x early-exit factor,
    Dirichlet(kappa * share) mixed, multinomial."""
    n, K = factor.shape
    sh = np.broadcast_to(shares, (n, K + 1))
    w = np.empty((n, K + 1))
    w[:, :K] = sh[:, :K] * factor
    w[:, K] = sh[:, K]
    w = np.maximum(w, 0.0)
    s = w.sum(axis=1, keepdims=True)
    p = np.where(s > 0, w / np.where(s > 0, s, 1.0), 1.0 / (K + 1))
    g = np.where(p > 0, rng.gamma(np.maximum(kappa * p, 1e-4)), 0.0)
    pv = g / g.sum(axis=1, keepdims=True)
    return rng.multinomial(total, pv)


def _anchored_counts(play: np.ndarray, scen: np.ndarray, anchor: np.ndarray, present: np.ndarray, factor: np.ndarray, kappa: float,
                     total: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Counts (n, K + 1) of `total` draws over K players + `rest`, centred on the Phase 3 anchors.

    play    (K + 1,) 4a.2 in-play shares (everyone plays);  scen (n, K + 1) 4a.2 shares in each simulated game's availability scenario
    anchor  (K,) Phase 3 prediction / expected team total (nan = no prediction: the player follows the 4a.2 scenario share)
    present (n, K) False where the player is out of that simulated game;  factor (n, K) early-exit factor
    Anchored players keep anchor x injury ratio x exit factor; the others and `rest` share what is left (see the module docstring)."""
    n, K = factor.shape
    has = ~np.isnan(anchor)
    ratio = np.ones((n, K))
    ok = has & (play[:K] >= INJURY_RATIO_MIN_BASE)
    if ok.any():
        ratio[:, ok] = np.clip(scen[:, :K][:, ok] / play[:K][ok], 0.0, INJURY_RATIO_CAP)
    a = np.where(has, anchor, 0.0)[None, :] * ratio * factor * present
    a_sum = a.sum(axis=1)
    a = np.where((a_sum > 1.0)[:, None], a / np.where(a_sum > 1.0, a_sum, 1.0)[:, None], a)
    remainder = np.clip(1.0 - a.sum(axis=1), 0.0, None)
    w = np.zeros((n, K + 1))
    w[:, :K] = np.where(has[None, :], 0.0, scen[:, :K] * factor)
    w[:, K] = scen[:, K]
    w = np.clip(w, 0.0, None)
    ws = w.sum(axis=1)
    w = np.where((ws > 0)[:, None], w / np.where(ws > 0, ws, 1.0)[:, None] * remainder[:, None], 0.0)
    w[:, K] += np.where(ws > 0, 0.0, remainder)
    p = w
    p[:, :K] += a
    g = np.where(p > 0, rng.gamma(np.maximum(kappa * p, 1e-4)), 0.0)          # a player with no share (out, or none) gets exactly none
    pv = g / g.sum(axis=1, keepdims=True)
    return rng.multinomial(total, pv)


def draw_efficiency(pred: np.ndarray, shock: float, team_z: np.ndarray, pools: list, den: np.ndarray, bounds: tuple, rng: np.random.Generator) -> np.ndarray:
    """(n, m) efficiency draws for m players: prediction x (1 + shock x the team's shared z) + a standardised residual drawn from the
    player's empirical pool and rescaled by sqrt(simulated denominator). `pools[j]` None = no player-level noise for player j."""
    n = len(team_z)
    z = np.zeros((n, len(pred)))
    for j, pool in enumerate(pools):
        if pool is not None:
            z[:, j] = pool[rng.integers(0, len(pool), n)]
    eff = pred[None, :] * (1.0 + shock * team_z[:, None]) + z / np.sqrt(np.maximum(den, 1))
    return np.clip(eff, *bounds)


def _anchors(t: si.TeamInput, v: dict, cal: si.WeekCalib, fam: list) -> dict:
    """stat -> (K,) Phase 3 anchor as a share of the simulated team's MEAN total (nan where the player has no prediction for the stat)."""
    K = len(t.player_ids)
    out = {}
    is_qb = np.array([f_ == "QB" for f_ in fam])
    if t.vol is None:
        return {st: np.full(K, np.nan) for st in si.STATS}
    nbar = {"carry": max(float(v["Rc"].mean()), 1.0), "target": max(float(v["T"].mean()), 1.0), "dropback": max(float(v["P"].mean()), 1.0)}
    rush = np.where(is_qb, t.vol["qb_rush_att"] / max(cal.qb_keep, 1e-6), t.vol["rush_att"])      # carries include kneels; the QB anchor is kneel-free
    out["carry"] = rush / nbar["carry"]
    out["target"] = t.vol["targets"] / nbar["target"]
    out["dropback"] = np.where(is_qb, t.vol["pass_att"], np.nan) / nbar["dropback"]
    return out


def _simulate_team(t: si.TeamInput, tm: np.ndarray, g: si.GameInput, rng: np.random.Generator, volume_source: str = "phase3") -> tuple[TeamSim, np.ndarray]:
    cal = g.calib
    n = len(tm)
    v = _team_volume(t, tm, cal, rng)
    K = len(t.player_ids)
    # availability: each uncertain player is out in a share q of the simulated games; shares = everyone-plays shares + the redistribution
    # that follows from whoever is out (4a.2 scenarios, superposed). Without scenario inputs the 4a.2 mixture shares are used as given.
    present = np.ones((n, K), dtype=bool)
    scen_shares = None
    if t.play_shares is not None and t.scen_q is not None and len(t.scen_q):
        out_flag = rng.random((n, len(t.scen_q))) < t.scen_q[None, :]
        for m, k in enumerate(t.scen_k):
            if k >= 0:
                present[:, k] = ~out_flag[:, m]
        shares = {st: np.clip(t.play_shares[st][None, :] + out_flag.astype(float) @ t.scen_delta[st], 0.0, None) for st in si.STATS}
        scen_shares = shares
    else:
        shares = t.play_shares if t.play_shares is not None else t.shares
    f = _exit_factors(t, g.exit_pools, g.exit_all, n, rng)
    fam = [si.FAMILY_OF_GROUP.get(gr) or si.FAMILY_OF_ROLE.get(ro, "other") for gr, ro in zip(t.groups, t.roles)]
    if volume_source == "phase3" and t.vol is not None:
        base = t.play_shares if t.play_shares is not None else t.shares
        sc = scen_shares if scen_shares is not None else {st: np.broadcast_to(base[st], (n, K + 1)) for st in si.STATS}
        anchors = _anchors(t, v, cal, fam)
        carries = _anchored_counts(base["carry"], sc["carry"], anchors["carry"], present, f, cal.kappa_anchor["carry"], v["Rc"], rng)[:, :K]
        targets = _anchored_counts(base["target"], sc["target"], anchors["target"], present, f, cal.kappa_anchor["target"], v["T"], rng)[:, :K]
        dropbacks = _anchored_counts(base["dropback"], sc["dropback"], anchors["dropback"], present, f, cal.kappa_anchor["dropback"], v["D"], rng)[:, :K]
    else:
        carries = _dirichlet_multinomial(shares["carry"], f, cal.kappa["carry"], v["Rc"], rng)[:, :K]
        targets = _dirichlet_multinomial(shares["target"], f, cal.kappa["target"], v["T"], rng)[:, :K]
        dropbacks = _dirichlet_multinomial(shares["dropback"], f, cal.kappa["dropback"], v["D"], rng)[:, :K]
    # volumes always exist; completions / receptions / yardage need a Phase 3 efficiency prediction and stay NaN (no outcome) without one
    out = np.full((n, K, len(PLAYER_STATS)), np.nan, dtype=np.float32)
    for c in COUNT_STATS:
        out[:, :, STAT_INDEX[c]] = 0.0
    is_qb = np.array([f_ == "QB" for f_ in fam])
    pa = np.rint(dropbacks * (v["P"] / np.maximum(v["D"], 1))[:, None]).astype(np.int64)
    qb_att = np.rint(carries * cal.qb_keep).astype(np.int64)

    def draw(q: str, idx: np.ndarray, den: np.ndarray) -> np.ndarray:
        """Efficiency draw for the players `idx` given their simulated denominators `den` (n, len(idx))."""
        pools = []
        for k in idx:
            pool = cal.pool.get((q, fam[k]))
            pools.append(pool if pool is not None else cal.pool.get((q, "other")))
        return draw_efficiency(t.eff[q][idx], cal.shock[q], (v["zp"] if si.SHOCK_KIND[q] == "pass" else v["zr"]), pools, den, si.BOUNDS[q], rng)

    have = {q: np.flatnonzero(~np.isnan(t.eff[q])) for q in si.QUANTITIES}
    out[:, :, STAT_INDEX["dropbacks"]] = dropbacks
    out[:, :, STAT_INDEX["rush_att"]] = carries
    out[:, :, STAT_INDEX["targets"]] = targets
    out[:, :, STAT_INDEX["pass_att"]] = np.where(is_qb[None, :], pa, 0)
    out[:, :, STAT_INDEX["qb_rush_att"]] = np.where(is_qb[None, :], qb_att, 0)
    # passing: attempts -> completions -> yards
    idx = have["comp_rate"]
    if len(idx):
        cmp_ = np.rint(pa[:, idx] * draw("comp_rate", idx, pa[:, idx])).astype(np.int64)
        out[:, idx, STAT_INDEX["pass_cmp"]] = cmp_
        j = have["yds_per_cmp"]
        both = np.intersect1d(idx, j)
        if len(both):
            pos = np.searchsorted(idx, both)
            out[:, both, STAT_INDEX["pass_yds"]] = np.rint(cmp_[:, pos] * draw("yds_per_cmp", both, cmp_[:, pos]))
    # receiving: targets -> receptions -> yards
    idx = have["catch_rate"]
    if len(idx):
        rec = np.rint(targets[:, idx] * draw("catch_rate", idx, targets[:, idx])).astype(np.int64)
        out[:, idx, STAT_INDEX["rec"]] = rec
        both = np.intersect1d(idx, have["yds_per_rec"])
        if len(both):
            pos = np.searchsorted(idx, both)
            out[:, both, STAT_INDEX["rec_yds"]] = np.rint(rec[:, pos] * draw("yds_per_rec", both, rec[:, pos]))
    # rushing yards: backs / receivers on carries, quarterbacks on kneel-free carries
    idx = have["ypc"]
    if len(idx):
        out[:, idx, STAT_INDEX["rush_yds"]] = np.rint(carries[:, idx] * draw("ypc", idx, carries[:, idx]))
    idx = have["qb_ypc"]
    if len(idx):
        out[:, idx, STAT_INDEX["qb_rush_yds"]] = np.rint(qb_att[:, idx] * draw("qb_ypc", idx, qb_att[:, idx]))
    team = TeamSim(team=t.team, present=present, plays=v["plays"], dropbacks=v["D"], sacks=v["S"], scrambles=v["C"], pass_att=v["P"], rush_att=v["R"],
                   targets=v["T"], carries=v["Rc"], points=np.zeros(n))
    return team, out


def simulate_game(g: si.GameInput, mode: SimMode = BACKTEST, seed: int = RUN_SEED, volume_source: str = VOLUME_SOURCE) -> GameSim:
    """All simulated games of one real game, in the order of the module docstring, from this game's own fixed random stream."""
    rng = game_rng(g.game_id, seed)
    n = mode.n_sim
    margin = rng.normal(g.margin_mean, g.margin_sd, n)
    total = np.maximum(rng.normal(g.total_mean, g.total_sd, n), MIN_TOTAL)
    home_ts, home_out = _simulate_team(g.home, margin, g, rng, volume_source)
    away_ts, away_out = _simulate_team(g.away, -margin, g, rng, volume_source)
    home_ts.points, away_ts.points = (total + margin) / 2.0, (total - margin) / 2.0
    stats = np.concatenate([home_out, away_out], axis=1)
    return GameSim(game_id=g.game_id, mode=mode, margin=margin, total=total, home=home_ts, away=away_ts,
                   player_ids=[*g.home.player_ids, *g.away.player_ids], teams=[*([g.home.team] * len(g.home.player_ids)), *([g.away.team] * len(g.away.player_ids))],
                   roles=[*g.home.roles, *g.away.roles], stats=stats,
                   present=np.concatenate([home_ts.present, away_ts.present], axis=1))


# ====================================================================== markets from simulated outcomes
def market_summary(values: np.ndarray, rungs) -> dict:
    """Mean, median, SD and P(X > rung) for each ladder rung of one market's simulated outcomes."""
    out = dict(sim_mean=float(values.mean()), sim_median=float(np.median(values)), sim_sd=float(values.std()),
               sim_q05=float(np.quantile(values, 0.05)), sim_q95=float(np.quantile(values, 0.95)))
    for r in rungs:
        out[f"p_over_{r}"] = float((values > r).mean())
    return out


def summarize_game(gsim: GameSim, eligible: dict, ladders: dict | None = None) -> list:
    """Rows (kind, market, entity, summary...) for every eligible player market of the game and the three game markets.
    Player rows are CONDITIONAL on the player being in the game (the simulated games where he is out are dropped), because a player-game
    only exists, and is only scored, when he played; p_play is the share of simulated games he is in.

    eligible: player_id -> iterable of markets the player-game is scored for."""
    ladders = ladders or config.LADDERS
    rows = []
    for k, pid in enumerate(gsim.player_ids):
        pres = gsim.present[:, k]
        for m in sorted(eligible.get(pid, ())):
            if m not in MARKET_STAT:
                continue
            v = gsim.stats[:, k, STAT_INDEX[MARKET_STAT[m]]][pres]
            if len(v) < MIN_PRESENT or np.isnan(v).any():
                continue          # out in (nearly) every simulated game, or no efficiency prediction: no conditional prediction
            rows.append(dict(kind="player", market=m, entity=pid, team=gsim.teams[k], p_play=float(pres.mean()), **market_summary(v, ladders[m])))
    rows.append(dict(kind="game", market="spread", entity=gsim.game_id, team=None, **market_summary(gsim.margin, ladders["spread"])))
    rows.append(dict(kind="game", market="total", entity=gsim.game_id, team=None, **market_summary(gsim.total, ladders["total"])))
    p_win = float((gsim.margin > 0).mean())
    rows.append(dict(kind="game", market="moneyline", entity=gsim.game_id, team=None, sim_mean=p_win, sim_median=float(p_win >= 0.5), sim_sd=None,
                     sim_q05=None, sim_q95=None, p_over_win=p_win))
    return rows


# ====================================================================== storage: once per game
def store_game(gsim: GameSim, out_dir, mode: SimMode | None = None) -> dict:
    """Write one game's whole simulated set: sim_games (one row per simulated game) and sim_players (one row per simulated game and
    player), keyed simulation_run_id, game_id, simulation_id. Returns the two paths. Never one row per individual prediction."""
    mode = mode or gsim.mode
    rid = run_id(mode)
    n = len(gsim.margin)
    out_dir = Path(out_dir) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = np.arange(n, dtype=np.int32)
    games = pl.DataFrame({
        "simulation_run_id": [rid] * n, "game_id": [gsim.game_id] * n, "simulation_id": sid, "backtest_mode": [mode.backtest] * n,
        "home_points": gsim.home.points.astype(np.float32), "away_points": gsim.away.points.astype(np.float32),
        "margin": gsim.margin.astype(np.float32), "total": gsim.total.astype(np.float32),
        "home_plays": gsim.home.plays.astype(np.int16), "away_plays": gsim.away.plays.astype(np.int16),
        "home_dropbacks": gsim.home.dropbacks.astype(np.int16), "away_dropbacks": gsim.away.dropbacks.astype(np.int16),
        "home_sacks": gsim.home.sacks.astype(np.int16), "away_sacks": gsim.away.sacks.astype(np.int16),
        "home_scrambles": gsim.home.scrambles.astype(np.int16), "away_scrambles": gsim.away.scrambles.astype(np.int16),
        "home_pass_att": gsim.home.pass_att.astype(np.int16), "away_pass_att": gsim.away.pass_att.astype(np.int16),
        "home_rush_att": gsim.home.rush_att.astype(np.int16), "away_rush_att": gsim.away.rush_att.astype(np.int16)})
    P = len(gsim.player_ids)
    cols = {"simulation_run_id": [rid] * (n * P), "game_id": [gsim.game_id] * (n * P), "simulation_id": np.repeat(sid, P),
            "player_id": np.tile(np.array(gsim.player_ids, dtype=object), n), "team": np.tile(np.array(gsim.teams, dtype=object), n)}
    for j, s in enumerate(PLAYER_STATS):
        cols[s] = (gsim.stats[:, :, j].reshape(-1)).astype(np.int16 if s in COUNT_STATS else np.float32)
    cols["present"] = gsim.present.reshape(-1)
    players = pl.DataFrame(cols)
    gp, pp = out_dir / f"sim_games_{gsim.game_id}.parquet", out_dir / f"sim_players_{gsim.game_id}.parquet"
    games.write_parquet(gp, compression="zstd")
    players.write_parquet(pp, compression="zstd")
    return {"games": gp, "players": pp}
