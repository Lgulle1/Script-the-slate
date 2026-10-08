"""Walk-forward inputs and calibrations for the game simulation (sim/simulate.py).

Everything here is a function of games BEFORE the simulated game's week (the week's first kickoff is the cutoff); nothing
reads the game being simulated or any later game. 2025+ is never loaded.

Per simulated game (GameInput):
  * margin / total mean and SD ............ models/game_state.py (4b.2); the SDs are RMS walk-forward errors
  * team expected plays and plays SD ....... 4b.2 expected plays; SD = RMS of (plays - expected plays) on earlier team-games
  * state shares ........................... the FINAL-MARGIN curve of 4b.3 (fit on the realised margins of earlier team-games): a
                                             drawn margin stands in for a realised outcome, so it maps through the realised-margin curve
  * dropback rate per state, sack and scramble rates ... 4b.3
  * players ................................ the pre-game team list of 4a.2 (depth-chart role holders + recent `other` players), their
                                             expected carry / target / dropback shares, P(early exit) and the exit model's position
                                             pools (4a.3), and the Phase 3 efficiency predictions for every player-game that is eligible
                                             for the quantity's market
Calibration (Calibration.week): constants estimated from earlier weeks only --
  * kappa (per stat): Dirichlet-multinomial concentration of a player's share of the team's carries / targets / dropbacks, from the
    dispersion of actual counts around the 4a.2 expected shares (method of moments, expanding window)
  * team shock SD (per efficiency quantity, RELATIVE to the prediction): from the historical covariance of the efficiency residuals
    of players on the same team-game (weighted pairs), so the simulated within-team covariance equals the historical one
  * residual pools (per quantity and position family): the empirical walk-forward residuals of the Phase 3 efficiency model, as
    standardised residual = residual * sqrt(denominator) so a draw can be rescaled by the simulated denominator, with the
    team-shock part taken out so the total variance is not counted twice
  * league ratios (targets per pass attempt, carries per play-based rush attempt, QB non-kneel share of carries)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

import config
from models import game_state as gs

QUANTITIES = ("comp_rate", "yds_per_cmp", "ypc", "qb_ypc", "catch_rate", "yds_per_rec")
DEN_COLUMN = {"comp_rate": "attempts", "yds_per_cmp": "completions", "ypc": "carries", "qb_ypc": "rush_att_ex_kneel",
              "catch_rate": "targets", "yds_per_rec": "receptions"}
SHOCK_SOURCE = {"catch_rate": "catch_rate", "yds_per_rec": "yds_per_rec", "comp_rate": "catch_rate", "yds_per_cmp": "yds_per_rec",
                "ypc": "ypc", "qb_ypc": "ypc"}   # a QB has no same-team pairs: his quantities borrow the receivers' / backs' loading
SHOCK_KIND = {"comp_rate": "pass", "yds_per_cmp": "pass", "catch_rate": "pass", "yds_per_rec": "pass", "ypc": "rush", "qb_ypc": "rush"}
STATS = ("carry", "target", "dropback")
FAMILY_OF_ROLE = {"QB1": "QB", "RB1": "RB", "RB2": "RB", "FB": "RB", "WR1": "WR", "WR2": "WR", "WR3": "WR", "slot": "WR", "TE1": "TE",
                  "TE2": "TE", "QB": "QB", "RB2FB": "RB", "other": "other"}
FAMILY_OF_GROUP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE"}
PAIR_WEIGHT_D0 = 3.0           # a residual with denominator d counts with weight d / (d + 3) in the team-covariance estimate
MIN_SHOCK_PAIR_WEIGHT = 300.0  # below this much pair weight the team shock is off (0)
MIN_POOL = 150                 # a (quantity, family) residual pool needs this many rows, else the quantity's pooled residuals
IDIO_FLOOR = 0.25              # the idiosyncratic part never drops below this share of the pooled variance
MIN_KAPPA_OBS = 400
KAPPA_MAX_P_OUT = 0.02
KAPPA_DEFAULT = {"carry": 30.0, "target": 30.0, "dropback": 300.0}   # carry / target: until MIN_KAPPA_OBS observations exist (early 2020); dropback: always
KAPPA_RANGE = (2.0, 5000.0)
MIN_PLAYS_SD_GAMES = 100
PLAYS_SD_DEFAULT = 9.0
BOUNDS = {"comp_rate": (0.0, 1.0), "catch_rate": (0.0, 1.0), "yds_per_cmp": (0.0, 60.0), "yds_per_rec": (-5.0, 80.0),
          "ypc": (-5.0, 40.0), "qb_ypc": (-5.0, 40.0)}
RATIO_DEFAULT = (0.96, 1.05, 0.95)


def key_of(season, week):
    return np.asarray(season, dtype=np.int64) * 100 + np.asarray(week, dtype=np.int64)


def _prefix(df: pl.DataFrame, key_col: str, cols: list) -> tuple:
    g = df.group_by(key_col).agg([pl.col(c).sum().cast(pl.Float64) for c in cols]).sort(key_col)
    return (g[key_col].to_numpy(), *[np.cumsum(g[c].to_numpy()) for c in cols])


def _before(keys: np.ndarray, key: int) -> int:
    """Number of key rows strictly before `key` (the prefix length that may be used)."""
    return int(np.searchsorted(keys, key, "left"))


# ====================================================================== calibration
@dataclass
class WeekCalib:
    key: int
    kappa: dict
    shock: dict                 # quantity -> relative SD of the shared team shock
    pool: dict                  # (quantity, family) -> scaled standardised residuals (np.ndarray, sorted)
    plays_sd: float
    target_ratio: float         # team targets per pass attempt
    carry_ratio: float          # player carries (kneels included) per play-based rush attempt
    qb_keep: float              # QB kneel-free carries / QB carries
    state_model: object         # gs.StateShareModel fit on the realised margins of earlier team-games


class Calibration:
    """Prefix-sum tables over (season, week) so any week's constants use only earlier weeks."""

    def __init__(self, res: pl.DataFrame, detail: pl.DataFrame, player_log: pl.DataFrame, tgs: pl.DataFrame, state_rows: pl.DataFrame | None = None):
        self._res = res.sort("key", "game_id", "team", "quantity")
        self._tgs = tgs
        self._state_rows = state_rows if state_rows is not None else tgs
        self._build_pairs()
        self._build_kappa(detail, player_log)
        self._build_plays(tgs)
        self._build_ratios(player_log, tgs)
        self._pools = {}
        for (q, fam), g in self._res.partition_by("quantity", "family", as_dict=True).items():
            self._pools[(q, fam)] = g
        self._state_cache: dict = {}

    # ---- team shock: pair statistics per team-game
    def _build_pairs(self):
        r = self._res.filter(pl.col("pred") > 0).with_columns(w=pl.col("den") / (pl.col("den") + PAIR_WEIGHT_D0),
                                                              rel=pl.col("res") / pl.col("pred"))
        self._pair = {}
        for q in sorted(set(SHOCK_SOURCE.values())):
            g = (r.filter(pl.col("quantity") == q).with_columns(wr=pl.col("w") * pl.col("rel"))
                 .group_by("key", "game_id", "team").agg(s1=pl.col("wr").sum(), s2=(pl.col("wr") ** 2).sum(), w1=pl.col("w").sum(),
                                                         w2=(pl.col("w") ** 2).sum())
                 .with_columns(num=(pl.col("s1") ** 2 - pl.col("s2")) / 2, den=(pl.col("w1") ** 2 - pl.col("w2")) / 2))
            self._pair[q] = _prefix(g, "key", ["num", "den"])

    def shock(self, quantity: str, key: int) -> float:
        k, num, den = self._pair[SHOCK_SOURCE[quantity]]
        i = _before(k, key)
        if not i or den[i - 1] < MIN_SHOCK_PAIR_WEIGHT:
            return 0.0
        return float(np.sqrt(max(num[i - 1] / den[i - 1], 0.0)))

    # ---- kappa: Pearson dispersion of counts around the expected shares
    def _build_kappa(self, detail: pl.DataFrame, player_log: pl.DataFrame):
        tot = player_log.group_by("game_id", "team").agg(n_carry=pl.col("carries").sum().cast(pl.Float64),
                                                         n_target=pl.col("targets").sum().cast(pl.Float64),
                                                         n_dropback=pl.col("attempts").sum().cast(pl.Float64))
        act = player_log.select("game_id", "team", "player_id", x_carry=pl.col("carries").cast(pl.Float64),
                                x_target=pl.col("targets").cast(pl.Float64), x_dropback=pl.col("attempts").cast(pl.Float64))
        # Only team-games with no real availability uncertainty (every listed player's absence probability below KAPPA_MAX_P_OUT): there the
        # mixture share IS the conditional share, so the dispersion left over is game-to-game noise in who gets the ball.
        calm = (detail.filter(pl.col("player_id") != "rest").group_by("game_id", "team").agg(worst=pl.col("p_out").max())
                .filter(pl.col("worst") < KAPPA_MAX_P_OUT).select("game_id", "team"))
        d = (detail.filter(pl.col("player_id") != "rest").join(calm, on=["game_id", "team"], how="inner")
             .join(tot, on=["game_id", "team"], how="inner").join(act, on=["game_id", "team", "player_id"], how="left"))
        self._kappa = {}
        for st in STATS:
            s, n = pl.col(f"b_{st}"), pl.col(f"n_{st}")
            x = pl.col(f"x_{st}").fill_null(0.0)
            t = (d.filter((s >= 0.05) & (s <= 0.95) & (n > 0))
                 .with_columns(term=(x - n * s) ** 2 / (n * s * (1 - s)), one=pl.lit(1.0), nn=n))
            self._kappa[st] = _prefix(t, "key", ["term", "one", "nn"])

    def kappa(self, stat: str, key: int) -> float:
        if stat == "dropback":
            # Not estimated: the only dropback shares between 0.05 and 0.95 are quarterback-availability games, whose dispersion is
            # availability uncertainty (carried by the mixture share itself), not game-to-game noise in a healthy starter's share.
            return KAPPA_DEFAULT["dropback"]
        k, t, c, n = self._kappa[stat]
        i = _before(k, key)
        if not i or c[i - 1] < MIN_KAPPA_OBS:
            return KAPPA_DEFAULT[stat]
        phi, nbar = t[i - 1] / c[i - 1], n[i - 1] / c[i - 1]
        if phi <= 1.0 + 1e-9:
            return KAPPA_RANGE[1]
        return float(np.clip((nbar - phi) / (phi - 1.0), *KAPPA_RANGE))

    # ---- team plays SD (RMS of earlier errors)
    def _build_plays(self, tgs: pl.DataFrame):
        t = tgs.filter(pl.col("exp_plays").is_not_null() & pl.col("plays").is_not_null()).with_columns(
            key=pl.col("season") * 100 + pl.col("week"), e2=(pl.col("plays") - pl.col("exp_plays")) ** 2, one=pl.lit(1.0))
        self._plays = _prefix(t, "key", ["e2", "one"])

    def plays_sd(self, key: int) -> float:
        k, e, c = self._plays
        i = _before(k, key)
        return float(np.sqrt(e[i - 1] / c[i - 1])) if i and c[i - 1] >= MIN_PLAYS_SD_GAMES else PLAYS_SD_DEFAULT

    # ---- league ratios
    def _build_ratios(self, player_log: pl.DataFrame, tgs: pl.DataFrame):
        tot = player_log.group_by("game_id", "team").agg(
            tg=pl.col("targets").sum().cast(pl.Float64), ca=pl.col("carries").sum().cast(pl.Float64),
            qc=pl.col("carries").filter(pl.col("family") == "QB").sum().cast(pl.Float64),
            qx=pl.col("rush_att_ex_kneel").filter(pl.col("family") == "QB").sum().cast(pl.Float64))
        t = (tgs.filter(pl.col("plays").is_not_null()).select("game_id", "team", "season", "week", "plays", "dropbacks", "sacks", "scrambles")
             .join(tot, on=["game_id", "team"], how="inner")
             .with_columns(key=pl.col("season") * 100 + pl.col("week"),
                           p_act=pl.col("dropbacks") - pl.col("sacks") - pl.col("scrambles"),
                           r_act=pl.col("plays") - pl.col("dropbacks") + pl.col("scrambles")))
        self._ratio = _prefix(t, "key", ["tg", "p_act", "ca", "r_act", "qc", "qx"])

    def ratios(self, key: int) -> tuple:
        k, tg, p, ca, r, qc, qx = self._ratio
        i = _before(k, key)
        if not i:
            return RATIO_DEFAULT
        return float(tg[i - 1] / p[i - 1]), float(ca[i - 1] / r[i - 1]), float(qx[i - 1] / qc[i - 1]) if qc[i - 1] > 0 else RATIO_DEFAULT[2]

    # ---- final-margin state curve
    def state_model(self, key: int):
        if key not in self._state_cache:
            rows = self._state_rows.filter((pl.col("season") * 100 + pl.col("week") < key) & pl.col("margin").is_not_null() & pl.col("plays").is_not_null())
            self._state_cache[key] = gs.fit_state_shares(rows, "margin") if rows.height >= gs.MIN_STATE_FIT_ROWS else None
        return self._state_cache[key]

    # ---- residual pools
    def _pool(self, quantity: str, family: str, key: int, shock: float):
        g = self._pools.get((quantity, family))
        if g is None:
            return None
        g = g.filter(pl.col("key") < key)
        return g

    def pools(self, key: int, shock: dict) -> dict:
        out = {}
        for q in QUANTITIES:
            allq = self._res.filter((pl.col("quantity") == q) & (pl.col("key") < key))
            for fam in ("QB", "RB", "WR", "TE", "other"):
                g = self._pool(q, fam, key, shock[q])
                g = g if g is not None and g.height >= MIN_POOL else allq
                if g.height < MIN_POOL:
                    out[(q, fam)] = None
                    continue
                z = g["res"].to_numpy() * np.sqrt(g["den"].to_numpy())
                ez2 = float(np.mean(z ** 2))
                team = shock[q] ** 2 * float(np.mean(g["pred"].to_numpy() ** 2 * g["den"].to_numpy()))
                v = max(ez2 - team, IDIO_FLOOR * ez2)
                out[(q, fam)] = np.sort(z * np.sqrt(v / ez2))
        return out

    def week(self, season: int, week: int) -> WeekCalib:
        key = season * 100 + week
        shock = {q: self.shock(q, key) for q in QUANTITIES}
        tr_, cr_, qk_ = self.ratios(key)
        return WeekCalib(key=key, kappa={s: self.kappa(s, key) for s in STATS}, shock=shock, pool=self.pools(key, shock),
                         plays_sd=self.plays_sd(key), target_ratio=tr_, carry_ratio=cr_, qb_keep=qk_, state_model=self.state_model(key))


# ====================================================================== inputs
@dataclass
class TeamInput:
    team: str
    is_home: bool
    exp_plays: float
    db_rate: np.ndarray                 # (5,) dropback rate per state, in gs.STATES order
    sack_rate: float
    scramble_rate: float
    player_ids: list
    roles: list
    groups: list
    shares: dict                        # stat -> (K + 1,) expected shares, the last entry is `rest` (unlisted players)
    p_exit: np.ndarray                  # (K,)
    eff: dict                           # quantity -> (K,) Phase 3 efficiency prediction (nan where the player has none)
    p_out: np.ndarray = None            # (K,) probability each simulated player is out (4a.1 / 4a.2)
    play_shares: dict = None            # stat -> (K + 1,) expected shares if every player plays (workload factors applied)
    scen_q: np.ndarray = None           # (M,) absence probability of each uncertain player (including linemen, who are not simulated)
    scen_k: np.ndarray = None           # (M,) index of that player among the K simulated players, -1 if he is not simulated
    scen_delta: dict = None             # stat -> (M, K + 1): shares when only that player is out, minus play_shares


@dataclass
class GameInput:
    game_id: str
    season: int
    week: int
    margin_mean: float
    margin_sd: float
    total_mean: float
    total_sd: float
    home: TeamInput
    away: TeamInput
    exit_pools: dict                    # group -> sorted array of share-completed values (4a.3)
    exit_all: np.ndarray
    calib: WeekCalib


@dataclass
class SimData:
    games: pl.DataFrame                 # game_expectations (2020-2024 rows)
    tgs: pl.DataFrame                   # team_game_state
    detail: pl.DataFrame                # per team-game player list with expected shares
    scen: pl.DataFrame                  # game_id, team, out_player, player_id, exp_*: shares when only out_player is out
    eff: pl.DataFrame                   # game_id, player_id, quantity, pred
    exit_pools: dict                    # (season, week) -> (group pools, pooled)
    calibration: Calibration
    eligible: pl.DataFrame              # player-game rows eligible for at least one market (game_id, player_id, elig)
    seasons: tuple
    _calib_cache: dict = field(default_factory=dict)

    def week_calib(self, season: int, week: int) -> WeekCalib:
        k = (season, week)
        if k not in self._calib_cache:
            self._calib_cache[k] = self.calibration.week(season, week)
        return self._calib_cache[k]

    def game_input(self, game_id: str) -> GameInput | None:
        g = self.games.filter(pl.col("game_id") == game_id)
        if not g.height:
            return None
        g = g.row(0, named=True)
        season, week = g["season"], g["week"]
        cal = self.week_calib(season, week)
        teams = {}
        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            t = self.tgs.filter((pl.col("game_id") == game_id) & (pl.col("team") == team))
            if not t.height or t["db_rate_tied"][0] is None or t["exp_plays"][0] is None:
                return None
            t = t.row(0, named=True)
            det = self.detail.filter((pl.col("game_id") == game_id) & (pl.col("team") == team)).sort("player_id")
            main = det.filter((pl.col("player_id") != "rest") & (pl.col("role") != "OL"))
            rest = det.filter(pl.col("player_id") == "rest")
            shares = {}
            for st in STATS:
                v = main[f"exp_{st}"].to_numpy().astype(float)
                shares[st] = np.append(v, float(rest[f"exp_{st}"][0]) if rest.height else max(0.0, 1.0 - v.sum()))
            eff = self.eff.filter(pl.col("game_id") == game_id)
            ids = main["player_id"].to_list()
            K = len(ids)
            order = main["player_id"].to_list() + ["rest"]
            bmap = {r["player_id"]: r for r in det.iter_rows(named=True)}
            play = {st: np.array([bmap[i][f"b_{st}"] for i in order], dtype=float) for st in STATS}
            sc = self.scen.filter((pl.col("game_id") == game_id) & (pl.col("team") == team))
            outs = sorted(sc["out_player"].unique().to_list())
            pout = {r["player_id"]: r["p_out"] for r in det.iter_rows(named=True)}
            delta = {st: np.zeros((len(outs), K + 1)) for st in STATS}
            for m, o in enumerate(outs):
                rows_o = {r["player_id"]: r for r in sc.filter(pl.col("out_player") == o).iter_rows(named=True)}
                for st in STATS:
                    delta[st][m] = np.array([rows_o[i][f"exp_{st}"] if i in rows_o else play[st][j] for j, i in enumerate(order)]) - play[st]
            pos = {i: j for j, i in enumerate(ids)}
            em = {q: np.full(len(ids), np.nan) for q in QUANTITIES}
            if eff.height:
                lookup = {(r["player_id"], r["quantity"]): r["pred"] for r in eff.iter_rows(named=True)}
                for q in QUANTITIES:
                    em[q] = np.array([lookup.get((p, q), np.nan) for p in ids], dtype=float)
            teams[side] = TeamInput(team=team, is_home=side == "home", exp_plays=float(t["exp_plays"]),
                                    db_rate=np.array([t[f"db_rate_{s}"] for s in gs.STATES], dtype=float), sack_rate=float(t["sack_rate"]),
                                    scramble_rate=float(t["scramble_rate"]), player_ids=ids, roles=main["role"].to_list(),
                                    groups=main["group"].to_list(), shares=shares,
                                    p_exit=np.nan_to_num(main["p_exit"].to_numpy().astype(float)), eff=em,
                                    p_out=np.array([pout[i] for i in ids], dtype=float), play_shares=play,
                                    scen_q=np.array([pout[o] for o in outs], dtype=float), scen_k=np.array([pos.get(o, -1) for o in outs], dtype=int),
                                    scen_delta=delta)
        pools, pooled = self.exit_pools[(season, week)]
        return GameInput(game_id=game_id, season=season, week=week, margin_mean=float(g["margin_mean"]), margin_sd=float(g["margin_sd"]),
                         total_mean=float(g["total_mean"]), total_sd=float(g["total_sd"]), home=teams["home"], away=teams["away"],
                         exit_pools=pools, exit_all=pooled, calib=cal)


# ====================================================================== builders
def _detail_and_exit(data, raw_db, cap):
    """Per team-game player lists with 4a.2 expected shares, plus the 4a.3 exit pools, fit week by week."""
    from models import injuries as ij

    pw = ij.build_player_weeks(raw_db, cap)
    frame = ij.build_role_frame(raw_db, cap)
    status_fn = ij.make_status_lookup(ij.load_injury_rows(raw_db, cap))
    ros = ij.load_roster_status(raw_db, cap)
    blocked = {k[:2]: set(g["gsis_id"].to_list()) for k, g in
               ros.filter(pl.col("roster_status").is_in(list(ij.BLOCKED_ROSTER_STATUSES))).partition_by("season", "week", as_dict=True).items()}
    groups = dict(pw.sort("gameday").group_by("gsis_id", maintain_order=True).agg(pl.col("group").last()).iter_rows())
    by_team = {k[0]: g for k, g in frame.partition_by("team", as_dict=True).items()}
    rows, scen_rows, exit_pools = [], [], {}
    for season, week, cutoff in data.weeks:
        if season > cap:
            continue
        model, shifts, exit_model = ij.fit_status_model(pw, cutoff), ij.fit_shifts(frame, cutoff), ij.fit_exit_model(pw, cutoff)
        prior, rhist = ij.role_prior_shares(frame, cutoff), ij.role_history(frame, cutoff)
        exit_pools[(season, week)] = ({g: np.asarray(v, dtype=float) for g, v in exit_model.shares.items()}, np.asarray(exit_model.all_shares, dtype=float))
        for g in data.team_log.filter((pl.col("season") == season) & (pl.col("week") == week)).iter_rows(named=True):
            out = ij.game_expected_shares(by_team[g["team"]], shifts, model, None, None, g["game_id"], g["team"], season, week, g["gameday"],
                                          ij.main_run_as_of(g["gameday"]), cutoff, exit_model=exit_model, status_fn=status_fn,
                                          blocked_ids=blocked.get((season, week), set()), player_groups=groups, with_eff=False)
            play, scenarios, _ = ij.game_share_scenarios(by_team[g["team"]], shifts, model, g["game_id"], g["team"], season, week,
                                                         ij.main_run_as_of(g["gameday"]), cutoff, status_fn=status_fn,
                                                         blocked_ids=blocked.get((season, week), set()), player_groups=groups, role_prior=prior, role_hist=rhist)
            pmap = {r["player_id"]: r for r in play.iter_rows(named=True)}
            gmap = {"rest": "rest"}
            for r in out.iter_rows(named=True):
                pb = pmap[r["player_id"]]
                rows.append(dict(season=season, week=week, key=season * 100 + week, game_id=g["game_id"], team=g["team"],
                                 player_id=r["player_id"], role=r["role"], group=groups.get(r["player_id"]) or gmap.get(r["player_id"]),
                                 exp_carry=r["exp_carry"], exp_target=r["exp_target"], exp_dropback=r["exp_dropback"], p_exit=r.get("p_exit"),
                                 p_out=r["p_out"], b_carry=pb["exp_carry"], b_target=pb["exp_target"], b_dropback=pb["exp_dropback"]))
            for r in scenarios.iter_rows(named=True):
                scen_rows.append(dict(game_id=g["game_id"], team=g["team"], **r))
    detail = pl.DataFrame(rows, infer_schema_length=None).sort("season", "week", "game_id", "team", "player_id")
    scen = pl.DataFrame(scen_rows, infer_schema_length=None).sort("game_id", "team", "out_player", "player_id")
    return detail, scen, exit_pools


def _efficiency_predictions(data, raw_db, cap):
    """The Phase 3 efficiency model's prediction for EVERY eligible player-game (not only those whose denominator turned out
    positive), run week by week through the same cutoffs as the harness."""
    from eval import backtest as bt
    from models import efficiency

    et = efficiency.load_efficiency_tables(data, raw_db=raw_db, max_season=cap)
    player, _ = efficiency.efficiency_predictors(et)
    rename = {"comp_pct": "comp_rate", "catch_pct": "catch_rate"}
    rows = []
    for season, week, cutoff in data.weeks:
        if season > cap:
            continue
        history = bt.History(data.player_log.filter(pl.col("gameday") < cutoff), data.team_log.filter(pl.col("gameday") < cutoff))
        targets, _, _, _, _ = bt._week_targets(data, season, week)
        if not targets:
            continue
        gid = dict(zip(zip(data.player_log["player_id"].to_list(), data.player_log["gameday"].to_list()), data.player_log["game_id"].to_list()))
        for t, p in zip(targets, player(history, targets, cutoff)):
            for q, v in p.items():
                if v is not None:
                    rows.append(dict(game_id=gid[(t.player_id, t.gameday)], player_id=t.player_id, quantity=rename.get(q, q), pred=float(v)))
    return pl.DataFrame(rows, schema={"game_id": pl.String, "player_id": pl.String, "quantity": pl.String, "pred": pl.Float64}).sort(
        "game_id", "player_id", "quantity")


def _residual_table(player_log: pl.DataFrame, wf_path) -> pl.DataFrame:
    """Walk-forward residuals of the Phase 3 efficiency model (data/processed/walkforward_predictions.parquet, built by
    build_phase4_tables.py), with the denominators of the games they came from."""
    if not wf_path.exists():
        raise FileNotFoundError(f"{wf_path} is missing: run build_phase4_tables.py first")
    w = pl.read_parquet(wf_path).filter(pl.col("quantity").is_in(list(QUANTITIES)) & pl.col("expected").is_not_null() & (pl.col("season") <= config.BACKTEST_SEASONS[-1]))
    dens = player_log.select("game_id", "player_id", *sorted(set(DEN_COLUMN.values())))
    w = w.join(dens, on=["game_id", "player_id"], how="inner")
    den = pl.coalesce(*[pl.when(pl.col("quantity") == q).then(pl.col(c).cast(pl.Float64)) for q, c in DEN_COLUMN.items()])
    w = w.with_columns(den=den, family=pl.col("role").replace_strict(FAMILY_OF_ROLE, default="other"), key=pl.col("season") * 100 + pl.col("week"))
    return (w.filter(pl.col("den") > 0).select("key", "game_id", "team", "quantity", "family", pred=pl.col("expected"), den="den", res="residual")
            .sort("key", "game_id", "team", "quantity"))


def build_sim_data(raw_db=config.RAW_DUCKDB_PATH, seasons=config.BACKTEST_SEASONS, wf_path=None) -> SimData:
    """Everything the simulation needs for `seasons` (2020-2024 only). Walk-forward throughout."""
    from eval import backtest as bt

    if not set(seasons) <= set(config.BACKTEST_SEASONS):
        raise config.HoldoutError(f"seasons {tuple(seasons)} are outside the backtest seasons {config.BACKTEST_SEASONS}")
    cap = max(seasons)
    wf_path = wf_path or config.PROCESSED_DIR / "walkforward_predictions.parquet"
    data = bt.load_backtest_data(raw_db)
    tables = gs.build_all(raw_db, cap, n_boot=0)
    ge = tables["game_expectations"].filter(pl.col("season").is_in(list(seasons)))
    tgs = tables["team_game_state"]
    detail, scen, exit_pools = _detail_and_exit(data, raw_db, cap)
    eff = _efficiency_predictions(data, raw_db, cap)
    res = _residual_table(data.player_log, wf_path)
    games = gs.load_games(raw_db, cap)
    pts = pl.concat([games.select("game_id", team="home_team", pf="home_score", pa="away_score"),
                     games.select("game_id", team="away_team", pf="away_score", pa="home_score")])
    state_rows = (gs.load_state_rows(raw_db, cap).join(pts, on=["game_id", "team"], how="left")
                  .with_columns(margin=(pl.col("pf") - pl.col("pa")).cast(pl.Float64)))    # every team-game from 2019 on, realised margin
    calib = Calibration(res, detail, data.player_log, tgs, state_rows)
    elig = data.player_log.filter(pl.col("elig") != "").select("game_id", "player_id", "team", "elig")
    return SimData(games=ge, tgs=tgs, detail=detail, scen=scen, eff=eff, exit_pools=exit_pools, calibration=calib, eligible=elig, seasons=tuple(seasons))
