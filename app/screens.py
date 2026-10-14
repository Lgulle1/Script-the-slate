"""The six screens. Each reads stored records through app.data and renders; none recomputes a model. The Lines log writes only prop_lines_log.csv (live/prop_lines.py)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

import config
from app import data as D
from live import prop_lines as PL

PLAYER_MARKETS = ("pass_att", "pass_cmp", "pass_yds", "qb_rush_att", "qb_rush_yds", "rush_att", "rush_yds", "targets", "rec", "rec_yds")
MARKET_COMPONENTS = {"pass_att": ("pass_att",), "pass_cmp": ("pass_att", "comp_pct"), "pass_yds": ("pass_att", "comp_pct", "yds_per_cmp"), "rush_att": ("rush_att",),
                     "rush_yds": ("rush_att", "ypc"), "qb_rush_att": ("qb_rush_att",), "qb_rush_yds": ("qb_rush_att", "qb_ypc"), "targets": ("targets",),
                     "rec": ("targets", "catch_pct"), "rec_yds": ("targets", "catch_pct", "yds_per_rec")}


def _preds():
    p = D.predictions()
    return p, D.latest(p)


def _none(msg: str) -> None:
    st.info(msg)


# ----------------------------------------------------------------------------------------------------------- Slate
def slate() -> None:
    st.header("Slate")
    allp, latest = _preds()
    if latest.is_empty():
        return _none("No predictions are stored yet.")
    weeks = sorted({(D.game_parts(g)["season"], D.game_parts(g)["week"]) for g in latest["game_id"].unique().to_list()}, reverse=True)
    season, week = st.selectbox("Week", weeks, format_func=lambda x: f"{x[0]} week {x[1]}", key="slate_week")
    comp, _ = D.explain_tables()
    rows = []
    for g in D.week_games(latest, season, week):
        gp = latest.filter((pl.col("game_id") == g) & pl.col("player_id").is_null())
        one = lambda m: (gp.filter(pl.col("market") == m).row(0, named=True) if gp.filter(pl.col("market") == m).height else None)
        sp, tot, ml = one("spread"), one("total"), one("moneyline")
        parts = D.game_parts(g)
        plays = {}
        if comp.height:
            c = comp.filter((pl.col("game_id") == g) & (pl.col("quantity") == "plays") & (pl.col("feature") == "(baseline)") & pl.col("team").is_not_null())
            plays = dict(zip(c["team"], c["component_prediction"]))
        rng = lambda r: f"{r['q10']:.1f} to {r['q90']:.1f}" if r and r["q10"] is not None else ""
        rows.append({"game_id": g, "kickoff (UTC)": str((sp or tot or ml or {}).get("kickoff_time", ""))[:16], "away": parts["away"], "home": parts["home"],
                     "expected margin (home)": None if not sp else round(sp["mean"], 1), "margin range 10-90%": rng(sp),
                     "expected total": None if not tot else round(tot["mean"], 1), "total range 10-90%": rng(tot),
                     "home win prob": None if not ml else round(ml["mean"], 2), "plays away": round(plays[parts["away"]], 1) if parts["away"] in plays else None,
                     "plays home": round(plays[parts["home"]], 1) if parts["home"] in plays else None, "stored predictions": latest.filter(pl.col("game_id") == g).height})
    st.dataframe(pl.DataFrame(rows).sort("kickoff (UTC)").to_pandas(), hide_index=True, width="stretch")
    st.caption("Stored values only. Margin is home minus away; the range is the stored 10th to 90th percentile.")
    pick = st.selectbox("Open a game in Matchup", [r["game_id"] for r in rows], key="slate_open")
    if st.button("Open in Matchup", key="slate_open_btn"):
        st.session_state["matchup_game"] = pick
        st.session_state["screen"] = "Matchup"
        st.rerun()


# ----------------------------------------------------------------------------------------------------- Player report
def _drivers(game_id: str, player_id: str, market: str, stack: str, n: int = 6) -> pl.DataFrame:
    comp, _ = D.explain_tables()
    if comp.is_empty():
        return pl.DataFrame()
    c = comp.filter((pl.col("game_id") == game_id) & (pl.col("player_id") == player_id) & pl.col("quantity").is_in(list(MARKET_COMPONENTS.get(market, ()))))
    if c.is_empty():
        return c
    stk = c.filter(pl.col("stack") == stack)
    c = stk if stk.height else c
    return (c.filter(pl.col("feature") != "(baseline)").with_columns(abs_c=pl.col("contribution").abs()).sort("abs_c", descending=True).group_by("quantity", maintain_order=True).head(n)
            .select("quantity", "feature", "value", "contribution", "component_prediction"))


def player_report() -> None:
    st.header("Player report")
    allp, latest = _preds()
    pl_rows = latest.filter(pl.col("player_id").is_not_null()) if latest.height else latest
    if pl_rows.is_empty():
        return _none("No player predictions are stored yet.")
    game = st.selectbox("Game", sorted(pl_rows["game_id"].unique().to_list()), key="pr_game")
    g = pl_rows.filter(pl.col("game_id") == game)
    ids = sorted(g["player_id"].unique().to_list())
    names = D.player_names(ids)
    player = st.selectbox("Player", ids, format_func=lambda i: names.get(i, i), key="pr_player")
    mk = sorted(g.filter(pl.col("player_id") == player)["market"].unique().to_list(), key=PLAYER_MARKETS.index)
    market = st.selectbox("Market", mk, key="pr_market")
    r = g.filter((pl.col("player_id") == player) & (pl.col("market") == market)).row(0, named=True)
    st.subheader(f"{names.get(player, player)}: {market}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model mean", f"{r['mean']:.1f}")
    c2.metric("Median (stored)", "-" if r["median"] is None else f"{r['median']:.1f}")
    c3.metric("Stack", r["stack"])
    st.caption(f"Version {r['version']}, stored {r['stored_at']:%Y-%m-%d %H:%M} UTC, kickoff {r['kickoff_time']:%Y-%m-%d %H:%M} UTC, model {r['model_version']}")
    if r["ladder"]:
        lad = json.loads(r["ladder"])
        st.markdown("**Threshold ladder** (probability the real number reaches each mark)")
        st.dataframe(pl.DataFrame({"at least": [float(k) for k in lad], "probability": [round(v, 2) for v in lad.values()]}).sort("at least").to_pandas(), hide_index=True)
    else:
        st.caption("No ladder is stored for this market (not enough history for a distribution).")
    if r["q05"] is not None:
        q = pl.DataFrame({"band": ["5-95%", "25-75%"], "lo": [r["q05"], r["q25"]], "hi": [r["q95"], r["q75"]]}).to_pandas()
        base = alt.Chart(q).mark_bar(height=18).encode(x=alt.X("lo:Q", title=market), x2="hi:Q", y=alt.Y("band:N", title=None), color=alt.Color("band:N", legend=None))
        marks = alt.Chart({"values": [{"x": r["q50"], "k": "median"}, {"x": r["mean"], "k": "mean"}]}).mark_tick(size=34, thickness=3).encode(x="x:Q", color=alt.Color("k:N", title=None))
        st.altair_chart(base + marks, width="stretch")
    else:
        st.caption("No distribution is stored for this market.")
    st.markdown("**Confidence components**")
    st.caption("Not stored: the components (sample size, similarity strength, input completeness, freshness, continuity, submodel agreement) belong to the comparables engine, which is parked. "
               "Nothing is shown rather than a made-up score.")
    d = _drivers(game, player, market, r["stack"])
    st.markdown("**What drove the number** (stage-1 attribution: baseline plus each input's push)")
    if d.height:
        st.dataframe(d.to_pandas(), hide_index=True)
    else:
        st.caption("No attribution is stored for this prediction.")


# ----------------------------------------------------------------------------------------------------------- Matchup
def matchup() -> None:
    st.header("Matchup")
    allp, latest = _preds()
    if latest.is_empty():
        return _none("No predictions are stored yet.")
    games = sorted(latest["game_id"].unique().to_list())
    start = st.session_state.get("matchup_game", games[0])
    game = st.selectbox("Game", games, index=games.index(start) if start in games else 0, key="mu_game")
    parts = D.game_parts(game)
    st.subheader(f"{parts['away']} at {parts['home']} ({parts['season']} week {parts['week']})")
    g = latest.filter(pl.col("game_id") == game)
    gm = g.filter(pl.col("player_id").is_null()).select("market", "stack", "mean", "q10", "q90", "version")
    st.dataframe(gm.to_pandas(), hide_index=True)
    pr = g.filter(pl.col("player_id").is_not_null())
    if pr.height:
        names = D.player_names(pr["player_id"].unique().to_list())
        t = pr.with_columns(player=pl.col("player_id").replace_strict(names, default=pl.col("player_id"))).select("player", "market", "stack", "mean", "q10", "q90")
        st.dataframe(t.sort("market", "mean", descending=[False, True]).to_pandas(), hide_index=True, height=320)
    comp, marg = D.explain_tables()
    if marg.height:
        m = marg.filter(pl.col("game_id") == game)
        if m.height:
            st.markdown("**Expected margin by term** (4b.2: league average, the constant and each input, home minus away)")
            st.dataframe(m.filter(pl.col("side") == "margin").select("term", "points").to_pandas(), hide_index=True)
    st.markdown("**Comparable searches S1 to S5**")
    for s in ("S1 own past games vs this defense", "S2 this matchup's offense and defense", "S3 similar players, same opponent type", "S4 similar team situations", "S5 two-sided blend"):
        st.write(f"{s}: **No reliable comparable.** Weight 0. The comparables engine is parked for this version (it did not clear the gate).")


# --------------------------------------------------------------------------------------------------------- Lines log
def _hash_predictions() -> str:
    import hashlib
    h = hashlib.sha256()
    for f in sorted((D.records_dir() / "predictions").rglob("*")):
        if f.is_file():
            h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def lines_log() -> None:
    st.header("Lines log")
    st.caption("Type a line you saw. It is saved in its own table and compared with the stored prediction for display only; it never changes a prediction.")
    allp, latest = _preds()
    pl_rows = latest.filter(pl.col("player_id").is_not_null()) if latest.height else latest
    if pl_rows.is_empty():
        return _none("No player predictions are stored yet.")
    game = st.selectbox("Game", sorted(pl_rows["game_id"].unique().to_list()), key="ll_game")
    g = pl_rows.filter(pl.col("game_id") == game)
    ids = sorted(g["player_id"].unique().to_list())
    names = D.player_names(ids)
    player = st.selectbox("Player", ids, format_func=lambda i: names.get(i, i), key="ll_player")
    market = st.selectbox("Market", sorted(g.filter(pl.col("player_id") == player)["market"].unique().to_list()), key="ll_market")
    c = st.columns(4)
    book = c[0].text_input("Book", "book", key="ll_book")
    side = c[1].selectbox("Side", ["over", "under", "at_least"], key="ll_side")
    line = c[2].number_input("Line", value=0.0, step=0.5, key="ll_line")
    price = c[3].number_input("Odds (American)", value=-110, step=5, key="ll_price")
    rec = g.filter((pl.col("player_id") == player) & (pl.col("market") == market)).row(0, named=True)
    be = D.breakeven(price)
    pa = D.prob_at_least(rec, line)
    if pa is None:
        st.caption("No stored distribution for this prediction: the model probability is not available.")
    else:
        p_at, exact = pa
        p = p_at if side in ("over", "at_least") else 1 - p_at
        gap = (p - be) * 100
        st.write(f"Break-even probability at these odds: **{be:.1%}**. Model probability: **{p:.1%}**{'' if exact else ' (line outside the stored range: bounded)'}. Gap: **{gap:+.1f} points**.")
        if abs(gap) >= config.LINES_FLAG_GAP_PP:
            st.warning(f"The gap is at least {config.LINES_FLAG_GAP_PP:.0f} points. It is a flag, not a recommendation: the model has not been shown to beat any line.")
    if st.button("Save line", key="ll_save"):
        parts = D.game_parts(game)
        kick = rec["kickoff_time"]
        sched = pl.DataFrame(dict(game_id=[game], home_team=[parts["home"]], away_team=[parts["away"]], gameday=[kick.date()], kickoff_time=[kick]), schema_overrides={"kickoff_time": pl.Datetime("us", "UTC")})
        pl_frame = pl.DataFrame(dict(gsis_id=[player], display_name=[names.get(player, player)], latest_team=[parts["home"]], position=[""]))
        name_of = {"pass_yds": "Passing Yards", "pass_att": "Pass Attempts", "pass_cmp": "Pass Completions", "rush_yds": "Rushing Yards", "rush_att": "Rush Attempts", "rec": "Receptions",
                   "rec_yds": "Receiving Yards", "targets": "Targets", "qb_rush_yds": "Rushing Yards", "qb_rush_att": "Rush Attempts"}
        row = pl.DataFrame(dict(book=[book], game_id=[game], player_id=[player], player_name=[names.get(player, player)], book_market=[name_of[market]], side=[side], line=[line], price=[price]))
        res = PL.log_prop_lines(row, datetime.now(timezone.utc), D.records_dir() / "lines" / "prop_lines_log.csv", sched, pl_frame, source="app")
        st.success(f"Saved: written {res.written}, unchanged {res.unchanged}, refused {res.refused} (a line seen at or after kickoff is refused).")
    log = D.prop_log()
    st.markdown("**Logged lines**")
    if log.height:
        st.dataframe(log.tail(50).to_pandas(), hide_index=True)
    else:
        st.caption("Nothing logged yet.")


# ------------------------------------------------------------------------------------------------------------ Grading
def grading() -> None:
    st.header("Grading")
    tab_pure, tab_market = st.tabs(["Pure track", "Market-aware track"])
    with tab_market:
        st.caption("Empty by design: nothing that used or compared a line is graded into the pure track.")
    with tab_pure:
        gr = D.graded()
        if gr.is_empty():
            return _none("Nothing has been graded yet. Predictions are graded once their games are final (grade_week.py).")
        gr = gr.with_columns(season=pl.col("game_id").str.slice(0, 4).cast(pl.Int32), week=pl.col("game_id").str.slice(5, 2).cast(pl.Int32))
        seasons = sorted(gr["season"].unique().to_list())
        season = st.selectbox("Season", seasons, key="gr_season")
        weeks = sorted(gr.filter(pl.col("season") == season)["week"].unique().to_list())
        wk = st.multiselect("Weeks", weeks, default=weeks, key="gr_weeks")
        f = gr.filter((pl.col("season") == season) & pl.col("week").is_in(wk))
        from live import grade as GR
        summ = GR.summary(f)
        st.dataframe(summ.to_pandas(), hide_index=True)
        base = D.graded_summary()
        if base.height and "baseline_mae" in base.columns:
            st.markdown("**Best simple baseline on the same rows**")
            st.dataframe(base.select("stack", "market", "n_compared", "model_mae", "baseline_mae").drop_nulls("n_compared").to_pandas(), hide_index=True)
        played = f.filter(~pl.col("did_not_play"))
        by_week = played.group_by("week", "market").agg(mae=pl.col("abs_error").mean()).sort("week")
        if by_week.height:
            mk = st.selectbox("Market for the weekly line", sorted(by_week["market"].unique().to_list()), key="gr_market")
            st.line_chart(by_week.filter(pl.col("market") == mk).select("week", "mae").to_pandas().set_index("week"))
        dnp = f.filter(pl.col("did_not_play"))
        st.caption(f"{dnp.height} did-not-play rows are graded as 0 and kept out of the played averages above.")
        preds = D.predictions().select("prediction_id", "ladder")
        j = f.filter(pl.col("ladder_hits").is_not_null()).join(preds, on="prediction_id", how="inner")
        if j.height:
            pts = []
            for r in j.iter_rows(named=True):
                lad, hits = json.loads(r["ladder"]), json.loads(r["ladder_hits"])
                pts += [(lad[t], hits[t]) for t in lad]
            a = np.array(pts)
            b = np.clip(np.digitize(a[:, 0], np.linspace(0, 1, 11)) - 1, 0, 9)
            rel = pl.DataFrame([dict(predicted=float(a[b == i, 0].mean()), observed=float(a[b == i, 1].mean())) for i in range(10) if (b == i).any()])
            st.markdown("**Reliability** (stored ladder probability vs how often the mark was reached)")
            st.line_chart(rel.to_pandas().set_index("predicted"))


# -------------------------------------------------------------------------------------------------------- Model health
def model_health() -> None:
    st.header("Model health")
    t = D.factor_table()
    on = t.filter(pl.col("status") == "on")
    red = on.filter(pl.col("red_flag"))
    if red.height:
        st.error(f"{red.height} signal group(s) are on but showed no gain in the most recent season: " + ", ".join(f"{a} / {b}" for a, b in zip(red["layer"], red["market"])))
    else:
        st.success("No active signal group is flagged: each one that is on gained in the most recent season.")
    st.markdown("**Active signal groups** (weight 1, the CLEARS rule)")
    st.dataframe(on.drop("source").to_pandas(), hide_index=True)
    st.markdown("**Every layer and market** (weight 0 means it did not clear and is flagged, never dropped)")
    st.dataframe(t.drop("source").to_pandas(), hide_index=True, height=360)
    st.caption("Live graded weeks do not exist yet; this table is the 2020-2024 backtest. Recommendation hit rate needs lines typed into the Lines log.")
