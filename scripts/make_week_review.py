"""Game-by-game review of a reconstruction week: the model's number, the FanDuel line, the real number, the misses, and a plain-rule account of the miss. Display only.

For each (player, market) with a line: miss = actual - number (positive = the real number was higher). The miss in yards is split in two, with the model's own predicted components:
  volume     = (actual volume - predicted volume) x the model's predicted yards (or catches) per unit of volume
  efficiency = the rest (what each unit of volume produced, against what the model expected)
Volume is attempts (passing), carries (rushing), targets (receiving). Drivers are the largest contributions to the model's volume number from the stored attribution.
  python scripts/make_week_review.py 1 out.json
"""
import json
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from app import data as D
from live import history as H

GRADED = config.ROOT / "data" / "processed" / "reconstruct_2026_graded.parquet"
LOG = config.ROOT / "records" / "lines" / "prop_lines_log.csv"
EXPLAIN = config.ROOT / "records" / "retro_2026" / "predictions" / "explain"
VOL = {"pass_yds": "pass_att", "rush_yds": "rush_att", "qb_rush_yds": "qb_rush_att", "rec_yds": "targets", "rec": "targets"}
SHOW = ["pass_yds", "qb_rush_yds", "rush_yds", "rec", "rec_yds"]
LABEL = {"pass_yds": "Pass yds", "qb_rush_yds": "QB rush yds", "rush_yds": "Rush yds", "rec": "Receptions", "rec_yds": "Rec yds"}
UNIT = {"pass_att": "pass attempts", "rush_att": "carries", "qb_rush_att": "carries", "targets": "targets"}
VOLQ = {"pass_att": "pass_att", "rush_att": "rush_att", "qb_rush_att": "qb_rush_att", "targets": "targets"}


def lines_for(week: int) -> pl.DataFrame:
    d = pl.read_csv(LOG, infer_schema_length=0).filter((pl.col("source") == "bettingpros_import") & pl.col("market").is_in(SHOW) & pl.col("game_id").str.starts_with(f"2026_{week:02d}_"))
    d = d.with_columns(line=pl.col("line").cast(pl.Float64), price=pl.col("price").cast(pl.Float64, strict=False), is_open=pl.col("note").str.starts_with("open")).sort("captured_at")
    key = ["game_id", "player_id", "market"]
    over = d.filter(pl.col("side") == "over")
    last = over.group_by(key, maintain_order=True).last().select(*key, line="line", over_price="price")
    opn = over.filter(pl.col("is_open")).group_by(key, maintain_order=True).first().select(*key, open_line="line")
    under = d.filter(pl.col("side") == "under").group_by(key, maintain_order=True).last().select(*key, under_price="price")
    return last.join(opn, on=key, how="left").join(under, on=key, how="left")


def drivers(expl: pl.DataFrame, game_id: str, player_id: str, quantity: str, stack: str, n: int = 3) -> list:
    x = expl.filter((pl.col("game_id") == game_id) & (pl.col("player_id") == player_id) & (pl.col("quantity") == quantity) & (pl.col("stack") == stack))
    if x.is_empty():
        return []
    base = x.filter(pl.col("feature") == "(baseline)")
    top = x.filter(pl.col("feature") != "(baseline)").with_columns(a=pl.col("contribution").abs()).sort("a", descending=True).head(n)
    out = [dict(feature="(typical player at this slot)", value=None, contribution=float(base["contribution"][0]))] if base.height else []
    return out + [dict(feature=r["feature"], value=r["value"], contribution=r["contribution"]) for r in top.iter_rows(named=True)]


def main(week: int, out: Path):
    g = pl.read_parquet(GRADED).filter(pl.col("week") == week)
    player_rows = g.filter(~pl.col("market").is_in(["spread", "total", "moneyline"]))
    expl = pl.read_parquet(EXPLAIN / f"retro_w{2026}_{week:02d}.parquet")
    L = lines_for(week)
    key = ["game_id", "player_id"]
    vol = player_rows.filter(pl.col("market").is_in(list(set(VOL.values())))).select(*key, vmarket="market", vpred="mean", vact="actual")
    rows = player_rows.filter(pl.col("market").is_in(SHOW)).join(L, on=["game_id", "player_id", "market"], how="inner")
    rows = rows.with_columns(vmarket=pl.col("market").replace(VOL)).join(vol, on=[*key, "vmarket"], how="left")
    played, _ = H.read_played_logs(2026)
    names = D.player_names(rows["player_id"].unique().to_list())
    team_of = {(r["game_id"], r["player_id"]): r["team"] for r in played.filter(pl.col("week") == week).iter_rows(named=True)}
    tpass = played.filter(pl.col("week") == week).group_by("game_id", "team").agg(att=pl.col("attempts").sum())
    # schedule facts
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    sched = con.execute("SELECT game_id, home_team, away_team, home_score, away_score, spread_line, total_line, home_moneyline, away_moneyline, roof, temp, wind, gameday FROM schedules "
                        f"WHERE season = 2026 AND week = {week} QUALIFY row_number() OVER (PARTITION BY game_id ORDER BY pulled_at DESC) = 1").pl()
    con.close()
    gm = g.filter(pl.col("market").is_in(["total", "moneyline"])).select("game_id", "market", "mean").pivot(on="market", index="game_id", values="mean")
    out_games = []
    for s in sched.sort("gameday", "game_id").iter_rows(named=True):
        gid = s["game_id"]
        r = rows.filter(pl.col("game_id") == gid)
        if r.is_empty():
            continue
        m = gm.filter(pl.col("game_id") == gid)
        pa = {t: v for t, v in tpass.filter(pl.col("game_id") == gid).select("team", "att").iter_rows()}
        qbpred = {}
        for q in player_rows.filter((pl.col("game_id") == gid) & (pl.col("market") == "pass_att")).iter_rows(named=True):
            qbpred[team_of.get((gid, q["player_id"]))] = (q["mean"], q["actual"], q["did_not_play"])
        items = []
        for x in r.iter_rows(named=True):
            if x["did_not_play"]:
                why = "Did not play: the line is void."
                items.append(dict(player=names.get(x["player_id"]), team=team_of.get((gid, x["player_id"])), market=LABEL[x["market"]], mk=x["market"], model=x["mean"], line=x["line"], open_line=x["open_line"],
                                  actual=None, dnp=True, why=why, drivers=[], vol=None, eff=None, vol_pred=x["vpred"], vol_act=None, status=x["injury_status"]))
                continue
            a, mod, ln = x["actual"], x["mean"], x["line"]
            v = e = None
            vp, va = x["vpred"], x["vact"]
            ratio = mod / vp if vp and vp > 0 else None
            if ratio is not None and va is not None:
                v = (va - vp) * ratio
                e = a - va * ratio
            why = ""
            if v is not None:
                unit = UNIT[x["vmarket"]]
                what = "yards" if x["market"] != "rec" else "catches"
                big = "volume" if abs(v) >= abs(e) else "efficiency"
                why = (f"{big.capitalize()} was the bigger part. Volume: {va:.0f} {unit} vs {vp:.1f} expected ({v:+.1f} {what}). "
                       f"Efficiency: {e:+.1f} {what} from what each {unit.rstrip('s') if unit != 'carries' else 'carry'} produced.")
            items.append(dict(player=names.get(x["player_id"]), team=team_of.get((gid, x["player_id"])), market=LABEL[x["market"]], mk=x["market"], model=mod, line=ln, open_line=x["open_line"], actual=a, dnp=False,
                              why=why, drivers=drivers(expl, gid, x["player_id"], VOLQ[x["vmarket"]], x["stack"]) if x["vmarket"] else [], vol=v, eff=e, vol_pred=vp, vol_act=va,
                              status=x["injury_status"], q10=x["q10"], q90=x["q90"], over_price=x["over_price"], under_price=x["under_price"]))
        hs, as_ = s["home_score"], s["away_score"]
        out_games.append(dict(game_id=gid, home=s["home_team"], away=s["away_team"], home_score=hs, away_score=as_, total_actual=(hs + as_) if hs is not None else None,
                              margin_actual=(hs - as_) if hs is not None else None, total_model=float(m["total"][0]) if m.height else None, p_home_model=float(m["moneyline"][0]) if m.height else None,
                              total_line=s["total_line"], spread_line=s["spread_line"], home_ml=s["home_moneyline"], away_ml=s["away_moneyline"], roof=s["roof"], temp=s["temp"], wind=s["wind"],
                              qb_pass_att={t: dict(pred=v[0], actual=pa.get(t)) for t, v in qbpred.items() if t}, items=items))
    Path(out).write_text(json.dumps(dict(week=week, games=out_games), default=str))
    n = sum(len(x["items"]) for x in out_games)
    print(f"week {week}: {len(out_games)} games, {n} player-market rows")


if __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]))
