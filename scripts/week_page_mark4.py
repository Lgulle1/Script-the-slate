"""Makes Mark 4 a full Mark on the week's picks page (records/cards/<season>_wkNN_picks.html, built by make_week_page.py): its own switch button,
selected by default (Mark 4 is the bet from 2026 week 7), its card (records/cards/<season>_wkNN_card_mark4.csv, mark4/live.py) and its legs; Marks 1-3
stay on the page as tracked. Run after make_week_page.py (card_all.py does both).

  python scripts/week_page_mark4.py --season 2026 --week 7
"""
import argparse
import html
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

D = config.ROOT / "records" / "cards"
SLOT_OF = {"boost": "the boost window before its first game"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    a = ap.parse_args(argv)
    page = D / f"{a.season}_wk{a.week:02d}_picks.html"
    card = pl.read_csv(D / f"{a.season}_wk{a.week:02d}_card_mark4.csv")
    legs_all = pl.read_csv(D / f"{a.season}_wk{a.week:02d}_legs_mark4.csv")
    e = html.escape
    slips = []
    for (pid,), g in card.group_by("parlay", maintain_order=True):
        r0 = g.row(0, named=True)
        rows = []
        for n, r in enumerate(g.iter_rows(named=True), 1):
            flag = '<p class="flag">Promotion overreaction: FanDuel raised this line far more than his role grew.</p>' if r["overreaction_under"] else ""
            rows.append(f'<div class="legrow"><span class="r">{n}</span><span class="who">{e(r["player"])}<span class="team">{e(str(r["team"]))}</span></span>'
                        f'<span class="price">{r["price"]:+d}</span><span class="bet"><b>{e(r["side"].upper())} {r["line"]:g}</b> {e(r["market"].lower())}</span>'
                        f'<span class="meta">{e(r["game"])} · {e(str(r["gameday"]))} · chance {r["chance"]:.1%} · edge {r["edge"]:+.1%}</span>{flag}</div>')
        dec = (1 + r0["parlay_price"] / 100) if r0["parlay_price"] > 0 else (1 + 100 / -r0["parlay_price"])
        bdec = 1 + (dec - 1) * 1.3
        slips.append(f'<div class="slip" data-u="{r0["stake_units"]}" data-bdec="{bdec:.6f}"><div class="slip-head"><span class="n">{e(pid)} · {r0["stake_units"]:g}u</span>'
                     f'<span class="odds">{r0["parlay_price"]:+d} → {r0["boosted_price"]:+d}</span></div>{"".join(rows)}'
                     f'<div class="slip-foot"><span>chance all hit <b>{r0["parlay_chance"]:.1%}</b></span><span>expected value with the boost <b>{r0["parlay_ev_boosted"]:+.0%}</b></span>'
                     f'<span><span class="boost">30% profit boost</span></span></div><div class="unitline"></div></div>')
    n_filled = card["parlay"].n_unique()
    # when the card's inputs were pulled: FanDuel lines in the log, the injury report in the database
    import duckdb
    lg = pl.read_csv(config.ROOT / "records" / "lines" / "prop_lines_log.csv", infer_schema_length=0).filter(
        (pl.col("book") == "FanDuel") & pl.col("game_id").str.starts_with(f"{a.season}_{a.week:02d}_"))
    lines_at = str(lg["captured_at"].max())[:16].replace("T", " ") + " UTC" if lg.height else "unknown"
    con = duckdb.connect(str(config.RAW_DUCKDB_PATH), read_only=True)
    inj = con.execute("SELECT max(pulled_at) FROM injuries WHERE season = ? AND week = ?", [a.season, a.week]).fetchone()[0]
    con.close()
    inj_at = str(inj)[:16] + " UTC" if inj else "unknown"
    asof = f"FanDuel lines from {lines_at}; injury report pulled {inj_at}"
    mlabel = {"rec": "receptions", "rec_yds": "receiving yards", "rush_yds": "rushing yards", "pass_yds": "passing yards", "rush_att": "rushing attempts", "pass_att": "passing attempts", "pass_cmp": "passing completions"}
    legrows = "".join(f'<tr><td>{"<span class=\'tag bet\'>On the card</span>" if (r["player"], float(r["line"])) in set(zip(card["player"].to_list(), [float(x) for x in card["line"].to_list()])) else "<span class=\'tag\'>Edge, not used</span>"}</td>'
                      f'<td>{e(r["player"])} <small class="note">{e(str(r["team"]))}</small></td><td>{e(r["side"])} {r["line"]:g} {e(mlabel.get(r["market"], r["market"]))}</td><td class="num">{int(r["price"]):+d}</td>'
                      f'<td class="num">{r["chance"]:.1%}</td><td class="num">{r["book"] if r["side"] == "Over" else 1 - r["book"]:.1%}</td><td class="num">{r["edge"]:+.1%}</td>'
                      f'<td>{e(r["game_id"])}</td></tr>' for r in legs_all.iter_rows(named=True))
    mlabel = {"rec": "receptions", "rec_yds": "receiving yards", "rush_yds": "rushing yards", "pass_yds": "passing yards", "rush_att": "rushing attempts", "pass_att": "passing attempts", "pass_cmp": "passing completions"}
    legs_all_note = f"{legs_all.height} FanDuel lines had an edge this pull."
    wrap4 = f'''
  <div class="markwrap" data-mark="4">
  <section>
    <h2>Mark 4 · the parlays</h2>
    <p class="lede">One boost per parlay: the 5-leg at a quarter unit, the 3-leg at half a unit, the two 2-legs at a full unit. Built long-first from the legs with the most
    expected value (over or under, whichever is better), FanDuel's own lines only, never two legs of one game. Place each in the boost window before its first game.
    {n_filled} of 4 filled from this pull; the rest fill when FanDuel posts more markets.</p>
    <p class="lede"><b>Built on: {asof}.</b> If a line or price on FanDuel no longer matches the card, send a fresh pull before betting.</p>
    <div class="unitbox"><label for="unit">1 unit =</label> <span class="dollar">$</span><input id="unit" type="number" inputmode="decimal" min="0" step="1" value="10" aria-describedby="unitsum"><span id="unitsum" class="unitsum"></span></div>
    <div class="grid" style="margin-top:14px">{"".join(slips)}</div>
  </section>
  <style>
  .markwrap,.markwrap>section{{min-width:0}}
  .switch{{flex-wrap:wrap}} .switch button{{white-space:nowrap}}
  .unitbox{{display:flex;flex-wrap:wrap;align-items:center;gap:8px 10px;margin-top:14px;background:var(--turf-soft);border-radius:10px;padding:12px 14px}}
  .unitbox label{{font:700 1.15rem/1 var(--display);text-transform:uppercase;letter-spacing:.04em}}
  .unitbox .dollar{{font:600 1.1rem/1 var(--mono)}}
  .unitbox input{{font:600 1.15rem/1 var(--mono);width:7em;padding:.45rem .55rem;border:1px solid var(--rule);border-radius:8px;background:var(--panel);color:var(--ink)}}
  .unitbox input:focus-visible{{outline:2px solid var(--turf);outline-offset:2px}}
  .unitsum{{font-size:.92rem;color:var(--ink)}}
  .unitline{{background:var(--turf-soft);border-radius:8px;padding:.55rem .7rem;font:600 1rem/1.35 var(--body)}}
  .unitline b{{font-family:var(--mono)}}
  </style>
  <script>
  (() => {{
    const inp = document.getElementById("unit");
    const money = v => "$" + (Math.round(v * 100) / 100).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    try {{ const v = localStorage.getItem("unit-dollars"); if (v) inp.value = v; }} catch (e) {{}}
    function upd() {{
      const u = Math.max(0, parseFloat(inp.value) || 0);
      try {{ localStorage.setItem("unit-dollars", String(u)); }} catch (e) {{}}
      let tot = 0, win = 0;
      document.querySelectorAll(".slip[data-u]").forEach(el => {{
        const stake = u * parseFloat(el.dataset.u), profit = stake * (parseFloat(el.dataset.bdec) - 1);
        tot += stake; win += profit;
        el.querySelector(".unitline").innerHTML = `Bet <b>${{money(stake)}}</b> with the boost · if it hits: wins <b>${{money(profit)}}</b>, pays back <b>${{money(stake + profit)}}</b>`;
      }});
      document.getElementById("unitsum").innerHTML = `Total to bet this week: <b>${{money(tot)}}</b> · if every parlay hits: <b>+${{money(win)}}</b>`;
    }}
    inp.addEventListener("input", upd); upd();
  }})();
  </script>
  <section class="panel">
    <h2>Mark 4 · every leg with an edge</h2>
    <p class="lede">{legs_all_note} Chance = Mark 4l (Mark 3's picker mixed 50/50 with FanDuel's no-vig price, the 2026 weights chosen from earlier seasons only; promotion-overreaction
    unders at the hit rate of earlier seasons). Book = FanDuel's own no-vig chance for the same side.</p>
    <div class="scroll"><table><thead><tr><th>Status</th><th>Player</th><th>Bet</th><th class="num">FanDuel</th><th class="num">Mark 4 chance</th><th class="num">Book chance</th><th class="num">Edge</th><th>Game</th></tr></thead><tbody>{legrows}</tbody></table></div>
  </section>
  <section class="panel">
    <h2>Why Mark 4</h2>
    <p class="lede">Played week by week 2022 to 2026 week 6 on this exact card: +106.8 units on 151.8 bet, up every season (+13, +22, +12, +25, +35). Every parlay its top legs could build made
    money every season at every length from 2 to 5 legs (Mark 3: 4-5 leg parlays lost in 2024 and 2025). It beats FanDuel's own price on log loss in every season it was played forward
    (2023-2026). Marks 1-3 stay on this page, tracked, not bet.</p>
  </section>
  </div>
'''
    s = page.read_text()
    if 'data-mark="4"' in s:
        raise SystemExit("page already has Mark 4: rebuild it first (scripts/make_week_page.py)")
    # switch: Mark 4 first and pressed, Mark 3 tracked
    s = s.replace('<button type="button" data-m="3" aria-pressed="true">Mark 3 · the bet</button>',
                  '<button type="button" data-m="4" aria-pressed="true">Mark 4 · the bet</button>\n      <button type="button" data-m="3" aria-pressed="false">Mark 3 · tracked</button>')
    i = s.index('  <div class="markwrap" data-mark="1"')
    s = s[:i] + wrap4 + s[i:]
    hero4 = (f'  4: {{eyebrow: "Mark 4 · the bet · FanDuel · {a.season} week {a.week}",\n'
             f'      sum: {json.dumps([f"{n_filled} boosted parlays", "5-leg 0.25u · 3-leg 0.5u · 2-leg 1u", "Over or under per leg", "No straight bets"])},\n'
             '      p: "These are the bets: one 30% profit boost per parlay. Rebuilt before each boost window with fresh FanDuel lines and the final injury report: use the latest version of this page."},\n')
    s = s.replace("const HERO = {\n", "const HERO = {\n" + hero4)
    s = s.replace('eyebrow: "Mark 3 · the bet', 'eyebrow: "Mark 3 · tracked, not bet')
    s = s.replace('p: "These are the bets: up to four 2-leg parlays', 'p: "Tracked, not bet: Mark 4 is the bet from week 7. Mark 3 would bet up to four 2-leg parlays')
    s = s.replace("let mark = 3;", "let mark = 4;")
    s = s.replace('localStorage.getItem("wk-mark-v3"); if (m === "1" || m === "2" || m === "3") mark = +m;',
                  'localStorage.getItem("wk-mark-v4"); if (m === "1" || m === "2" || m === "3" || m === "4") mark = +m;')
    s = s.replace('localStorage.setItem("wk-mark-v3", String(m));', 'localStorage.setItem("wk-mark-v4", String(m));')
    s = s.replace("function renderSlots() {\n", 'function renderSlots() {\nif (mark === 4) { $("#slots").innerHTML = `<p class="none">Mark 4 picks over or under per leg from every market: its legs are listed above. The slot table below shows the tracked Marks\' unders.</p>`; return; }\n', 1)
    for old in ('aria-pressed="true">Mark 3', ):
        assert old not in s
    page.write_text(s)
    print("wrote", page)


if __name__ == "__main__":
    main()
