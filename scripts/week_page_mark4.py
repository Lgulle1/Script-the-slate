"""Adds the Mark 4 card (records/cards/<season>_wkNN_card_mark4.csv, mark4/live.py) to the week's picks page as the bet; Marks 1-3 stay tracked.

  python scripts/week_page_mark4.py --season 2026 --week 7
"""
import argparse
import html
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

D = config.ROOT / "records" / "cards"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    a = ap.parse_args(argv)
    page = D / f"{a.season}_wk{a.week:02d}_picks.html"
    card = pl.read_csv(D / f"{a.season}_wk{a.week:02d}_card_mark4.csv")
    e = html.escape
    slips = []
    for (pid,), g in card.group_by("parlay", maintain_order=True):
        r0 = g.row(0, named=True)
        legs = []
        for n, r in enumerate(g.iter_rows(named=True), 1):
            flag = '<p class="flag">Promotion overreaction: FanDuel raised this line far more than his role grew.</p>' if r["overreaction_under"] else ""
            legs.append(f'<div class="legrow"><span class="r">{n}</span><span class="who">{e(r["player"])}<span class="team">{e(str(r["team"]))}</span></span>'
                        f'<span class="price">{r["price"]:+d}</span><span class="bet"><b>{e(r["side"])} {r["line"]:g}</b> {e(r["market"])}</span>'
                        f'<span class="meta">{e(r["game"])} · chance {r["chance"]:.1%} · edge {r["edge"]:+.1%}</span>{flag}</div>')
        slips.append(f'<div class="slip"><div class="slip-head"><span class="n">{e(pid)} · {r0["stake_units"]:g}u</span>'
                     f'<span class="odds">{r0["parlay_price"]:+d} → {r0["boosted_price"]:+d} boosted</span></div>{"".join(legs)}'
                     f'<div class="slip-foot"><span>chance <b>{r0["parlay_chance"]:.1%}</b></span><span>expected value boosted <b>{r0["parlay_ev_boosted"]:+.0%}</b></span>'
                     f'<span><span class="boost">30% profit boost</span></span></div></div>')
    n_filled = card["parlay"].n_unique()
    section = f'''
  <section class="panel" id="mark4">
    <h2>Mark 4 · the bet</h2>
    <p class="lede">Mark 4l: Mark 3's picker mixed 50/50 with FanDuel's own price (the 2026 weights, chosen from earlier seasons only), plus promotion-overreaction unders.
    Legs ranked by expected value, over or under per leg, FanDuel lines only, never two legs of one game. One boost per parlay: 5-leg at a quarter unit, 3-leg at half a unit,
    two 2-legs at a full unit, built long-first. {n_filled} of 4 slots filled from this pull; the rest fill when more of FanDuel's markets are posted.
    First pass on the Wednesday practice report: rebuilt before each boost window with fresh lines and the final injury report.</p>
    <div class="grid" style="margin-top:14px">{"".join(slips)}</div>
  </section>
'''
    s = page.read_text()
    if 'id="mark4"' in s:
        s = s[: s.index('\n  <section class="panel" id="mark4">')] + s[s.index("</section>\n", s.index('id="mark4"')) + len("</section>\n"):]
    i = s.index("</section>", s.index('<section class="hero">')) + len("</section>")
    s = s[:i] + section + s[i:]
    s = s.replace('Mark 3 · the bet</button>', 'Mark 3 · tracked</button>').replace('eyebrow: "Mark 3 · the bet', 'eyebrow: "Mark 3 · tracked, not bet')
    s = s.replace('p: "These are the bets: up to four 2-leg parlays', 'p: "Tracked, not bet: Mark 4 (above) is the bet from week 7. Mark 3 would bet up to four 2-leg parlays')
    page.write_text(s)
    print("wrote", page)


if __name__ == "__main__":
    main()
