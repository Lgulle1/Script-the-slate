"""Every comp component for every player-game row 2021-2026, with the settings the practice ladder picked (records/tuning/comps2_ladder.csv; fixed
2026-10-19 before any 2024-2026 row was evaluated):
  own_*   the player's own past games, no similarity (control won in practice)
  dfn_*   tonight's defense vs the family, no similarity (control won)
  int_*   league-wide (offense, defense) pair + archetype similarity, h = 0.75 (beat its control in rush, rec and pass)
  sit_*   role-change / played-hurt situations, h = 0.5
Each uses only games before the row's gameday, so 2025-2026 rows are built exactly as a live week would be.
"""
import time

import polars as pl

import config
from comps2 import searches as S
from comps2 import situations as SI

OUT = config.PROCESSED_DIR / "comps2_components.parquet"


def build():
    e = S.load()
    tg = e.filter(pl.col("season") >= 2021)["rid"].to_numpy()
    t0 = time.time()
    flat = S.run(e, S.Params(h_def=None, h_off=None, h_arch=None, h_pair=None), tg, ("own", "dfn"))
    print(f"own+dfn {time.time() - t0:.0f}s", flush=True)
    it = S.run(e, S.Params(h_pair=0.75, h_arch=0.75), tg, ("int",)).select("rid", "int_shift", "int_neff", "int_sw")
    print(f"int {time.time() - t0:.0f}s", flush=True)
    sit = SI.search(e, pl.read_parquet(SI.OUT), tg, h=0.5)
    c = flat.join(it, on="rid").join(sit, on="rid")
    c.write_parquet(OUT)
    print(c.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    build()
