"""The locked picker's replay pool (data/processed/replay_pool_uncapped.parquet): every qualifying pick (4+ edge, 3+ games with the team) for every replay week
(2024 week 4 - 2026 week 5), the picker retrained each week on every line before that week. The Track Record replays the strategies from it.
Rebuilt 2026-10-16 after 2021-2023 joined the line study (the first pool was built from 2024 on only, and its picker was far less cautious).

  python scripts/replay_pool.py
"""
import importlib.util
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("card_replay", HERE / "card_replay.py")
CR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CR)
CF, TN = CR.CF, CR.TN
COLS = ["season", "week", "game_id", "player_id", "player_name", "team", "market", "line", "dec_u", "mean", "actual", "picker_under_chance", "won", "profit"]


def main():
    d = TN.build()
    A = TN.Arr(d)
    X = TN.feats(d, layered=True)
    d = d.with_columns(history=pl.Series(CR.history_counts(d)))
    out = []
    for s, ws in CR.WEEKS.items():
        for w in ws:
            out.append(CF.picks_for(d, A, X, s, w, True).filter(pl.col("history") >= CF.MIN_HISTORY).select(COLS))
        print(s, "done", flush=True)
    pl.concat(out).write_parquet(config.PROCESSED_DIR / "replay_pool_uncapped.parquet")


if __name__ == "__main__":
    main()
