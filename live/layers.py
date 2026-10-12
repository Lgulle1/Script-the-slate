"""Which Phase 4 layers a live prediction uses: layer_weights.csv decides, the code holds only the registry of builders (plan P5.7).

A layer is on for a market exactly when the file gives weight 1.0 to (layer, market). Weights other than 0.0 and 1.0 are refused (the one weight rule is
binary). A layer on without a builder, or two layers on for one market, stops the run: nothing is dropped or chosen silently.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

import config

CODE = {"4a_injuries": "4a", "4b_simulation": "4b"}          # the layers that have a live builder; their stack code
BUILDERS = set(CODE)


class LayerError(RuntimeError):
    pass


def read_active(path=None) -> dict:
    """{layer: {markets}} for every (layer, market) with weight 1.0."""
    t = pl.read_csv(Path(path or config.ROOT / "layer_weights.csv"))
    bad = t.filter(~pl.col("weight").is_in([0.0, 1.0]))
    if bad.height:
        raise LayerError(f"weights must be 0.0 or 1.0 (the one weight rule); found {bad['weight'].to_list()} for {bad.select('layer', 'market').rows()}")
    out: dict = {}
    for r in t.filter(pl.col("weight") == 1.0).iter_rows(named=True):
        out.setdefault(r["layer"], set()).add(r["market"])
    return out


def validate(active: dict) -> None:
    missing = sorted(set(active) - BUILDERS)
    if missing:
        raise LayerError(f"layer(s) {missing} have weight 1.0 in layer_weights.csv but no live builder: wire them before running (a layer is never dropped silently)")
    claims: dict = {}
    for layer, markets in active.items():
        for m in markets:
            claims.setdefault(m, []).append(layer)
    clash = {m: ls for m, ls in claims.items() if len(ls) > 1}
    if clash:
        raise LayerError(f"markets claimed by more than one layer: {clash}")


def layer_of(market: str, active: dict):
    for layer, markets in active.items():
        if market in markets:
            return layer
    return None


def stack_label(layer) -> str:
    return "phase3" if layer is None else f"phase3+{CODE[layer]}"
