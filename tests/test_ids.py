import polars as pl
import pytest

from ingest import ids

MASTER = pl.DataFrame({
    "gsis_id": ["G1", "G2", "G3", "G4"],
    "pfr_id": ["P1", "P2", "P3", "DUP"],
    "espn_id": ["1", "2", "3", "DUP2"],
})
MASTER_AMBIG = pl.concat([MASTER, pl.DataFrame({"gsis_id": ["G5"], "pfr_id": ["DUP"], "espn_id": ["5"]})])


def test_resolves_each_id_type_and_preserves_order_and_columns():
    df = pl.DataFrame({"pfr_player_id": ["P2", "P1", "P3"], "x": [1, 2, 3]})
    out = ids.add_canonical_gsis_id(df, master=MASTER)
    assert out["gsis_id"].to_list() == ["G2", "G1", "G3"]
    assert out["x"].to_list() == [1, 2, 3]
    out = ids.add_canonical_gsis_id(pl.DataFrame({"espn_id": [3, 1]}), master=MASTER)  # int espn ids
    assert out["gsis_id"].to_list() == ["G3", "G1"]


def test_falls_through_to_next_id_column():
    df = pl.DataFrame({"gsis_id": ["G1", None], "pfr_id": [None, "P2"]})
    assert ids.add_canonical_gsis_id(df, master=MASTER)["gsis_id"].to_list() == ["G1", "G2"]


def test_raises_over_threshold_not_silent_null():
    df = pl.DataFrame({"pfr_id": ["P1"] * 98 + ["nope", "nada"]})
    with pytest.raises(ids.IdMatchError, match="2/100"):
        ids.add_canonical_gsis_id(df, master=MASTER)


def test_at_or_under_threshold_keeps_null():
    df = pl.DataFrame({"pfr_id": ["P1"] * 99 + ["nope"]})
    out = ids.add_canonical_gsis_id(df, master=MASTER)
    assert out["gsis_id"].null_count() == 1 and ids.MAX_UNMATCHED_FRACTION == 0.01


def test_ambiguous_master_id_is_unmatched_not_guessed():
    df = pl.DataFrame({"pfr_id": ["DUP"] * 50})
    with pytest.raises(ids.IdMatchError):
        ids.add_canonical_gsis_id(df, master=MASTER_AMBIG)


def test_needs_an_id_column():
    with pytest.raises(ValueError, match="at least one"):
        ids.add_canonical_gsis_id(pl.DataFrame({"foo": [1]}), master=MASTER)
