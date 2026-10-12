"""Which layers a prediction uses comes from layer_weights.csv alone (plan P5.7 items 1-2)."""
import polars as pl
import pytest

from live import layers as LY


def _csv(tmp_path, rows):
    p = tmp_path / "w.csv"
    pl.DataFrame(rows, schema={"layer": pl.String, "market": pl.String, "verdict": pl.String, "weight": pl.Float64}, orient="row").write_csv(p)
    return p


def test_weight_one_turns_a_layer_on_for_that_market_only(tmp_path):
    p = _csv(tmp_path, [("4a_injuries", "rush_att", "CLEARS", 1.0), ("4a_injuries", "pass_att", "EDGE", 0.0), ("4b_simulation", "spread", "CLEARS", 1.0), ("4c_comparables", "rec", "NO", 0.0)])
    a = LY.read_active(p)
    assert a == {"4a_injuries": {"rush_att"}, "4b_simulation": {"spread"}}
    assert LY.layer_of("rush_att", a) == "4a_injuries" and LY.layer_of("pass_att", a) is None and LY.layer_of("spread", a) == "4b_simulation"
    assert LY.stack_label("4a_injuries") == "phase3+4a" and LY.stack_label("4b_simulation") == "phase3+4b" and LY.stack_label(None) == "phase3"


def test_changing_a_weight_changes_the_stack_with_no_code_change(tmp_path):
    on = LY.read_active(_csv(tmp_path, [("4a_injuries", "pass_att", "CLEARS", 1.0)]))
    off = LY.read_active(_csv(tmp_path, [("4a_injuries", "pass_att", "CLEARS", 0.0)]))
    assert LY.layer_of("pass_att", on) == "4a_injuries" and LY.layer_of("pass_att", off) is None


def test_a_layer_at_weight_one_without_a_builder_stops_the_run(tmp_path):
    a = LY.read_active(_csv(tmp_path, [("4c_comparables", "rec", "CLEARS", 1.0)]))
    with pytest.raises(LY.LayerError, match="4c_comparables"):
        LY.validate(a)


def test_two_layers_claiming_one_market_is_an_error_not_a_silent_choice(tmp_path):
    a = LY.read_active(_csv(tmp_path, [("4a_injuries", "spread", "CLEARS", 1.0), ("4b_simulation", "spread", "CLEARS", 1.0)]))
    with pytest.raises(LY.LayerError, match="spread"):
        LY.validate(a)


def test_a_weight_other_than_zero_or_one_is_refused(tmp_path):
    with pytest.raises(LY.LayerError, match="0.5"):
        LY.read_active(_csv(tmp_path, [("4a_injuries", "rec", "EDGE", 0.5)]))


def test_the_committed_weights_file_reads_and_validates():
    a = LY.read_active()
    LY.validate(a)
    assert a["4a_injuries"] == {"rush_att", "rush_yds", "targets", "rec", "rec_yds"} and a["4b_simulation"] == {"spread"}
