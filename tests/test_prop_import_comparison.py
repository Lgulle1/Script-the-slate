"""scripts/prop_import_comparison.py: the arithmetic of the week-5 model-vs-line comparison."""
import importlib.util
from pathlib import Path

import polars as pl

spec = importlib.util.spec_from_file_location("pic", Path(__file__).resolve().parents[1] / "scripts" / "prop_import_comparison.py")
pic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pic)


def _x():
    return pl.DataFrame(dict(game_id=["g1", "g1", "g2", "g2", "g3"], market=["rec"] * 5, model=[6.0, 3.0, 5.0, 4.0, 7.0], last_line=[5.5, 3.5, 5.5, 4.5, 6.5],
                             open_line=[5.5, 3.5, 5.5, 4.5, 6.5], actual=[7.0, 2.0, 4.0, 5.0, 8.0]))


def test_gain_is_positive_when_the_model_is_closer_than_the_line():
    gain, lo, hi = pic.boot_diff(_x(), "last_line")
    err_line = [1.5, 1.5, 1.5, 0.5, 1.5]
    err_model = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert abs(gain - (sum(err_line) - sum(err_model)) / 5) < 1e-9 and lo <= gain <= hi


def test_hit_rate_counts_the_model_leaning_over_when_above_the_line_and_removes_pushes():
    x = _x().with_columns(actual=pl.Series([7.0, 2.0, 4.0, 5.0, 6.5]))                # the last row is a push
    h = pic.hit_rates(x, "last_line")
    assert h["n_lean"] == 4 and h["lean_over_share"] == 0.25
    # leans: over/hit (7>5.5), under/hit (2<3.5), under/hit (4<5.5), under/miss (5>4.5) -> 3 of 4; the real number went over in 2 of 4
    assert h["hit_all"] == 0.75 and h["always_over"] == 0.5


def test_the_bootstrap_is_deterministic():
    assert pic.boot_diff(_x(), "last_line") == pic.boot_diff(_x(), "last_line")
