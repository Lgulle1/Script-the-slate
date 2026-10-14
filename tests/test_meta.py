"""Phase 6.1 stage-1 shrinkage blend and 6.2 stage-2 stacking (eval/meta.py): rules of docs/overnight_plan.md, "Phase 7 app and Phase 6.1 / 6.2"."""
import numpy as np
import polars as pl
import pytest

from eval import meta as M


def _rows(n_weeks=12, per=150, seed=0, model_noise=1.0, base_noise=2.0):
    rng = np.random.default_rng(seed)
    out = []
    for w in range(n_weeks):
        truth = rng.uniform(2, 12, per)
        out.append(pl.DataFrame(dict(market="targets", season=2021 + w // 6, week=w % 6 + 1, entity=[f"e{w}_{i}" for i in range(per)], actual=truth,
                                      model=truth + rng.normal(0, model_noise, per), baseline=truth + rng.normal(0, base_noise, per))))
    return pl.concat(out)


def test_a_zero_reproduces_the_baseline_and_a_one_reproduces_the_model_exactly():
    r = _rows()
    assert np.array_equal(M.blend(r["model"].to_numpy(), r["baseline"].to_numpy(), 0.0), r["baseline"].to_numpy())
    assert np.array_equal(M.blend(r["model"].to_numpy(), r["baseline"].to_numpy(), 1.0), r["model"].to_numpy())


def test_the_shrunk_weight_is_a_fit_times_n_over_n_plus_500_and_stays_in_zero_one():
    assert M.shrink(0.8, 500) == pytest.approx(0.4) and M.shrink(1.0, 0) == 0.0 and 0.0 <= M.shrink(1.0, 10 ** 9) <= 1.0


def test_each_weeks_weight_is_fit_only_on_strictly_earlier_weeks():
    r = _rows()
    base = M.stage1(r, "targets")
    changed = r.with_columns(actual=pl.when((pl.col("season") == 2022) & (pl.col("week") >= 3)).then(pl.col("baseline")).otherwise(pl.col("actual")))      # from week 3 on the baseline is exactly right
    again = M.stage1(changed, "targets")
    keep = lambda d: d.filter((pl.col("season") == 2021) | ((pl.col("season") == 2022) & (pl.col("week") <= 3))).sort("season", "week").select("a_used", "final")
    assert keep(base).equals(keep(again))                                    # through week 3 of 2022 (whose own actuals changed but are not used for its own weight)
    assert not base.filter((pl.col("season") == 2022) & (pl.col("week") == 4))["a_used"].equals(again.filter((pl.col("season") == 2022) & (pl.col("week") == 4))["a_used"])


def test_a_better_model_gets_a_higher_weight_than_a_worse_one_and_the_blend_is_in_the_grid():
    good = M.stage1(_rows(model_noise=0.5, base_noise=3.0), "targets")["a_fit"].max()
    bad = M.stage1(_rows(model_noise=3.0, base_noise=0.5), "targets")["a_fit"].max()
    assert good > bad and 0.0 <= bad <= good <= 1.0
    assert set(np.round(M.stage1(_rows(), "targets")["a_fit"].unique().to_numpy() * 20) % 1) == {0.0}      # a multiple of 0.05


def test_stage_two_does_not_run_below_the_activation_threshold():
    r = _rows(n_weeks=12, per=150)                                           # 1,800 rows
    assert M.stage2_active(r, "targets", min_games=3000) is False and M.stage2_active(r, "targets", min_games=1000) is True
    assert M.stage2(r, "targets", min_games=3000) is None


def test_stage_two_is_walk_forward_by_season_and_only_scores_seasons_with_earlier_training_rows():
    r = _rows(n_weeks=18, per=200)                                           # 3 seasons of 6 weeks
    out = M.stage2(r, "targets", min_games=1000)
    assert out is not None and set(out["season"].unique().to_list()) == {2022, 2023}
    assert out["pred"].null_count() == 0
