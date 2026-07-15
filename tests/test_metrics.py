import math

import pytest

from vla.eval.metrics import wilson_interval, paraphrase_gap, ConditionResult


def test_wilson_point_estimate_matches_ratio():
    w = wilson_interval(15, 20)
    assert w.point == pytest.approx(0.75)


def test_wilson_bounds_are_inside_unit_interval_at_extremes():
    # 20/20: the normal approximation would give an upper bound > 1; Wilson must not.
    # (Wilson's upper bound equals exactly 1.0 here, but its LOWER bound stays
    # well below 1 — that lower bound is the honest thing to report.)
    w = wilson_interval(20, 20)
    assert 0.0 <= w.low <= 1.0
    assert w.high <= 1.0
    assert w.low < 1.0  # a perfect 20/20 still admits a plausibly lower true rate

    # 0/20: lower bound must stay >= 0.
    w0 = wilson_interval(0, 20)
    assert w0.low >= 0.0
    assert w0.point == 0.0


def test_wilson_interval_brackets_point():
    w = wilson_interval(12, 20)
    assert w.low < w.point < w.high


def test_wilson_zero_trials_is_degenerate():
    w = wilson_interval(0, 0)
    assert (w.point, w.low, w.high) == (0.0, 0.0, 1.0)


def test_wilson_rejects_out_of_range():
    with pytest.raises(ValueError):
        wilson_interval(21, 20)


def test_wilson_width_shrinks_with_more_trials():
    narrow = wilson_interval(80, 100)
    wide = wilson_interval(8, 10)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_paraphrase_gap_sign():
    assert paraphrase_gap(0.8, 0.5) == pytest.approx(0.3)
    assert paraphrase_gap(0.5, 0.8) == pytest.approx(-0.3)


def test_condition_result_rates():
    r = ConditionResult("c", n=20, successes=10, partials=4, failures=6)
    assert r.success_rate.point == pytest.approx(0.5)
    assert r.partial_or_success_rate.point == pytest.approx(0.7)
