import pytest

from benchmarks import metrics


def test_percentile_matches_numpys_linear_definition():
    import numpy as np

    for values in ([1.0, 2.0, 3.0, 4.0], [0.3], [5.0, 1.0, 9.0], list(range(30))):
        for q in (0, 5, 50, 95, 100):
            assert metrics.percentile([float(v) for v in values], q) == pytest.approx(
                float(np.percentile(np.asarray(values, dtype=float), q))
            )
    assert metrics.percentile([], 50) == 0.0


def test_percentile_ignores_input_order():
    ordered = [1.0, 2.0, 3.0, 4.0]
    assert metrics.percentile([4.0, 1.0, 3.0, 2.0], 50) == metrics.percentile(
        ordered, 50
    )


def test_perfect_segmentation_scores_zero_error_and_full_f1():
    ref = [4, 8]
    assert metrics.pk(ref, ref, 12) == 0.0
    assert metrics.windowdiff(ref, ref, 12) == 0.0
    assert metrics.boundary_prf(ref, ref, 12) == (1.0, 1.0, 1.0)


def test_no_boundaries_is_penalised():
    ref = [4, 8]
    assert metrics.pk(ref, [], 12) > 0.0
    assert metrics.windowdiff(ref, [], 12) > 0.0
    p, r, f = metrics.boundary_prf(ref, [], 12)
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_tolerance_matches_near_boundaries_once():
    ref = [4, 8]
    p, r, f = metrics.boundary_prf(ref, [5, 9], 12, tolerance=1)
    assert (p, r, f) == (1.0, 1.0, 1.0)
    p, r, f = metrics.boundary_prf(ref, [5, 6], 12, tolerance=1)
    assert p == 0.5 and r == 0.5


def test_windowdiff_counts_extra_boundaries():
    ref = [6]
    over = [3, 6, 9]
    assert metrics.windowdiff(ref, over, 12) > metrics.windowdiff(ref, ref, 12)
