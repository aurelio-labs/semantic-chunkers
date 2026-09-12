from benchmarks import metrics


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
