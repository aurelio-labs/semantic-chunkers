"""Text segmentation metrics.

Boundaries are expressed as sorted lists of sentence indices, where index i
means "a new segment starts at sentence i". Index 0 is never a boundary.

Pk (Beeferman et al. 1999) and WindowDiff (Pevzner and Hearst 2002) are the
standard error rates for text segmentation; lower is better, 0 is perfect.
Boundary F1 with a tolerance window is the more intuitive companion; higher
is better.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile of ``values`` (0-100), linearly interpolated.

    Same definition as ``numpy.percentile`` with the default ``linear``
    method, written out so the harness has no numeric dependency for it.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (q / 100) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def _segment_ids(boundaries: Iterable[int], n_sentences: int) -> list[int]:
    """Map each sentence index to the id of the segment it belongs to."""
    cuts = set(boundaries)
    ids = []
    current = 0
    for i in range(n_sentences):
        if i in cuts and i > 0:
            current += 1
        ids.append(current)
    return ids


def default_k(reference: Sequence[int], n_sentences: int) -> int:
    """Half the mean reference segment length, the conventional window size."""
    n_segments = len([b for b in reference if 0 < b < n_sentences]) + 1
    k = max(1, round(n_sentences / n_segments / 2))
    return min(k, max(1, n_sentences - 1))


def pk(
    reference: Sequence[int],
    hypothesis: Sequence[int],
    n_sentences: int,
    k: int | None = None,
) -> float:
    """Probability that two sentences k apart are wrongly judged same- or different-segment."""
    if n_sentences < 2:
        return 0.0
    k = default_k(reference, n_sentences) if k is None else k
    ref = _segment_ids(reference, n_sentences)
    hyp = _segment_ids(hypothesis, n_sentences)
    errors = 0
    windows = n_sentences - k
    for i in range(windows):
        ref_same = ref[i] == ref[i + k]
        hyp_same = hyp[i] == hyp[i + k]
        if ref_same != hyp_same:
            errors += 1
    return errors / windows if windows > 0 else 0.0


def windowdiff(
    reference: Sequence[int],
    hypothesis: Sequence[int],
    n_sentences: int,
    k: int | None = None,
) -> float:
    """Fraction of windows where the number of boundaries differs between reference and hypothesis."""
    if n_sentences < 2:
        return 0.0
    k = default_k(reference, n_sentences) if k is None else k
    ref = set(b for b in reference if 0 < b < n_sentences)
    hyp = set(b for b in hypothesis if 0 < b < n_sentences)
    errors = 0
    windows = n_sentences - k
    for i in range(windows):
        # boundaries strictly inside the window (i, i+k]
        r = sum(1 for b in range(i + 1, i + k + 1) if b in ref)
        h = sum(1 for b in range(i + 1, i + k + 1) if b in hyp)
        if r != h:
            errors += 1
    return errors / windows if windows > 0 else 0.0


def boundary_prf(
    reference: Sequence[int],
    hypothesis: Sequence[int],
    n_sentences: int,
    tolerance: int = 1,
) -> tuple[float, float, float]:
    """Precision, recall, F1 of hypothesised boundaries against reference boundaries.

    A hypothesis boundary counts as correct if a reference boundary lies within
    ``tolerance`` sentences and has not already been matched.
    """
    ref = sorted(b for b in reference if 0 < b < n_sentences)
    hyp = sorted(b for b in hypothesis if 0 < b < n_sentences)
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    matched_ref: set[int] = set()
    tp = 0
    for h in hyp:
        candidates = [
            r for r in ref if abs(r - h) <= tolerance and r not in matched_ref
        ]
        if candidates:
            best = min(candidates, key=lambda r: abs(r - h))
            matched_ref.add(best)
            tp += 1
    precision = tp / len(hyp) if hyp else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
