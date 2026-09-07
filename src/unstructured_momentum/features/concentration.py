"""Statistics for "does a SUBSET of this book share one direction?"

Why not mean pairwise cosine
----------------------------
Mean pairwise cosine is algebraically the squared centroid norm: for k unit vectors it equals
(k‖c‖² − 1)/(k − 1). A sub-bet covering a fraction f of the book therefore enters the
statistic as f SQUARED. On the 2019 book, at the observed null spread, a forty-percent
sub-bet would have needed internal cosine of 0.138 to move the statistic two standard
deviations — while a whole GICS sector inside that leg achieves between 0.00 and 0.13. The
test could not have passed even if its hypothesis were true, and a positive control on
single-sector groups passed at z between 6.8 and 14.6, which measured power against a
hundred-percent-loading alternative that was not the one on trial.

The two statistics here are built for the actual alternative. The leading eigenvalue asks how
much of the leg's variation lies along one direction, which is what "one shared bet" means.
The densest-subset scan asks whether SOME subset is unusually cohesive, without being told
which one or how large.
"""

from __future__ import annotations

import numpy as np


def leading_eigenvalue(Ec: np.ndarray) -> float:
    """Largest eigenvalue of the similarity matrix, scaled by size.

    One shared direction across a subset shows up here where it cannot show up in a mean:
    the eigenvalue is driven by the dominant direction rather than diluted by every pair
    that does not share it.
    """
    k = len(Ec)
    if k < 3:
        return float("nan")
    C = Ec @ Ec.T
    w = np.linalg.eigvalsh(C)
    return float(w[-1] / k)


def densest_subset(Ec: np.ndarray, size: int) -> tuple[float, np.ndarray]:
    """Greedy peel to the most internally cohesive subset of a given size.

    Repeatedly drops the member with the lowest mean similarity to the rest, which is the
    standard greedy approximation to the densest-k-subgraph problem. Returns the subset's
    mean pairwise cosine and its indices, so a caller can ask what the model actually found.
    """
    k = len(Ec)
    if k <= size or size < 3:
        return float("nan"), np.arange(k)
    C = Ec @ Ec.T
    np.fill_diagonal(C, np.nan)
    keep = np.arange(k)
    while len(keep) > size:
        sub = C[np.ix_(keep, keep)]
        worst = int(np.nanargmin(np.nanmean(sub, axis=1)))
        keep = np.delete(keep, worst)
    sub = C[np.ix_(keep, keep)]
    iu = np.triu_indices(len(keep), 1)
    v = sub[iu]
    return float(np.nanmean(v)), keep


def mean_pairwise(Ec: np.ndarray) -> float:
    """The old statistic, kept so the comparison can be reported rather than asserted.

    ADMISSIBLE ONLY ON A STRICT SUBSET of the population that was centred. Centring forces the
    vectors to sum to zero, which forces the sum of all off-diagonal cosines to minus k, which
    pins this statistic to exactly -1/(k-1) -- agreeing with theory to 2.7e-06 at k of 168. Pass
    it the same set that was centred and it measures nothing: a forty percent planted bet at
    cosine 0.30 moves it by 0.0024 where the leading eigenvalue moves by 0.16.

    The failed embedding gate escaped this only because it centred on a 900-name pool and
    measured a 168-name leg. Prefer `leading_eigenvalue` or `densest_subset`, which have genuine
    range under centring. See trp-47.
    """
    k = len(Ec)
    if k < 2:
        return float("nan")
    C = Ec @ Ec.T
    iu = np.triu_indices(k, 1)
    return float(C[iu].mean())
