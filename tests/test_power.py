"""Tests for the planted-effect harness.

Every one of these encodes a bug the harness's own self-test caught while it was being built.
A harness that certifies other instruments has to be certified itself, and it was wrong four
times before it was right.
"""

from __future__ import annotations

import numpy as np
import pytest

from unstructured_momentum.features.concentration import mean_pairwise
from unstructured_momentum.validation import power as P

POOL = "data/gate1_E.npy"


@pytest.fixture(scope="module")
def pool():
    p = np.load(POOL)
    if len(p) < 300:
        pytest.skip("gate1_E.npy missing or too small")
    return p


def test_plant_delivers_its_target_cosine(pool):
    """The plant's parameter is the achieved cosine, not an opaque mixing weight.

    A first version solved the weight analytically in RAW space. Real embeddings are strongly
    anisotropic -- any two filings sit near 0.99 raw cosine -- so a requested 0.05 delivered
    0.655 once centring removed the common direction. The weight is now solved numerically in
    the centred space the statistics use.
    """
    rng = np.random.default_rng(0)
    for target in (0.05, 0.136, 0.30):
        got = []
        for _ in range(8):
            b = pool[rng.choice(len(pool), 168, replace=False)]
            Y, idx = P.plant_partial_bet(b, 0.4, target, rng)
            C = P._normalise(Y)[idx]
            G = C @ C.T
            iu = np.triu_indices(len(C), 1)
            got.append(G[iu].mean())
        assert abs(np.mean(got) - target) < 0.02, (
            f"target {target} delivered {np.mean(got):.3f}")


def test_null_has_non_degenerate_spread(pool):
    """The null must come from basket SELECTION, not row scrambling.

    Mean pairwise cosine on a centred, unit-normalised matrix is algebraically about
    -1/(k-1) whatever the data. A scramble null therefore had ZERO variance and produced
    z-scores in the hundreds. The spread that matters comes from drawing different real firms.
    """
    null = P.null_distribution(mean_pairwise, pool, 168, n_rep=120)
    assert null.std() > 1e-4, f"null spread {null.std():.2e} is degenerate"


def test_planting_raises_the_statistic_not_lowers_it(pool):
    """Order of operations: centre on the POOL, then index the basket out.

    Centring on the BASKET forces it to sum to zero, so a shared direction in a minority of
    rows pushes the majority the other way and the cross-pairs cancel the effect -- planting
    then made the statistic go DOWN as cohesion rose. The failed gate centred on its 900-name
    pool, and reproducing its power requires the same order.
    """
    rng = np.random.default_rng(5)
    base, planted = [], []
    for _ in range(10):
        sel = rng.choice(len(pool), 168, replace=False)
        base.append(mean_pairwise(P._normalise(pool)[sel]))
        Y = pool.astype(np.float64).copy()
        sub, local = P.plant_partial_bet(pool[sel], 0.4, 0.25, rng)
        Y[sel] = sub
        planted.append(mean_pairwise(P._normalise(Y)[sel]))
    assert np.mean(planted) > np.mean(base), (
        f"planting lowered the statistic: {np.mean(planted):.5f} vs {np.mean(base):.5f}")


def test_false_alarm_rate_is_near_nominal(pool):
    far = P.false_alarm_rate(mean_pairwise, pool, 168, n_rep=200, alpha=0.05)
    assert far < 0.15, f"false-alarm rate {far:.3f} far above nominal 0.05"


def test_certificate_reports_a_region_not_a_point(pool):
    """A single planted shape reproduces the original error one level up."""
    c = P.certify(mean_pairwise, pool, 168, name="mean_pairwise", n_rep=25,
                  fractions=(0.2, 0.4, 0.6), cohesions=(0.05, 0.10, 0.25))
    assert c.plant_layer == "vector"
    assert "Null:" in c.statement() and "False-alarm" in c.statement()
    assert c.mde["detectable_anywhere"], "harness detects nothing on a real pool"
    # The boundary must be monotone: a bigger sub-bet needs no more cohesion than a smaller one.
    b = dict(c.mde["boundary"])
    fs = sorted(b)
    assert all(b[fs[i]] >= b[fs[i + 1]] for i in range(len(fs) - 1)), f"non-monotone: {b}"
