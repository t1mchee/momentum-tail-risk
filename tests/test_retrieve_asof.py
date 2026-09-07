"""The leakage fixtures for exp-064 retrieval. These are the tests that must never pass wrongly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unstructured_momentum.analogue.retrieve import COLLAPSE_MONTHS, Engine


@pytest.fixture(scope="module")
def deep():
    return Engine("deep")


def test_query_at_2019_09_cannot_retrieve_2020_03(deep):
    """The registered fixture. A neighbour from the future is the failure that ends a project."""
    got = {n.date for n in deep.query("2019-09-30", horizon_months=3)}
    assert pd.Timestamp("2020-03-31") not in got
    assert all(d < pd.Timestamp("2019-09-30") for d in got)


def test_eligibility_requires_the_outcome_window_to_have_closed(deep):
    """Stricter than as-of: a neighbour whose own forward window straddles the query leaks."""
    q = pd.Timestamp("2019-09-30")
    for h in (1, 3):
        elig = deep.eligible(q, h)
        assert (elig + pd.offsets.MonthEnd(h) < q).all()
        # The months immediately before the query are excluded BECAUSE of that rule.
        assert elig.max() < q - pd.offsets.MonthEnd(h)


def test_block_shares_sum_to_one(deep):
    for n in deep.query("2019-09-30"):
        assert abs(sum(n.block_share.values()) - 1.0) < 1e-9


def test_collapse_zone_holds(deep):
    got = [n.date for n in deep.query("2019-09-30")]
    for i, a in enumerate(got):
        for b in got[i + 1:]:
            assert abs((a - b).days) >= COLLAPSE_MONTHS * 30


def test_ranking_matches_cosine_on_the_weighted_key(deep):
    """The unit-normalised Euclidean implementation must rank identically to cosine itself."""
    q = pd.Timestamp("2019-09-30")
    cand = deep.eligible(q, 3)
    W = deep.X.to_numpy() * np.sqrt(deep.weights.to_numpy())
    qi = deep.index.get_loc(q)
    ci = deep.index.get_indexer(cand)
    cos = 1 - (W[ci] @ W[qi]) / (np.linalg.norm(W[ci], axis=1) * np.linalg.norm(W[qi]))
    u = deep._U
    euc = ((u[ci] - u[qi]) ** 2).sum(axis=1)
    assert np.corrcoef(np.argsort(cos), np.argsort(euc))[0, 1] > 0.9999
    assert np.allclose(euc, 2 * cos, atol=1e-9)


def test_random_control_is_drawn_from_the_same_eligible_pool(deep):
    rng = np.random.default_rng(0)
    q = pd.Timestamp("2019-09-30")
    picks = deep.random_neighbours(q, rng, horizon_months=3)
    assert set(picks) <= set(deep.eligible(q, 3))
