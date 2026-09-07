"""Hand-computable checks on the jump model, before it is pointed at real data."""

from __future__ import annotations

import numpy as np

from unstructured_momentum.analogue.jump import _assign_path, _objective, fit


def test_zero_penalty_reduces_to_nearest_centroid():
    """With lam=0 the path problem decouples and must return the argmin per row."""
    D2 = np.array([[0.0, 5.0], [4.0, 1.0], [9.0, 2.0], [0.5, 3.0]])
    assert list(_assign_path(D2, 0.0)) == [0, 1, 1, 0]


def test_large_penalty_forces_a_single_state():
    """A penalty above the total distance budget makes any switch uneconomic."""
    D2 = np.array([[0.0, 5.0], [4.0, 1.0], [9.0, 2.0], [0.5, 3.0]])
    s = _assign_path(D2, 1e6)
    assert len(set(s)) == 1
    # It must pick the cheaper of the two constant paths: col0 = 13.5, col1 = 11.0.
    assert set(s) == {1}


def test_penalty_tips_a_single_outlier_by_hand():
    """One row prefers state 1 by 3.0; two switches cost 2*lam. The tipping point is 1.5."""
    D2 = np.array([[0.0, 9.0], [3.0, 0.0], [0.0, 9.0]])
    assert list(_assign_path(D2, 1.4)) == [0, 1, 0]   # 2*1.4 = 2.8 < 3.0, worth switching
    assert list(_assign_path(D2, 1.6)) == [0, 0, 0]   # 3.2 > 3.0, not worth it


def test_objective_matches_a_hand_sum():
    X = np.array([[0.0], [0.0], [4.0]])
    mu = np.array([[0.0], [4.0]])
    s = np.array([0, 0, 1])
    # distance 0 + 0 + 0, one switch at lam=2.5
    assert _objective(X, s, mu, 2.5) == 2.5


def test_recovers_two_planted_blocks():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 0.3, size=(60, 2)),
                        rng.normal(4, 0.3, size=(60, 2))])
    f = fit(X, k=2, lam=1.0, n_init=5)
    assert f.n_switches == 1
    assert (f.states[:60] == f.states[0]).all()
    assert f.states[0] != f.states[-1]


def test_states_are_ordered_by_severity_so_labels_are_stable():
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(0, 0.2, size=(50, 2)),
                        rng.normal(6, 0.2, size=(50, 2))])
    a = fit(X, k=2, lam=1.0, n_init=5, seed=1)
    b = fit(X, k=2, lam=1.0, n_init=5, seed=99)
    assert list(a.states) == list(b.states)
    assert np.linalg.norm(a.centroids[0]) < np.linalg.norm(a.centroids[1])
