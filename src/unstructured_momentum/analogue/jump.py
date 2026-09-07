"""exp-065 Part A — the statistical jump model.

Temporal clustering with an explicit penalty on state switches. The objective is

    min over s and mu:  sum_t || x_t - mu_{s_t} ||^2  +  lambda * sum_t 1[s_t != s_{t-1}]

which is k-means with a cost for changing your mind. The penalty is what separates this
from clustering the rows independently and reading persistence into the result afterwards:
persistence is IN the objective, so a state that lasts one month has to earn its keep
against the cost of two switches.

Fitted by coordinate descent, alternating two exact steps:

  1. state path given centroids -- dynamic programming over the K states, which is exact
     because the switch cost is Markov in the state index;
  2. centroids given the path -- the within-state mean, which is the exact minimiser of a
     sum of squared distances.

Each step cannot increase the objective and the objective is bounded below, so the descent
terminates. It terminates at a LOCAL optimum, which is why the fit restarts from several
k-means++ initialisations and keeps the best objective rather than trusting one run.

Implemented here rather than imported: the published packages for this model are
unmaintained, the algorithm is two screens of code, and a hand-computable toy case
(tests/test_jump.py) is worth more than a dependency whose failure mode nobody can inspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fit:
    states: np.ndarray        # (T,) integer state per observation
    centroids: np.ndarray     # (K, D)
    objective: float
    n_switches: int
    lam: float
    n_iter: int


def _assign_path(D2: np.ndarray, lam: float) -> np.ndarray:
    """Exact minimum-cost state path by dynamic programming.

    `D2[t, k]` is the squared distance from observation t to centroid k. The recursion
    carries, for each state, the best total cost of any path ending in that state:

        V[t, k] = D2[t, k] + min_j ( V[t-1, j] + lam * 1[j != k] )

    The inner minimisation is over K terms but only needs the best and second-best of
    V[t-1], because the penalty takes one of two values -- which is what keeps this O(TK)
    rather than O(TK^2), and matters at T over a thousand with many restarts.
    """
    T, K = D2.shape
    V = np.empty((T, K))
    back = np.empty((T, K), dtype=np.int64)
    V[0] = D2[0]
    back[0] = -1
    for t in range(1, T):
        prev = V[t - 1]
        best = int(np.argmin(prev))
        best_val = prev[best]
        # Second best, for the case where staying put is cheaper than the global best.
        second_val = np.min(np.delete(prev, best)) if K > 1 else np.inf
        for k in range(K):
            stay = prev[k]
            # Cheapest predecessor that is NOT k, plus the switch cost.
            move_val = (best_val if best != k else second_val) + lam
            if stay <= move_val:
                V[t, k] = D2[t, k] + stay
                back[t, k] = k
            else:
                V[t, k] = D2[t, k] + move_val
                back[t, k] = best if best != k else int(
                    np.argmin(np.where(np.arange(K) == k, np.inf, prev)))
    s = np.empty(T, dtype=np.int64)
    s[-1] = int(np.argmin(V[-1]))
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s


def _objective(X: np.ndarray, s: np.ndarray, mu: np.ndarray, lam: float) -> float:
    fit = ((X - mu[s]) ** 2).sum()
    return float(fit + lam * int((s[1:] != s[:-1]).sum()))


def fit(X: np.ndarray, k: int = 3, lam: float = 1.0, *, n_init: int = 10,
        max_iter: int = 100, seed: int = 20260830) -> Fit:
    """Coordinate descent from several k-means++ starts; best objective wins."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    best: Fit | None = None

    for _ in range(n_init):
        mu = _kmeanspp(X, k, rng)
        s = np.zeros(len(X), dtype=np.int64)
        prev_obj = np.inf
        for it in range(max_iter):
            D2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
            s = _assign_path(D2, lam)
            for j in range(k):
                m = s == j
                if m.any():
                    mu[j] = X[m].mean(axis=0)
                # An empty state keeps its centroid. Re-seeding it here would let the fit
                # wander between restarts and makes the descent non-monotone.
            obj = _objective(X, s, mu, lam)
            if prev_obj - obj < 1e-10:
                break
            prev_obj = obj
        obj = _objective(X, s, mu, lam)
        cand = Fit(states=s, centroids=mu.copy(), objective=obj,
                   n_switches=int((s[1:] != s[:-1]).sum()), lam=lam, n_iter=it + 1)
        if best is None or cand.objective < best.objective:
            best = cand

    assert best is not None
    return _relabel_by_severity(X, best)


def _kmeanspp(X: np.ndarray, k: int, rng) -> np.ndarray:
    mu = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - np.array(mu)[None]) ** 2).sum(axis=2), axis=1)
        tot = d2.sum()
        p = d2 / tot if tot > 0 else np.full(len(X), 1 / len(X))
        mu.append(X[rng.choice(len(X), p=p)])
    return np.array(mu, dtype=float)


def _relabel_by_severity(X: np.ndarray, f: Fit) -> Fit:
    """State 0 is the calmest centroid, K-1 the most extreme.

    Cluster labels are arbitrary and a run-to-run permutation would make every downstream
    comparison meaningless. Ordering by centroid norm makes the label stable and readable:
    higher means further from the panel's centre.
    """
    order = np.argsort(np.linalg.norm(f.centroids, axis=1))
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    return Fit(states=remap[f.states], centroids=f.centroids[order],
               objective=f.objective, n_switches=f.n_switches, lam=f.lam, n_iter=f.n_iter)


def switch_dates(states: np.ndarray, index) -> list:
    """Index positions where the state changes -- the model's own break dates."""
    return [index[i + 1] for i in np.flatnonzero(states[1:] != states[:-1])]
