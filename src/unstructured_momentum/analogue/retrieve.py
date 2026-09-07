"""exp-064 Stage 2 — numeric-only retrieval over the certified state vector.

The baseline every later layer is measured against. Registered on RETRIEVAL QUALITY and
never against a severity forecaster: this project already established that vol-scaled
climatology beats the analogue ensemble as a severity estimate, so the question here is the
different one of whether SIMILARITY does work that RECENCY does not.

Two arms, because the panel is two panels
------------------------------------------
Stage 1 established that conferred loadings cannot be estimated before mid-2013, which left
a century-deep three-block archive and a six-year seven-block one. Retrieval runs separately
on each and nothing is pooled. A single key with coverage-normalised distances would compare
a three-block state description against an eight-block one through a rescaling, which is the
span confound that has already killed three results in this project.

Cosine, implemented so the explanation is exact
------------------------------------------------
The registered metric is cosine. Cosine has no additive decomposition, and the output
contract owes the reader a per-block share of the distance. Both are satisfied at once by
normalising each state to unit length under the block weights and then taking squared
Euclidean distance: for unit vectors d_euclid^2 = 2(1 - cos), so the RANKING is cosine's
exactly, while the squared distance decomposes across blocks additively and the shares are
exact rather than attributed.

Eligibility is stricter than the usual as-of convention
--------------------------------------------------------
A neighbour must precede the query AND have its own forward outcome window closed before the
query date. Condition two closes a leakage channel most of the analogue literature does not
name, and it is inherited from this project's existing ruling rather than re-decided here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SV = Path("data/processed/state_vector.parquet")
SPEC = Path("data/processed/state_vector_spec.json")

#: Registered in exp-064 and not swept. A swept k scored on the same battery that selects it
#: would be selection on the outcome.
K = 5

#: Two neighbours inside one quarter are one episode; the later-ranked one is dropped.
#: The monthly analogue of the existing COLLAPSE=63 trading days.
COLLAPSE_MONTHS = 3

#: A neighbour beyond this share of the median random-pair distance is annotated
#: `is_analogue=False` and KEPT. "No analogue exists" is an answer, and suppressing the row
#: hides that the system had nothing to say.
FLOOR_FRAC = 0.50

ARMS: dict[str, list[str]] = {
    "deep": ["dispersion", "volregime", "mechanism"],
    # The augmented arm. Same three century-deep blocks plus newspaper-based narrative about
    # the economy, which is the first text source here that reaches as far back as the returns
    # do. Registered as an ADDITION so every comparison against `deep` is a measured delta.
    "deep_text": ["dispersion", "volregime", "mechanism", "textmacro"],
    "rich": ["tilt", "comovement", "dispersion", "valuation", "volregime", "mechanism",
             "options"],
}


@dataclass(frozen=True)
class Neighbour:
    date: pd.Timestamp
    distance: float
    rank: int
    block_share: dict[str, float]      # share of the remaining DISTANCE -- dissimilarity
    block_closeness: dict[str, float]  # per-block distance / random-pair median -- similarity
    is_analogue: bool
    cross_regime: bool | None   # filled by Stage 3; frozen in the contract now


class Engine:
    """Retrieval over one arm of the state vector."""

    def __init__(self, arm: str = "deep", *, sv: pd.DataFrame | None = None,
                 spec: dict | None = None) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {list(ARMS)}")
        self.arm = arm
        self.spec = spec if spec is not None else json.loads(SPEC.read_text())
        d = sv if sv is not None else pd.read_parquet(SV)

        self.blocks = {b: self.spec["blocks"][b] for b in ARMS[arm]}
        self.columns = [c for cols in self.blocks.values() for c in cols]

        # Equal weight per block within THIS arm, split evenly across the block's columns.
        # Re-derived for the arm rather than inherited from the eight-block spec: an arm that
        # kept the eight-block weights would silently leave a fraction of the weight on
        # blocks it does not carry.
        self.weights = pd.Series(
            {c: 1.0 / (len(self.blocks) * len(cols))
             for b, cols in self.blocks.items() for c in cols}
        )[self.columns]

        X = d[[f"z_{c}" for c in self.columns]].dropna()
        X.columns = self.columns
        self.X = X
        self.index = X.index

        # Weighted, then unit-normalised: squared Euclidean on these rows ranks exactly as
        # cosine on the weighted key, and decomposes additively across blocks.
        W = X.to_numpy() * np.sqrt(self.weights.to_numpy())
        norm = np.linalg.norm(W, axis=1, keepdims=True)
        self._U = np.divide(W, norm, out=np.zeros_like(W), where=norm > 0)
        self._col_block = np.array(
            [b for b, cols in self.blocks.items() for _ in cols]
        )
        # Median random-pair distance, for the floor. Computed once on the arm's own rows.
        self._floor = float(np.median(self._pair_sample()))
        self._block_ref = self._block_reference()

    def _block_reference(self, n: int = 20000, seed: int = 20260830) -> dict:
        """Median per-block squared distance over random pairs.

        Needed because the raw per-block share answers "where does the remaining distance
        come from", which is a DISSIMILARITY statement. A reader asking why two months matched
        wants the opposite: which blocks are closer than two arbitrary months would be. That
        is only expressible against a reference, so the reference is computed once here.
        """
        rng = np.random.default_rng(seed)
        m = len(self._U)
        i, j = rng.integers(0, m, n), rng.integers(0, m, n)
        ok = i != j
        sq = (self._U[i[ok]] - self._U[j[ok]]) ** 2
        return {b: float(np.median(sq[:, self._col_block == b].sum(axis=1)))
                for b in self.blocks}

    def _pair_sample(self, n: int = 20000, seed: int = 20260830) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m = len(self._U)
        i, j = rng.integers(0, m, n), rng.integers(0, m, n)
        ok = i != j
        return np.sqrt(((self._U[i[ok]] - self._U[j[ok]]) ** 2).sum(axis=1))

    # -- eligibility -----------------------------------------------------------------

    def eligible(self, query: pd.Timestamp, horizon_months: int) -> pd.DatetimeIndex:
        """Precedes the query AND its own outcome window closed before the query.

        The second condition is what makes this stricter than the usual convention: a
        neighbour at month m carries a forward window running to m + horizon, and if that
        window has not closed by the query date then reading its outcome is reading the
        future of the query.
        """
        return self.index[self.index + pd.offsets.MonthEnd(horizon_months) < query]

    # -- retrieval -------------------------------------------------------------------

    def query(self, date, *, k: int = K, horizon_months: int = 3,
              pool: pd.DatetimeIndex | None = None) -> list[Neighbour]:
        q = pd.Timestamp(date) + pd.offsets.MonthEnd(0)
        if q not in self.index:
            raise KeyError(f"{q.date()} has no complete state row in arm {self.arm!r}")
        cand = self.eligible(q, horizon_months) if pool is None else pool
        if len(cand) == 0:
            return []

        qi = self.index.get_loc(q)
        ci = self.index.get_indexer(cand)
        diff = self._U[ci] - self._U[qi]          # (n_cand, n_cols)
        sq = diff ** 2
        d2 = sq.sum(axis=1)

        # Ties broken by recency: order by distance, then by date descending. lexsort takes
        # the LAST key as primary, so distance goes last.
        order = np.lexsort((-cand.asi8, d2))

        out, taken = [], []
        for pos in order:
            when = cand[pos]
            # Collapse zone: two neighbours inside one quarter are one episode.
            if any(abs((when - t).days) < COLLAPSE_MONTHS * 30 for t in taken):
                continue
            taken.append(when)
            tot = d2[pos]
            share = {b: float(sq[pos][self._col_block == b].sum() / tot) if tot > 0 else 0.0
                     for b in self.blocks}
            # Below 1.0 means this block is CLOSER than two arbitrary months would be, which
            # is the quantity that answers "why did these match". A block can carry most of
            # the remaining distance and still be the reason for the match, if every other
            # block is closer still -- which is why both numbers are emitted and only this one
            # is labelled as the reason.
            closeness = {b: float(sq[pos][self._col_block == b].sum()
                                  / self._block_ref[b]) if self._block_ref[b] > 0 else np.nan
                         for b in self.blocks}
            out.append(Neighbour(
                date=when, distance=float(np.sqrt(tot)), rank=len(out) + 1,
                block_share=share, block_closeness=closeness,
                is_analogue=bool(np.sqrt(tot) <= FLOOR_FRAC * self._floor),
                cross_regime=None,
            ))
            if len(out) == k:
                break
        return out

    def random_neighbours(self, date, rng, *, k: int = K,
                          horizon_months: int = 3) -> pd.DatetimeIndex:
        """k random dates from the SAME eligible pool -- the control for tests one and two.

        Drawn from the eligible pool, not from the whole index, so the comparison isolates
        feature-based selection and does not accidentally reward the as-of filter.
        """
        q = pd.Timestamp(date) + pd.offsets.MonthEnd(0)
        cand = self.eligible(q, horizon_months)
        if len(cand) < k:
            return pd.DatetimeIndex([])
        return cand[rng.choice(len(cand), size=k, replace=False)]
