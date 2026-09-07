"""Ownership overlap: how far the same managers hold a basket together.

The distinction this measures
-----------------------------
Popularity and crowding are different quantities and the difference is the whole point. A
stock is held by many institutions because it is large and index-tracked; that is
popularity, and it says nothing about fragility. Crowding is the stronger condition that an
*overlapping* set of managers holds the same basket, so that one manager's forced exit moves
prices against the others who are still in it. A basket can be held by thousands of
institutions with almost no overlap, and it is then not crowded at all.

So the statistic is pairwise, and it is a LIFT rather than a raw overlap. For every pair of
names in a leg, take the number of filers holding both, divided by the number that would
hold both if holders were drawn independently. That is 1.0 when ownership is unrelated and
rises only when the same managers genuinely hold the names together.

The lift, rather than the Jaccard, because the controls demanded it
------------------------------------------------------------------
The first version used raw Jaccard and a negative control killed it. Two sets drawn
independently from N filers already share about ``|A||B|/N`` members, so Jaccard climbs with
breadth on its own. Raising winner breadth from 60 filers to 220, holders drawn
independently every time and no common constituency anywhere, produced a difference of
+0.0280 -- against +0.0342 for genuinely crowding 60 percent of the leg into one clique. The
measure was reporting popularity and calling it crowding, which is the precise error this
module exists to avoid. Under the lift the same negative control reads +0.002 at a
probability of 0.673, while a clique of only 10 percent is detected at 0.000.

Why the permutation runs within size strata
-------------------------------------------
Institutional breadth rises steeply with market capitalisation, and a momentum winner leg is
mechanically larger than a loser leg because it has just risen. The null shuffles the winner
and loser labels *within* size deciles, holding leg sizes fixed, so the question it answers
is whether the legs differ beyond what their size distributions already imply. Strata are
ranked with ties broken at random: breaking them by position put every winner in the low
strata and every loser in the high ones, which made the shuffle a no-op and returned a
probability of exactly 1.000 for every input, including differences it should have called
significant.

What 13F cannot see
-------------------
Long positions only. The loser leg here is the set of stocks institutions are long that have
fallen, which is not the set that momentum investors are short. So an asymmetry between the
legs is evidence about the long side and is silent on short crowding, which needs FINRA
short interest instead. Nothing in this module can repair that; it is a property of the
disclosure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A name held by fewer filers than this has an overlap lift that is mostly sampling noise.
MIN_FILERS = 5

#: Pairs sampled when a leg is large enough that exhaustive pairing is wasteful. 290 names
#: is only 42,000 pairs, so this rarely binds, but it bounds the cost for wider baskets.
MAX_PAIRS = 200_000


def holder_sets(
    panel: pd.DataFrame, tickers: dict[str, str], *, min_filers: int = MIN_FILERS
) -> dict[str, set]:
    """Filer sets per ticker, keyed by the crosswalk, dropping thinly held names."""
    sub = panel[panel["cusip"].isin(tickers)].copy()
    sub["ticker"] = sub["cusip"].map(tickers)
    out = {}
    for tkr, g in sub.groupby("ticker"):
        ciks = set(g["cik"].unique())
        if len(ciks) >= min_filers:
            out[tkr] = ciks
    return out


def _matrix(sets: dict[str, set], names: list[str]):
    """Names-by-filers sparse indicator matrix, and the filer index it was built on."""
    from scipy import sparse

    present = [n for n in names if n in sets]
    if len(present) < 2:
        return None, None
    filers = sorted(set().union(*(sets[n] for n in present)))
    pos = {f: i for i, f in enumerate(filers)}
    rows, cols = [], []
    for r, n in enumerate(present):
        for f in sets[n]:
            rows.append(r)
            cols.append(pos[f])
    m = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(present), len(filers)),
    )
    return m, present


def _lift_from_matrix(m, universe: int) -> tuple[float, int]:
    """Mean pairwise co-ownership lift for the rows of a prebuilt indicator matrix."""
    k = m.shape[0]
    if k < 2 or universe <= 0:
        return float("nan"), 0
    co = (m @ m.T).toarray()
    sizes = np.asarray(m.sum(axis=1)).ravel()
    iu = np.triu_indices(k, 1)
    e = (np.outer(sizes, sizes) / universe)[iu]
    ok = e > 0
    if not ok.any():
        return float("nan"), 0
    return float((co[iu][ok] / e[ok]).mean()), int(ok.sum())


def mean_pairwise_jaccard(
    sets: dict[str, set],
    names: list[str],
    *,
    universe: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, int]:
    """Mean overlap LIFT over every pair of the given names, and the pair count.

    Not raw Jaccard. Raw Jaccard rises with breadth on its own: two sets drawn independently
    from a universe of N filers share about ``|A||B|/N`` members, so a widely held basket
    scores higher than a narrowly held one with no common constituency whatever. A negative
    control made this concrete -- raising winner breadth from 60 filers to 220, with holders
    drawn independently every time, produced a difference of +0.0280, which is most of the
    +0.0342 produced by genuinely crowding 60 percent of the leg into one clique. The
    statistic was reporting popularity and calling it crowding.

    So each pair is scored as observed co-ownership over its independence expectation. That
    is 1.0 when holders are unrelated, regardless of how many of them there are, and rises
    only when the same managers actually hold the names together.

    Computed as one sparse product rather than a Python loop over pairs. The loop version
    built a list of every pair before sampling it down, on every permutation draw, which for
    a 1,500-name leg is 1.1 million tuples constructed two thousand times per date. It was
    still running after 33 minutes. This is exact rather than sampled, and finishes in
    milliseconds.
    """
    m, present = _matrix(sets, names)
    if m is None:
        return float("nan"), 0
    return _lift_from_matrix(m, universe or m.shape[1])


def leg_overlap(
    sets: dict[str, set],
    winners: list[str],
    losers: list[str],
    size: pd.Series,
    *,
    n_perm: int = 1000,
    n_strata: int = 10,
    seed: int = 0,
) -> dict:
    """Winner and loser overlap, and a permutation null that holds size fixed.

    ``size`` is any capitalisation proxy indexed by ticker; deciles of it define the strata
    within which labels are shuffled.
    """
    rng = np.random.default_rng(seed)
    w = [t for t in winners if t in sets]
    l = [t for t in losers if t in sets]
    if len(w) < 2 or len(l) < 2:
        return {"error": "a leg has fewer than two covered names", "n_winners": len(w), "n_losers": len(l)}

    universe = len(set().union(*(sets[t] for t in w + l)))
    # Build the indicator matrix ONCE for the whole pool and take row slices per draw.
    # Rebuilding it inside the loop was the second bottleneck found here: for a 1,500-name
    # pool it is roughly 760,000 Python-level appends, repeated two thousand times per date,
    # and it kept the run at minutes per date even after the pair enumeration was vectorised.
    pool = w + l
    pool_m, pool_names = _matrix(sets, pool)
    if pool_m is None:
        return {"error": "pool too small to build an indicator matrix"}
    row_of = {n: i for i, n in enumerate(pool_names)}
    w_rows = np.array([row_of[t] for t in w if t in row_of])
    l_rows = np.array([row_of[t] for t in l if t in row_of])
    ow, pw = _lift_from_matrix(pool_m[w_rows], universe)
    ol, pl = _lift_from_matrix(pool_m[l_rows], universe)
    observed = ow - ol
    s = size.reindex(pool).astype(float)
    # Ranked with ties broken at random, NOT by position. `rank(method="first")` breaks ties
    # in index order, and the pool is built winners-then-losers, so a size proxy with many
    # ties produced strata that were entirely winner or entirely loser. Shuffling labels
    # inside them was a no-op, and the permutation returned exactly 1.000 for every input
    # including a difference it should have called significant.
    jitter = pd.Series(rng.random(len(pool)), index=pool)
    strata = pd.qcut(
        (s.rank(method="average") + jitter).rank(method="first"),
        min(n_strata, max(2, len(pool) // 20)),
        labels=False,
    )
    labels = pd.Series([1] * len(w) + [0] * len(l), index=pool)

    draws = []
    for _ in range(n_perm):
        shuffled = labels.copy()
        for _, members in labels.groupby(strata):
            vals = labels.reindex(members.index).to_numpy().copy()
            rng.shuffle(vals)
            shuffled.loc[members.index] = vals
        lab = shuffled.reindex(pool_names).to_numpy()
        a, _ = _lift_from_matrix(pool_m[np.flatnonzero(lab == 1)], universe)
        b, _ = _lift_from_matrix(pool_m[np.flatnonzero(lab == 0)], universe)
        draws.append(a - b)

    draws = np.array([d for d in draws if d == d])
    p = float((draws >= observed).mean()) if len(draws) else float("nan")
    return {
        "winner_overlap": ow,
        "loser_overlap": ol,
        "difference": observed,
        "winner_pairs": pw,
        "loser_pairs": pl,
        "n_winners": len(w),
        "n_losers": len(l),
        "perm_p": p,
        "perm_mean": float(draws.mean()) if len(draws) else float("nan"),
        "perm_sd": float(draws.std()) if len(draws) else float("nan"),
        "n_perm_used": int(len(draws)),
    }
