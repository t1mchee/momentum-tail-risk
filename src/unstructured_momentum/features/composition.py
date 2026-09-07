"""The momentum book's own composition, as a point-in-time state.

Why this exists
---------------
The book re-forms every month and reallocates about a third of its sector tilt when
it does. A trailing 250-day beta is therefore describing a portfolio that overlaps
the one you hold by roughly a fifth, and at the extremes has inverted: the estimator
reports the opposite bet. That is not noise, it is a cadence mismatch, and it is the
measured reason every return-based estimate of this book's macro exposure fails in
time (Daniel-Moskowitz's ex-ante hedge, Grundy-Martin's feasible hedge, and our own
rolling-beta test all land in the same place from different directions).

Composition is the one description of the book that is exact, available the day the
book forms, and carries no inferential claim at all. It is arithmetic over holdings.
Everything else in the system -- the severity chain's nowcast, analogue retrieval,
the naming layer -- attaches to this state, so it is deliberately the least clever
module in the repo.

What this module does NOT claim
-------------------------------
It does not forecast. The book's forward realised rate beta has a twelve-month
self-correlation of +0.017, so it is not a stable quantity and nothing predicts it;
composition included (tested, and the apparent -0.476 was a 2020-22 artifact that
fell to -0.085 once those years were dropped). Composition describes the book you
have. That is its whole job.

Point-in-time
-------------
iShares publishes a holdings file for session T on the morning of T+1. `LAG_DAYS`
encodes that: a tilt computed for date D may only read snapshots with
``as_of <= D - LAG_DAYS``. Using ``as_of <= D`` would silently grant same-session
knowledge of the close-of-day book.
"""

from __future__ import annotations

import glob
from functools import lru_cache

import numpy as np
import pandas as pd

#: iShares publishes session T's holdings on the morning of T+1.
LAG_DAYS = 1

#: A leg needs this many matched names before its sector mix means anything. Below it
#: the tilt is a statement about the crosswalk, not about the book.
MIN_MATCHED = 50

PANEL = "data/raw/ishares/IWV/panel/IWV_*.parquet"


@lru_cache(maxsize=32)
def _year(y: int) -> pd.DataFrame | None:
    hits = [f for f in sorted(glob.glob(PANEL)) if f.endswith(f"IWV_{y}.parquet")]
    if not hits:
        return None
    p = pd.read_parquet(hits[0], columns=["as_of", "ticker", "sector", "weight_pct"])
    p["as_of"] = pd.to_datetime(p.as_of)
    return p


def leg_mix(date: pd.Timestamp, members: set[str]) -> pd.Series | None:
    """Sector weights of one leg, normalised, from the latest snapshot available at `date`.

    Returns None rather than a partial answer when coverage is too thin -- a tilt
    computed off twenty matched names would be a crosswalk artifact wearing a
    portfolio's clothes.
    """
    cutoff = date - pd.Timedelta(days=LAG_DAYS)
    frames = [f for f in (_year(cutoff.year), _year(cutoff.year - 1)) if f is not None]
    if not frames:
        return None
    p = pd.concat(frames, ignore_index=True)
    snap = p[p.as_of <= cutoff]
    if snap.empty:
        return None
    snap = snap[snap.as_of == snap.as_of.max()]
    # a ticker can appear twice in one snapshot (dual share lines); keep the larger
    snap = snap.groupby(["ticker", "sector"], as_index=False).weight_pct.max()
    m = snap[snap.ticker.isin(members)]
    if len(m) < MIN_MATCHED:
        return None
    w = m.groupby("sector").weight_pct.sum()
    tot = w.sum()
    return w / tot if tot > 0 else None


def book_tilt(date: pd.Timestamp, winners: set[str], losers: set[str]) -> pd.Series | None:
    """Net sector tilt: winner-leg mix minus loser-leg mix.

    Signed on purpose. The book is a long/short bet, and a sector held on both legs
    is not a bet on that sector -- it is a bet inside it, which this statistic
    correctly reports as near zero.
    """
    w, l = leg_mix(date, winners), leg_mix(date, losers)
    if w is None or l is None:
        return None
    idx = sorted(set(w.index) | set(l.index))
    return w.reindex(idx).fillna(0) - l.reindex(idx).fillna(0)


def tilt_panel(legs: dict, dates=None) -> pd.DataFrame:
    """Tilt for every book date, as a dense sector x date panel."""
    rows = {}
    for d in sorted(dates if dates is not None else legs.keys()):
        t = book_tilt(d, set(legs[d]["winners"]), set(legs[d]["losers"]))
        if t is not None:
            rows[d] = t
    return pd.DataFrame(rows).T.fillna(0).sort_index()


def turnover(T: pd.DataFrame, periods: int = 1) -> pd.Series:
    """Fraction of the book's tilt reallocated over `periods` months.

    L1/2 so that moving one unit of weight from sector A to sector B counts once,
    not twice.
    """
    return T.diff(periods).abs().sum(axis=1) / 2


def overlap(T: pd.DataFrame, periods: int = 12) -> pd.Series:
    """How much of today's tilt was present `periods` months ago.

    This is the number that indicts the trailing beta: it is what a 250-day
    estimation window is actually averaging over. It can go NEGATIVE, which means
    the book has not merely drifted but inverted -- the estimator is describing the
    opposite bet to the one held.
    """
    return 1 - turnover(T, periods)


def distance(T: pd.DataFrame) -> pd.DataFrame:
    """Pairwise L1/2 distance between book tilts. 0 = identical, 1 = disjoint."""
    A = T.values
    D = np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2) / 2
    return pd.DataFrame(D, index=T.index, columns=T.index)
