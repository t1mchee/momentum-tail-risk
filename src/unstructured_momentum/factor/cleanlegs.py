"""Momentum legs from unadjusted index prices, with corporate actions corrected.

Two biases live in the iShares price field and they are not equally tractable.

Splits are correctable. A reverse split is a step change in an unadjusted series, so a 1-for-5
reads as +400 percent momentum. This project already recorded a 1-for-20 reading as +1,900
percent and responded with a filter at five times, which catches the extreme case and misses
the common one: Spirit Realty and Rite Aid, both post-reverse-split, ranked as the two
strongest winners in the book that failed the embedding gate. Here the jump is DETECTED and
DIVIDED OUT rather than used to drop the name, because dropping loses a real constituent and
the correction is exact up to the size of the true return on the split day.

Dividends are not correctable from this data and the bias is the one that matters. Price-only
returns understate a REIT by roughly five points a year and a utility by three, against about
one for technology. At a decile cut near 28 percent that is a systematic handicap applied to
exactly the high-yield rate-sensitive names, and it is the plausible reason the 2019 winner leg
came out 27 percent technology and 2.4 percent utilities when the project's own rate-beta work
describes a bond-proxy book. A sector-median yield add-back is offered here as a SENSITIVITY,
never as a correction: it says how much the composition depends on the bias, which is the
honest thing this data can support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A single-day move beyond this in an unadjusted series is a corporate action.
JUMP = 0.45

#: Approximate 2018-19 sector median dividend yields, annual, for the SENSITIVITY only.
#: Deliberately coarse: the question is whether leg composition is sensitive to the bias,
#: not what any individual name yielded.
SECTOR_YIELD = {
    "Real Estate": 0.040, "Utilities": 0.033, "Consumer Staples": 0.029,
    "Energy": 0.035, "Communication": 0.021, "Financials": 0.023,
    "Materials": 0.021, "Industrials": 0.019, "Health Care": 0.014,
    "Consumer Discretionary": 0.013, "Information Technology": 0.011,
}
DEFAULT_YIELD = 0.018


def correct_splits(panel: pd.DataFrame, jump: float = JUMP,
                   *, assert_daily: bool = True) -> tuple[pd.DataFrame, dict]:
    """DEPRECATED for new work -- prefer factor.splits.adjust, which detects corporate
    actions from share counts rather than from return magnitude and is frequency-agnostic.

    `jump` is a DAILY calibration. On monthly data a 45% move is often a real crisis move,
    and this function erased BAC's genuine -53% (Jan-2009) and +73% (Mar-2009). Measured on
    the IWV panel it classified 3,056 names as split-affected against 690 genuine actions,
    destroying 94% of what it touched. The guard below refuses mixed or monthly panels;
    pass assert_daily=False only with a stated reason.
    """
    if assert_daily:
        from ..frequency import require
        require(panel.index, expect="daily", instrument="cleanlegs.correct_splits")
    """Divide out detected corporate-action jumps instead of discarding the name.

    Walking forward, any day whose gross return leaves the plausible band is treated as a
    ratio change and every later price is rescaled by it. The correction is exact up to the
    true return on that day, which is a far smaller error than either keeping the jump or
    losing the constituent.
    """
    out = panel.copy()
    hits = {}
    g = panel / panel.shift(1)
    for c in panel.columns:
        r = g[c]
        bad = r.index[(r > 1 + jump) | (r < 1 - jump)]
        bad = [d for d in bad if np.isfinite(r.get(d, np.nan))]
        if not bad:
            continue
        factor = 1.0
        adj = out[c].copy()
        for d in bad:
            factor *= float(r[d])
            adj.loc[d:] = panel[c].loc[d:] / factor
        out[c] = adj
        hits[c] = [(str(pd.Timestamp(d).date()), round(float(r[d]), 2)) for d in bad]
    return out, hits


def momentum(panel: pd.DataFrame, asof: pd.Timestamp, *, months_back: int = 12,
             skip: int = 1, sectors: pd.Series | None = None,
             add_dividends: bool = False, min_obs: int = 60,
             fund: str | None = "IWV") -> pd.Series:
    """12-1 momentum on a split-corrected panel, optionally with a yield add-back.

    `min_obs` is a DAILY calibration: 60 observations is ~3 months of trading days, and
    five years of monthly ones. On the monthly panel it silently returned empty for every
    date before 2011 while the underlying ticker intersection was ~2,770 names -- the
    failure looked like missing data and was a frequency assumption. Callers working at
    monthly frequency must set it explicitly.
    """
    dates = panel.index[panel.index <= asof]
    if len(dates) < min_obs:
        return pd.Series(dtype=float)
    now = dates[-1]
    b_i = panel.index[panel.index <= now - pd.DateOffset(months=skip)]
    a_i = panel.index[panel.index <= now - pd.DateOffset(months=months_back)]
    if not len(b_i) or not len(a_i):
        return pd.Series(dtype=float)
    a, b = panel.loc[a_i[-1]], panel.loc[b_i[-1]]
    common = a.dropna().index.intersection(b.dropna().index)

    # POINT-IN-TIME INDEX MEMBERSHIP, applied here rather than by each caller. Requiring a
    # price at both ends of the formation window already drops names deleted before the
    # window closes, which is why this construction was less exposed than the other one --
    # but it does not drop a name deleted between the window close and the as-of date, and
    # at June reconstitution that is up to 114 names, a loser-leg Jaccard of 0.619. Set
    # fund=None only to reproduce a pre-revision number deliberately.
    if fund:
        from ..data import ishares
        members = ishares.members_at(asof, fund)
        if members:
            common = common.intersection(pd.Index(sorted(members)))

    mom = (b[common] / a[common]) - 1
    if add_dividends and sectors is not None:
        yrs = (b_i[-1] - a_i[-1]).days / 365.25
        y = sectors.reindex(mom.index).map(SECTOR_YIELD).fillna(DEFAULT_YIELD)
        mom = mom + y * yrs
    return mom.dropna()


def legs(mom: pd.Series, decile: float = 0.10) -> tuple[list, list]:
    k = max(20, int(len(mom) * decile))
    return list(mom.nlargest(k).index), list(mom.nsmallest(k).index)


def composition(names: list[str], sectors: pd.Series) -> pd.Series:
    s = sectors.reindex(names).dropna()
    return (s.value_counts() / len(s)).sort_values(ascending=False)
