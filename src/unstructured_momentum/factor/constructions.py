"""Momentum constructed many ways, with the legs kept separate.

Why several constructions rather than one
-----------------------------------------
The first substantive finding of this project was that *which episodes count as events
depends materially on how momentum is built*. Modern-era worst-10-day percentiles::

                            Aug 2007   Sept 2019   Nov 2020   Jan 2021
    FF WML (VW, official)     -4.53%     -11.68%    -15.66%     -2.68%
    WML large-cap (VW)        -8.37%     -11.49%    -21.33%     -6.88%
    WML small-cap (EW)        -3.38%      -9.44%    -11.84%     -1.43%

Aug 2007 is roughly twice as damaging in large-cap momentum as in the headline factor,
and Jan 2021 looks like a genuine (if modest) large-cap event while being invisible
equal-weighted. A monitor calibrated on one construction and deployed against a book
built on another is measuring the wrong thing. So the exploration keeps them all.

Why the legs are separated
--------------------------
Daniel & Moskowitz's mechanism is a statement about the *short* leg: after a market
decline the past-loser decile becomes high-beta and option-like -- loser-decile beta can
exceed 3 while the winner decile falls below 0.5 -- so WML carries a large conditional
negative beta and gets destroyed when the market rebounds. A winners-minus-losers spread
throws away exactly the variable that carries the mechanism. `legs()` and `leg_betas()`
keep it.

All series are simple daily returns in decimals. French portfolio files are *total*
returns while the factor files are *excess*, so anything beta-related subtracts RF first;
getting this backwards biases betas in a way that is easy to miss and hard to spot later.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..config import SEALED_START, Tier, require_unlocked
from ..data import french

#: Post-1990 is the default exploration window: comparable market structure, and the
#: pre-1990 record is dominated by low-volatility mid-century episodes whose
#: vol-standardized severity is enormous but whose relevance to a modern book is thin.
MODERN_START = "1990-01-01"

_BLOCK = {"vw": 0, "ew": 1}


def deciles(weighting: str = "vw") -> pd.DataFrame:
    """Ten momentum-sorted decile portfolios, D1 (losers) .. D10 (winners)."""
    df = french.load("mom_10_daily", block=_BLOCK[weighting])
    df.columns = [f"D{i}" for i in range(1, 11)]
    return df


def wml(weighting: str = "vw") -> pd.Series:
    """Decile winners-minus-losers spread, D10 - D1."""
    d = deciles(weighting)
    return (d["D10"] - d["D1"]).rename(f"wml_{weighting}")


def legs(weighting: str = "vw") -> pd.DataFrame:
    """Winner and loser legs separately, in excess-return terms, plus the market.

    Excess returns are what beta is defined against; French portfolio returns are total,
    so RF is subtracted here.
    """
    d = deciles(weighting)
    ff = french.market()
    out = pd.DataFrame(index=d.index)
    out["winners"] = d["D10"] - ff["RF"]
    out["losers"] = d["D1"] - ff["RF"]
    out["wml"] = d["D10"] - d["D1"]
    out["mkt"] = ff["Mkt-RF"]
    out["rf"] = ff["RF"]
    return out.dropna(subset=["winners", "losers", "mkt"])


def leg_betas(weighting: str = "vw", window: int = 126) -> pd.DataFrame:
    """Rolling market betas of each leg, and the implied beta of the spread.

    This is the Daniel-Moskowitz diagnostic. ``wml_beta`` going sharply negative is the
    setup for a crash: the strategy is short the market precisely when the market is
    poised to rebound. Computed on a trailing window and therefore usable point-in-time.
    """
    lg = legs(weighting)
    mkt_var = lg["mkt"].rolling(window).var()
    out = pd.DataFrame(index=lg.index)
    out["winner_beta"] = lg["winners"].rolling(window).cov(lg["mkt"]) / mkt_var
    out["loser_beta"] = lg["losers"].rolling(window).cov(lg["mkt"]) / mkt_var
    out["wml_beta"] = out["winner_beta"] - out["loser_beta"]
    return out


def size_split(weighting: str = "vw") -> pd.DataFrame:
    """WML within each size quintile, from the 25 size x momentum portfolios."""
    df = french.load("size_mom_25_daily", block=_BLOCK[weighting])
    names = {1: ("SMALL LoPRIOR", "SMALL HiPRIOR"), 5: ("BIG LoPRIOR", "BIG HiPRIOR")}
    out = pd.DataFrame(index=df.index)
    for i in range(1, 6):
        lo, hi = names.get(i, (f"ME{i} PRIOR1", f"ME{i} PRIOR5"))
        out[f"wml_me{i}"] = df[hi] - df[lo]
    out["wml_allsize"] = out.mean(axis=1)
    return out


def panel(*, modern_only: bool = True, include_sealed: bool = False) -> pd.DataFrame:
    """Every momentum construction in one frame, for side-by-side exploration.

    Truncates at ``config.SEALED_START`` by default. This is not paranoia: an earlier
    exploration run passed the full panel to the episode detector and surfaced two
    post-2023 events before the spec was frozen. Opting in is now explicit and audited,
    so the same slip cannot happen silently.
    """
    parts = [
        french.momentum().rename("wml_official"),
        wml("vw"),
        wml("ew"),
        size_split("vw").add_suffix("_vw"),
        size_split("ew")[["wml_me1", "wml_me5", "wml_allsize"]].add_suffix("_ew"),
    ]
    out = pd.concat(parts, axis=1)
    if modern_only:
        out = out.loc[MODERN_START:]
    if not include_sealed:
        out = out.loc[: str(SEALED_START - dt.timedelta(days=1))]
    else:
        require_unlocked(Tier.SEALED)
    return out


def context(*, modern_only: bool = True) -> pd.DataFrame:
    """Market/leg context aligned to the construction panel."""
    lg = legs("vw")
    bt = leg_betas("vw")
    out = lg[["mkt", "winners", "losers"]].join(bt)
    return out.loc[MODERN_START:] if modern_only else out
