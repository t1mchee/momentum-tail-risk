"""Cross-factor co-movement: the deleveraging channel.

Why this module exists
----------------------
The Aug 2007 "quant quake" -- the founding episode of the entire crowded-quant-unwind
literature -- **is not detectable in daily momentum returns**. Measured on Fama-French
daily WML it sits at the 5.97th percentile of modern-era 10-day returns, and it does not
clear a 1% or 2.5% event threshold in any of the twelve momentum constructions tested. It
only appears at a 5% cut, alongside 98 other "events".

The reason is that it was never a momentum crash. Over 2007-08-06 to 08-09 every
long-short factor moved together, moderately, in the same direction, and then violently
reversed on 08-10::

    worst 10d, Aug 2007      raw      modern pctile      z
    HML                    -3.72%          3.10       -6.53
    WML                    -4.53%          5.97       -5.05
    ST_Rev                 -3.85%          3.43       -4.52
    SMB                    -3.18%          4.41       -3.00

No single factor's move was historically extreme. The *joint* move was. Averaging the
standardized 10-day returns across factors puts Aug 2007 at the **0.10th percentile** --
the second worst joint deleveraging of the modern era, behind only March 2020.

Design implication
------------------
Momentum reversal risk has a component that is not about momentum at all: it is about
leverage in the multi-factor arbitrage complex. When levered market-neutral books are
forced to degross, they sell every factor at once, and momentum is collateral damage. A
monitor that only watches momentum is structurally blind to this channel, which is
precisely the channel that produced the most famous event in the field.

So the system carries two channels: a momentum-specific one and this joint one. They are
reported separately rather than blended, because they imply different responses -- a
momentum-specific reversal argues for cutting momentum, while a joint deleveraging argues
for cutting gross exposure across the book.

Caveat, stated plainly: this is a *contemporaneous* stress measure built from returns, so
on its own it identifies deleveraging as it happens rather than forecasting it. Its role
is to define the event class and to condition the text and positioning layers, not to
serve as a leading indicator by itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import french

#: The long-short factor complex a levered market-neutral book typically runs. ST_Rev is
#: included deliberately: Khandani & Lo identify contrarian/short-term-reversal strategies
#: as the epicentre of Aug 2007, and it is the single most extreme series in March 2020
#: (-17.7 sigma), so omitting it would understate exactly the events of interest.
FACTORS = ("wml", "hml", "smb", "st_rev")

_EXTRA = {
    "st_rev_daily": "F-F_ST_Reversal_Factor_daily_CSV.zip",
    "lt_rev_daily": "F-F_LT_Reversal_Factor_daily_CSV.zip",
}


def _register() -> None:
    french.DATASETS.update(_EXTRA)


def factor_panel() -> pd.DataFrame:
    """Daily returns for the long-short factor complex."""
    _register()
    ff = french.market()
    return pd.DataFrame(
        {
            "wml": french.momentum(),
            "hml": ff["HML"],
            "smb": ff["SMB"],
            "st_rev": french.load("st_rev_daily")["ST_Rev"],
        }
    ).dropna()


def standardized(window: int = 10, vol_window: int = 100) -> pd.DataFrame:
    """Rolling ``window``-day factor returns in units of their own trailing volatility.

    The volatility scaler is lagged by ``window`` days so it is estimated entirely before
    the return window opens -- otherwise the denominator would embed the very move being
    standardized, which flatters extreme events.
    """
    f = factor_panel()
    cum = (1 + f).rolling(window).apply(np.prod, raw=True) - 1
    vol = f.rolling(vol_window).std().shift(window) * np.sqrt(window)
    return (cum / vol).dropna()


def joint_stress(window: int = 10, vol_window: int = 100) -> pd.Series:
    """Mean standardized move across the factor complex.

    Strongly negative means every factor lost together in volatility-adjusted terms --
    the deleveraging signature.
    """
    return standardized(window, vol_window).mean(axis=1).rename("joint_z")


def dispersion(window: int = 10, vol_window: int = 100) -> pd.Series:
    """Cross-factor dispersion of standardized moves.

    Distinguishes a *broad* degrossing (low dispersion, everything down together) from a
    *rotation* (high dispersion, one factor down while another rallies). Aug 2007 is the
    former; Sept 2019, where momentum fell as value rallied, is the latter.
    """
    return standardized(window, vol_window).std(axis=1).rename("factor_dispersion")


def rolling_correlation(window: int = 63) -> pd.Series:
    """Average pairwise correlation among factor returns.

    Rising co-movement across nominally independent factors is the classic signature of a
    common levered holder -- the return-based crowding intuition behind Lou & Polk's
    comomentum, applied across factors rather than within a leg.
    """
    f = factor_panel()
    corr = f.rolling(window).corr()
    n = len(f.columns)
    # Mean of the strict upper triangle at each date.
    out = corr.groupby(level=0).apply(
        lambda x: (x.to_numpy().sum() - n) / (n * (n - 1))
    )
    return out.rename("factor_avg_corr")


def panel(window: int = 10) -> pd.DataFrame:
    """Assemble the deleveraging-channel features."""
    return pd.concat(
        [joint_stress(window), dispersion(window), rolling_correlation()], axis=1
    ).dropna(how="all")


def worst_joint_events(
    start: str = "1990", end: str = "2022", *, top: int = 12, cluster_days: int = 21
) -> pd.DataFrame:
    """Distinct worst joint-deleveraging episodes, with the per-factor breakdown."""
    z = standardized()
    s = joint_stress().loc[start:end]

    keep: list[pd.Timestamp] = []
    for d in s.nsmallest(top * 6).index:
        if all(abs((d - k).days) > cluster_days for k in keep):
            keep.append(d)
        if len(keep) >= top:
            break

    out = z.loc[keep].copy()
    out.insert(0, "joint_z", s.loc[keep])
    out["pctile"] = [(s < v).mean() * 100 for v in out["joint_z"]]
    return out.sort_values("joint_z")
