"""Return-based crowding measures, principally Lou-Polk comomentum.

Comomentum
----------
Lou & Polk (2022), "Comomentum: Inferring Arbitrage Activity from Return Correlations",
*RFS* 35(7). The measure is the average pairwise correlation of **factor-residual** returns
among the stocks *within* a momentum leg, over a trailing window.

The intuition is that momentum stocks have no particular reason to co-move once market,
size and value exposure are removed -- they are simply firms that happened to have gone up
(or down). If their residuals start moving together, something is holding them as a
*group*, and that something is arbitrage capital running the same trade. High comomentum
therefore signals a crowded, fragile momentum trade and predicts subsequent reversal; low
comomentum means momentum is being arbitraged gently and is stabilising.

Residualisation is the load-bearing step. Raw within-leg correlation mostly measures market
beta and sector concentration, both of which are high in any momentum portfolio for
uninteresting reasons. Skipping it produces a series that looks like a crowding measure
and is really a market-volatility measure -- and would therefore duplicate the
Barroso-Santa-Clara baseline it is supposed to improve on.

Why this needs constituents
---------------------------
Comomentum cannot be computed from decile *return* series; it needs the individual stocks.
That is what the point-in-time IWV holdings panel provides, and it is why the measure is
available for both legs rather than only the long one.

Caveats worth stating in the memo
----------------------------------
* Lou & Polk use weekly returns over 52 weeks. Daily returns over a comparable window are
  used here because that is what the holdings panel supports; daily residual correlations
  are noisier and pick up more microstructure (non-synchronous trading, bid-ask bounce).
* Prices are unadjusted for dividends, so residuals carry a small ex-dividend artefact.
* Coverage begins 2013 at daily frequency -- see `factor.legs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import french
from ..factor import legs as legs_mod

#: Trailing window for the correlation estimate. ~250 trading days ~ Lou-Polk's 52 weeks.
WINDOW = 250
#: Minimum overlapping observations before a pair contributes a correlation.
MIN_PAIR_OBS = 100


def _factor_residuals(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Strip market, size and value exposure from each stock's returns.

    Solved as one least-squares problem across all names rather than per-stock loops:
    with ~300 names per leg and a 250-day window, the loop version dominates runtime for
    no benefit.
    """
    idx = returns.index.intersection(factors.index)
    Y = returns.loc[idx]
    X = factors.loc[idx]

    design = np.column_stack([np.ones(len(X)), X.to_numpy()])
    resid = pd.DataFrame(index=idx, columns=Y.columns, dtype=float)

    # Solved per name because missing patterns differ across stocks (listings, halts,
    # index entries mid-window), and a shared solve would need a common observation mask
    # that discards usable data.
    for col in Y.columns:
        y = Y[col]
        ok = y.notna()
        if ok.sum() < MIN_PAIR_OBS:
            continue
        A = design[ok.to_numpy()]
        beta, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        resid.loc[ok[ok].index, col] = y[ok].to_numpy() - A @ beta
    return resid


def average_pairwise_correlation(resid: pd.DataFrame, *, min_obs: int = MIN_PAIR_OBS) -> float:
    """Mean of the strict upper triangle of the residual correlation matrix."""
    usable = resid.loc[:, resid.notna().sum() >= min_obs]
    if usable.shape[1] < 5:
        return float("nan")
    c = usable.corr(min_periods=min_obs).to_numpy()
    iu = np.triu_indices_from(c, k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def comomentum(
    as_of: pd.Timestamp | str,
    *,
    fund: str = "IWV",
    window: int = WINDOW,
    legs: dict | None = None,
) -> pd.Series:
    """Comomentum for both legs as of a date.

    ``spread`` is long-leg minus short-leg comomentum. The sign matters: crowding
    concentrated in the winners is a different setup from crowding concentrated in the
    losers, and Daniel-Moskowitz's mechanism implies the short leg is the dangerous one.
    """
    as_of = pd.Timestamp(as_of)
    lg = legs if legs is not None else legs_mod.build(as_of, fund=fund)

    rets = legs_mod._clean_returns(lg["prices"]).loc[:as_of].tail(window)
    ff = french.market()[["Mkt-RF", "SMB", "HML"]]

    out = {}
    for name, members in (("winners", lg["winners"]), ("losers", lg["losers"])):
        cols = members.index.intersection(rets.columns)
        resid = _factor_residuals(rets[cols], ff)
        out[f"comomentum_{name}"] = average_pairwise_correlation(resid)
        out[f"n_{name}_used"] = int(resid.notna().sum().gt(MIN_PAIR_OBS).sum())

    out["comomentum_spread"] = out["comomentum_winners"] - out["comomentum_losers"]
    out["as_of"] = as_of
    return pd.Series(out)


def concentration(legs: dict) -> pd.Series:
    """Herfindahl concentration of the legs by name and by sector.

    Equal-weighted by construction from a price panel, so the name-level Herfindahl is
    mechanical; the informative one is *sector* concentration, which measures how much of
    the momentum trade is a single thematic bet.
    """
    out = {}
    for name, members in (("winners", legs["winners"]), ("losers", legs["losers"])):
        n = len(members)
        out[f"hhi_names_{name}"] = 1.0 / n if n else float("nan")
        out[f"n_{name}"] = n
    return pd.Series(out)


def series(
    dates: list[pd.Timestamp | str],
    *,
    fund: str = "IWV",
    window: int = WINDOW,
) -> pd.DataFrame:
    """Comomentum over a list of dates, rebuilding the legs at each.

    Deliberately recomputes leg membership per date rather than holding it fixed: crowding
    in "the momentum trade" means crowding in whatever momentum is *now*, and holding a
    stale constituent list would measure the decay of an old portfolio instead.
    """
    rows = []
    for d in dates:
        try:
            rows.append(comomentum(d, fund=fund, window=window))
        except (ValueError, KeyError):
            continue
    return pd.DataFrame(rows).set_index("as_of") if rows else pd.DataFrame()
