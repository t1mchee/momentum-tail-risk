"""Stock-level momentum legs built from point-in-time index holdings.

Why build legs rather than use French's decile portfolios
---------------------------------------------------------
French gives decile *returns*, which is enough to measure what momentum did but not
enough to ask *who* momentum was. Every interesting positioning question -- how crowded is
the long leg, how constrained is the short leg, which names are on the Reg SHO threshold
list, what narrative does the book embody -- needs constituents, not a return series.

The universe comes from IWV (Russell 3000) point-in-time holdings, which are
survivorship-free by construction: a name is present exactly while it was in the index,
including names later acquired or delisted. That property matters most for the *short*
leg, whose members disproportionately stop existing, and it is precisely what free price
APIs destroy.

Construction
------------
Standard 12-1 momentum: cumulative return from t-12 months to t-1 month, skipping the most
recent month to avoid the short-term reversal effect that would otherwise contaminate the
signal. Ranked into deciles; the long leg is D10, the short leg is D1.

Prices are the fund's own reported prices: close prices for the as-of date, **unadjusted**
for dividends or splits. Two consequences, both handled explicitly rather than hidden:

* *Dividends.* Omitting them biases the ranking against high-yield names by roughly their
  yield over the formation window. Second-order for a 12-month sort, but real.
* *Splits.* These are **not** handled implicitly, contrary to what one might assume from
  the fund repricing its own position. A reverse split is a step change in the price
  series, so an end/start price ratio reads it as enormous momentum: Rite Aid's 1-for-20
  in April 2019 scored as roughly +1,900% and put RAD in the top decile of winners as of
  2019-09-06. Momentum is therefore computed by compounding *masked* daily returns
  (`_clean_returns` blanks moves beyond `MAX_ABS_DAILY_RETURN`), which treats the split
  day as flat and leaves the rest of the window intact. RAD then scores -0.777, which is
  correct.

Known coverage limits, inherited from the holdings archive:

* daily from 2013; month-end only from Sept 2006 to 2012
* Jan-Jul 2017 missing upstream
* Aug 2007 is month-end only, and the quant quake round-tripped inside a single month,
  so it cannot be seen at this granularity
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..data import ishares

#: Momentum formation window in trading days: skip the most recent month, look back a year.
LOOKBACK_DAYS = 252
SKIP_DAYS = 21

#: A single-day move beyond this is treated as a corporate action rather than a return.
#: Holdings prices are unadjusted, so splits appear as ~50% or ~-50% jumps.
MAX_ABS_DAILY_RETURN = 0.60

#: Holdings files carry non-security rows whose ticker is a placeholder ("-"), a cash
#: marker, or an internal code. They parse cleanly and would otherwise be ranked as if
#: they were stocks -- "-" turned up in the loser decile on the first run.
_TICKER_OK = re.compile(r"^[A-Z][A-Z0-9.]{0,6}$")


def valid_tickers(cols) -> list[str]:
    """Keep only plausible US equity symbols."""
    return [c for c in cols if isinstance(c, str) and _TICKER_OK.match(c)]


def _clean_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns from an unadjusted price panel, with corporate actions blanked."""
    rets = prices.sort_index().pct_change()
    return rets.mask(rets.abs() > MAX_ABS_DAILY_RETURN)


def momentum_scores(
    prices: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    lookback: int = LOOKBACK_DAYS,
    skip: int = SKIP_DAYS,
    min_obs: int | None = None,
) -> pd.Series:
    """12-1 momentum score per name, computed strictly from data available at ``as_of``.

    Returns NaN for names without enough history, which are then excluded from the sort
    rather than silently ranked as if they had a score.
    """
    as_of = pd.Timestamp(as_of)
    px = prices.loc[:as_of].sort_index()
    if len(px) < lookback + skip:
        raise ValueError(
            f"need {lookback + skip} observations before {as_of.date()}, have {len(px)}"
        )

    # Compound CLEANED daily returns rather than taking an end/start price ratio.
    # Holdings prices are unadjusted, so a reverse split is a step change in price: Rite
    # Aid's 1-for-20 in April 2019 makes a raw ratio read as roughly +1,900% momentum, and
    # RAD duly appeared in the top decile of winners as of 2019-09-06. Compounding masked
    # returns treats the split day as flat, which is approximately the true economic
    # return, and keeps the rest of the window intact.
    window = px.iloc[-(lookback + skip) :]
    rets = _clean_returns(window)
    formation = rets.iloc[: lookback]  # t-12m .. t-1m, i.e. skipping the recent month

    min_obs = min_obs or int(0.8 * lookback)
    enough = formation.notna().sum() >= min_obs

    score = (1.0 + formation.fillna(0.0)).prod() - 1.0
    return score.where(enough).dropna()


def deciles(scores: pd.Series, n: int = 10) -> pd.Series:
    """Assign decile ranks 1..n, 1 = worst momentum (losers), n = best (winners)."""
    return pd.qcut(scores.rank(method="first"), n, labels=range(1, n + 1)).astype(int)


def build(
    as_of: pd.Timestamp | str,
    *,
    fund: str = "IWV",
    n_deciles: int = 10,
    prices: pd.DataFrame | None = None,
) -> dict:
    """Construct the momentum legs as of a date.

    Returns the winner and loser constituent lists, their scores, and the price panel used,
    so downstream crowding measures can work from exactly the same universe.
    """
    as_of = pd.Timestamp(as_of)
    px = prices if prices is not None else ishares.price_panel(fund, end=str(as_of.date()))
    px = px[valid_tickers(px.columns)]

    scores = momentum_scores(px, as_of)

    # Restrict to POINT-IN-TIME index membership. Without this the sort ranks every ticker
    # that has ever appeared in the fund, which is not an investable universe at any date: a
    # name deleted at the June reconstitution keeps prices for a few sessions and then stops,
    # so its final formation month is zero-filled and it is ranked as though it could still be
    # held. Measured at 25 percent of the August and September loser leg before this change.
    members = ishares.members_at(as_of, fund)
    if members:
        scores = scores[scores.index.isin(members)]

    dec = deciles(scores, n_deciles)

    winners = scores[dec == n_deciles].sort_values(ascending=False)
    losers = scores[dec == 1].sort_values()

    return {
        "as_of": as_of,
        "fund": fund,
        "n_universe": int(len(scores)),
        "winners": winners,
        "losers": losers,
        "scores": scores,
        "deciles": dec,
        "prices": px,
    }


def leg_returns(legs: dict, *, weighting: str = "equal") -> pd.DataFrame:
    """Realised daily returns of each leg over the panel's history.

    Uses the leg membership fixed at ``as_of``, so this is the *forward* or *backward*
    performance of a portfolio formed on that date, not a rebalanced strategy.
    """
    rets = _clean_returns(legs["prices"])
    w = rets[legs["winners"].index.intersection(rets.columns)]
    l = rets[legs["losers"].index.intersection(rets.columns)]

    if weighting != "equal":
        raise NotImplementedError("only equal weighting is supported from a price panel")

    out = pd.DataFrame({"winners": w.mean(axis=1), "losers": l.mean(axis=1)})
    out["wml"] = out["winners"] - out["losers"]
    return out


def summary(legs: dict) -> pd.Series:
    """Compact description of the legs, for the memo and for sanity checking."""
    return pd.Series(
        {
            "as_of": legs["as_of"].date(),
            "universe": legs["n_universe"],
            "n_winners": len(legs["winners"]),
            "n_losers": len(legs["losers"]),
            "winner_median_12_1": float(legs["winners"].median()),
            "loser_median_12_1": float(legs["losers"].median()),
            "spread": float(legs["winners"].median() - legs["losers"].median()),
        }
    )
