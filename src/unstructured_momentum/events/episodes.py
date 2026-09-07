"""Detection and characterisation of momentum reversal events.

Deliberately descriptive, not predictive. The point at this stage is to let the event set
*emerge* from the return data across many constructions rather than to defend a
hand-curated list of famous crashes. The famous list is used only as a sanity check: a
detector that does not rediscover Aug 2007, Mar 2009 and Sept 2019 unprompted is wrong.

Two different risks, and only one of them is this project's subject
-------------------------------------------------------------------
An unconstrained peak-to-trough drawdown detector run on momentum does not find crashes.
It finds momentum's **secular decay**: single "episodes" running from 2009 to 2021, with
depths near -99% in equal-weighted constructions and cumulative market returns of +900%
inside the window. Those are real and a PM should care, but they are a slow erosion of
the premium, not a reversal.

This project is about the *fast* risk, so the primary primitive here is a **rolling
h-day return below a threshold**, clustered into distinct events -- not peak-to-trough.
`secular_drawdowns()` retains the slow view for contrast, explicitly labelled.

Telling the two mechanisms apart
--------------------------------
* **Type A, panic rebound** -- weak prior two-year market, high trailing volatility, and a
  *strongly positive* market return during the drawdown. Momentum loses because it is
  conditionally short the market into a rebound.
* **Type B, crowded rotation** -- benign prior market state, unremarkable volatility, and
  a market return during the drawdown near zero. AQR's observation about Aug 2007 makes
  this the discriminator: the factor bled while the index was flat.

``mkt_during`` is therefore the most diagnostic field in the table. The classification
rule is pre-registered so the taxonomy stays falsifiable rather than fitted to the
narrative after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Trading-day lookback for the "bear state" indicator, per Daniel & Moskowitz.
BEAR_WINDOW = 504
VOL_WINDOW = 100


# --------------------------------------------------------------------------------------
# Context — computed on FULL history, then sampled at episode dates
# --------------------------------------------------------------------------------------


def market_context(returns: pd.Series, market: pd.Series) -> pd.DataFrame:
    """Trailing state variables, all knowable at each date.

    Computed on the full available history and only afterwards sampled at episode dates.
    Computing them on a post-1990 slice would leave the first two years of every rolling
    window undefined and silently drop the early episodes from any classification.
    """
    r, m = returns.dropna(), market.dropna()
    idx = r.index.union(m.index)
    r, m = r.reindex(idx), m.reindex(idx)

    return pd.DataFrame(
        {
            "trail_vol": r.rolling(VOL_WINDOW).std() * np.sqrt(252),
            "mkt_vol": m.rolling(VOL_WINDOW).std() * np.sqrt(252),
            "mkt_prior_2y": (1 + m).rolling(BEAR_WINDOW).apply(np.prod, raw=True) - 1,
        },
        index=idx,
    )


# --------------------------------------------------------------------------------------
# Primary detector — sharp reversals
# --------------------------------------------------------------------------------------


def sharp_reversals(
    returns: pd.Series,
    market: pd.Series,
    *,
    window: int = 10,
    threshold: float | None = None,
    percentile: float = 1.0,
    cluster_days: int = 21,
) -> pd.DataFrame:
    """Distinct episodes where the ``window``-day return was extremely negative.

    Either an absolute ``threshold`` or, by default, the bottom ``percentile`` of the
    series' own rolling-return distribution. Overlapping windows are collapsed so that one
    crash produces one row: candidate days within ``cluster_days`` of each other are
    treated as the same event and represented by their worst window.
    """
    r = returns.dropna()
    cum = (1 + r).rolling(window).apply(np.prod, raw=True) - 1
    cum = cum.dropna()
    if not len(cum):
        return pd.DataFrame()

    cut = threshold if threshold is not None else np.nanpercentile(cum, percentile)
    cand = cum[cum <= cut]
    if not len(cand):
        return pd.DataFrame()

    # Collapse near-in-time candidates into single events.
    groups, last, gid = [], None, -1
    for t in cand.index:
        if last is None or (t - last).days > cluster_days:
            gid += 1
        groups.append(gid)
        last = t
    cand = cand.to_frame("cum_ret")
    cand["g"] = groups

    ctx = market_context(returns, market)
    m = market.dropna()

    rows: list[dict] = []
    for _, chunk in cand.groupby("g"):
        trough_at = chunk["cum_ret"].idxmin()
        depth = float(chunk["cum_ret"].min())
        pos = r.index.get_loc(trough_at)
        peak_at = r.index[max(pos - window, 0)]

        c = ctx.reindex([peak_at]).iloc[0]
        rows.append(
            {
                "peak_at": peak_at,
                "trough_at": trough_at,
                "depth": depth,
                "window": window,
                # Sampled at peak_at => strictly knowable before the episode began.
                "trail_vol_at_peak": float(c["trail_vol"]),
                "mkt_vol_at_peak": float(c["mkt_vol"]),
                "mkt_prior_2y": float(c["mkt_prior_2y"]),
                "mkt_during": float((1 + m.loc[peak_at:trough_at]).prod() - 1),
            }
        )

    out = pd.DataFrame(rows)
    # Severity in units of its own trailing volatility -- the "surprise" scale that
    # strips out the part a volatility forecast already explains.
    out["vol_std_depth"] = out["depth"] / (
        out["trail_vol_at_peak"] * np.sqrt(out["window"] / 252.0)
    )
    return out.sort_values("depth").reset_index(drop=True)


def classify(
    eps: pd.DataFrame,
    *,
    bear_threshold: float = 0.0,
    rebound_threshold: float = 0.02,
) -> pd.DataFrame:
    """Pre-registered Type A / Type B rule. Deliberately crude and stated in advance.

    Type A requires *both* a weak prior two-year market and a positive market return
    during the drawdown. Anything not clearly one or the other is left ``mixed`` rather
    than forced into a bucket; how many land in ``mixed`` is itself evidence about
    whether the taxonomy holds up.
    """
    if not len(eps):
        return eps
    out = eps.copy()
    panic = out["mkt_prior_2y"] < bear_threshold
    rebound = out["mkt_during"] > rebound_threshold
    flat = out["mkt_during"].abs() < rebound_threshold

    out["mech_type"] = "mixed"
    out.loc[panic & rebound, "mech_type"] = "A_panic_rebound"
    out.loc[~panic & flat, "mech_type"] = "B_crowded_rotation"
    return out


def catalogue(returns: pd.Series, market: pd.Series, **kw) -> pd.DataFrame:
    """Detect and classify sharp reversals in one call."""
    return classify(sharp_reversals(returns, market, **kw))


def across_constructions(panel: pd.DataFrame, market: pd.Series, **kw) -> pd.DataFrame:
    """Run the detector over every construction and stack the results.

    Lets us ask which episodes are robust to construction and which are artefacts of one
    way of building the factor -- the question Jan 2021 raised.
    """
    frames = []
    for col in panel.columns:
        c = catalogue(panel[col].dropna(), market, **kw)
        if len(c):
            c.insert(0, "construction", col)
            frames.append(c)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def consensus(stacked: pd.DataFrame, *, tolerance_days: int = 21) -> pd.DataFrame:
    """Cluster episodes across constructions into distinct events.

    ``n_constructions`` measures robustness: an event visible in one construction only is
    a very different object from one visible in all twelve.
    """
    if not len(stacked):
        return stacked
    s = stacked.sort_values("trough_at").reset_index(drop=True)
    cl, last, cid = [], None, -1
    for t in s["trough_at"]:
        if last is None or (t - last).days > tolerance_days:
            cid += 1
        cl.append(cid)
        last = t
    s["event_id"] = cl

    return (
        s.groupby("event_id")
        .agg(
            trough_at=("trough_at", "median"),
            n_constructions=("construction", "nunique"),
            worst_depth=("depth", "min"),
            median_depth=("depth", "median"),
            worst_vol_std=("vol_std_depth", "min"),
            mkt_during=("mkt_during", "median"),
            mkt_prior_2y=("mkt_prior_2y", "median"),
            trail_vol=("trail_vol_at_peak", "median"),
            types=("mech_type", lambda x: ",".join(sorted(set(x)))),
        )
        .sort_values("worst_depth")
    )


# --------------------------------------------------------------------------------------
# Contrast view — the slow risk, kept separate and labelled
# --------------------------------------------------------------------------------------


def secular_drawdowns(returns: pd.Series, *, min_depth: float = 0.20) -> pd.DataFrame:
    """Unbounded peak-to-trough drawdowns: momentum's slow decay, not its crashes.

    Retained for contrast and for the memo's limitations section. These episodes run for
    years and are a different risk from the reversal risk this system monitors; conflating
    the two is how a "momentum crash model" ends up predicting the equity risk premium.
    """
    r = returns.dropna()
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    underwater = dd < 0

    rows = []
    grp = (underwater != underwater.shift()).cumsum()
    for _, chunk in dd[underwater].groupby(grp[underwater]):
        depth = float(chunk.min())
        if depth > -min_depth:
            continue
        start = r.index.get_loc(chunk.index[0])
        rows.append(
            {
                "peak_at": r.index[max(start - 1, 0)],
                "trough_at": chunk.idxmin(),
                "depth": depth,
                "days_underwater": int(len(chunk)),
            }
        )
    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)
