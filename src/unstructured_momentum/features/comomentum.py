"""Lou-Polk comomentum, and the first post-2015 out-of-sample test of it.

The measure
-----------
Lou & Polk (RFS 2022, 35(7):3272-3302). Comomentum is the average **abnormal** pairwise
correlation of weekly returns among the stocks in the extreme momentum deciles, measured
over the 12-month ranking window. "Abnormal" means after stripping common factor exposure
-- the paper adjusts for FF3 and 30 industry portfolios; this implementation adjusts for
FF3, and the omission of industry controls is recorded as a deviation rather than hidden.

The economics: momentum stocks have no particular reason to co-move once market, size and
value exposure are removed -- they are simply firms that happened to have gone up or down.
If their residuals start moving together, something is holding them as a *group*, and that
something is arbitrage capital running the same trade. High comomentum therefore indicates
a crowded, fragile momentum trade.

Why this test is worth running
------------------------------
The published sample **ends in 2015** and an adversarially-verified literature pass found
**no post-2015 out-of-sample evidence**. So computing it on 2013-2026 is a genuine test of
the field's canonical free crowding measure, and it is publishable in either direction.

It also has a sharply-posed local target. The project's severity model is vol-driven and
was *less* alarmed than the unconditional baseline before the Sept 2019 unwind, because
realised volatility was low. The specific question is therefore not "does comomentum
work" but: **was comomentum elevated on 2019-09-06 when volatility was not?** If yes, it
is the state variable the severity model is missing. If no, this avenue closes cleanly.

Deviations from the paper, stated up front
------------------------------------------
* FF3 residualisation only; no 30-industry adjustment (French's daily industry portfolios
  are available, but the paper's exact industry assignment per stock is not).
* Universe is IWV constituents (survivorship-free, from holdings) rather than CRSP.
* Daily-panel coverage begins 2013, so the test window is 2014-2026 once a 52-week
  formation window is required. Archive gaps (Jan-Jul 2017, Dec 2014-Jan 2015) are holes,
  never interpolated.
* The paper reports an average *partial* correlation of each stock's residual against the
  leg excluding that stock. This computes the mean of the strict upper triangle of the
  residual correlation matrix -- closely related, not algebraically identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import french, ishares
from ..factor import legs as legs_mod

#: The paper's formation window: 52 weekly observations.
WEEKS = 52
#: Minimum overlapping weeks before a pair contributes a correlation.
MIN_WEEKS = 40
#: Minimum names in a leg before the average is meaningful.
MIN_NAMES = 20


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Friday-to-Friday weekly returns from a daily price panel."""
    wk = prices.resample("W-FRI").last()
    return wk.pct_change().mask(lambda x: x.abs() > 0.8)


def _residualise(rets: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Strip FF3 exposure from each name's weekly returns.

    Solved per name because missing patterns differ (listings, index entries, halts), and
    a shared observation mask would discard usable data.
    """
    idx = rets.index.intersection(factors.index)
    Y, X = rets.loc[idx], factors.loc[idx]
    design = np.column_stack([np.ones(len(X)), X.to_numpy()])
    out = pd.DataFrame(index=idx, columns=Y.columns, dtype=float)

    for col in Y.columns:
        y = Y[col]
        ok = y.notna()
        if ok.sum() < MIN_WEEKS:
            continue
        A = design[ok.to_numpy()]
        beta, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        out.loc[ok[ok].index, col] = y[ok].to_numpy() - A @ beta
    return out


def average_pairwise(resid: pd.DataFrame) -> float:
    """Mean of the strict upper triangle of the residual correlation matrix."""
    usable = resid.loc[:, resid.notna().sum() >= MIN_WEEKS]
    if usable.shape[1] < MIN_NAMES:
        return float("nan")
    c = usable.corr(min_periods=MIN_WEEKS).to_numpy()
    iu = np.triu_indices_from(c, k=1)
    vals = c[iu][np.isfinite(c[iu])]
    return float(vals.mean()) if vals.size else float("nan")


def series(
    fund: str = "IWV",
    *,
    start: str = "2014-01-01",
    end: str | None = None,
    freq: str = "ME",
) -> pd.DataFrame:
    """Comomentum for both legs at each formation date.

    ``spread`` is winner-leg minus loser-leg comomentum. The sign matters: crowding
    concentrated in the losers is a different setup from crowding in the winners, and
    Daniel-Moskowitz locate the crash mechanism in the short leg.
    """
    px = ishares.price_panel(fund)
    px = px[legs_mod.valid_tickers(px.columns)]
    wk = weekly_returns(px)

    ff = french.market()[["Mkt-RF", "SMB", "HML"]]
    ff_wk = (1 + ff).resample("W-FRI").prod() - 1

    dates = pd.date_range(start, end or px.index.max(), freq=freq)
    rows = []

    for d in dates:
        hist = px.loc[:d]
        if len(hist) < 300:
            continue
        try:
            lg = legs_mod.build(d, prices=hist)
        except (ValueError, KeyError):
            continue

        window = wk.loc[:d].tail(WEEKS)
        if len(window) < MIN_WEEKS:
            continue

        rec = {"as_of": d, "n_universe": lg["n_universe"]}
        for name, members in (("winners", lg["winners"]), ("losers", lg["losers"])):
            cols = members.index.intersection(window.columns)
            if len(cols) < MIN_NAMES:
                rec[f"comom_{name}"] = np.nan
                continue
            rec[f"comom_{name}"] = average_pairwise(_residualise(window[cols], ff_wk))
        rec["comom_spread"] = rec.get("comom_winners", np.nan) - rec.get("comom_losers", np.nan)
        rec["comom_avg"] = np.nanmean([rec.get("comom_winners"), rec.get("comom_losers")])
        rows.append(rec)

    return pd.DataFrame(rows).set_index("as_of").sort_index()


def crash_day_frequency(
    comom: pd.Series,
    momentum: pd.Series,
    *,
    months_ahead: int = 3,
    threshold: float = -0.01,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """The paper's tail-risk test: crash-day frequency by comomentum bucket.

    Lou & Polk report the fraction of days with momentum returns below -1% in the three
    months after formation rising from 8.4% (low comomentum) to 22.5% (high) -- a 2.7x
    increase. This reproduces that test on the out-of-sample window.
    """
    df = comom.dropna().to_frame("comom")
    df["bucket"] = pd.qcut(df["comom"].rank(method="first"), n_buckets, labels=range(1, n_buckets + 1))

    rows = []
    for as_of, row in df.iterrows():
        fwd = momentum.loc[as_of : as_of + pd.DateOffset(months=months_ahead)]
        if len(fwd) < 20:
            continue
        rows.append(
            {
                "as_of": as_of,
                "bucket": int(row["bucket"]),
                "comom": row["comom"],
                "crash_day_freq": float((fwd < threshold).mean()),
                "fwd_return": float((1 + fwd).prod() - 1),
                "n_days": len(fwd),
            }
        )

    out = pd.DataFrame(rows)
    if not len(out):
        return out
    return out.groupby("bucket").agg(
        n_formations=("as_of", "size"),
        mean_comom=("comom", "mean"),
        crash_day_freq=("crash_day_freq", "mean"),
        mean_fwd_return=("fwd_return", "mean"),
    )
