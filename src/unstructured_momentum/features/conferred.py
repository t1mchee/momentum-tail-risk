"""The conferred-tilt channel: what factor loadings the momentum sort BUILDS at each date.

The exposure that made September 2019 fragile was not a property the constituent firms had
and disclosed. It was conferred on them by the 12-1 ranking. Measured across 134 month-ends
the sort produces a NEGATIVE market-beta spread between its legs in 63 percent of months, so
a defensive tilt is momentum's modal output rather than an artefact of one episode.

Two estimands, and they are not interchangeable
-----------------------------------------------
**Conferred during formation.** Loadings of each leg's returns estimated over the formation
window itself, which closes before the book is formed. This is leak-free by construction and
is what the monitor reads: it says what tilt the sort has just built.

**Realised after formation.** Loadings estimated over the window that follows. This says what
the legs became, and it is NOT available at the time the book forms. The 2019 book shows a
1.46-point spread on this measure against 0.36 on the conferred one; quoting either number for
the other is a leakage error wearing a plausible figure.

The schema keeps them in separate columns and the availability stamp differs between them, so
a downstream consumer cannot reach the realised block by accident.

Coverage travels as data
------------------------
The iShares holdings panel is incomplete: nothing usable before September 2013, and 2017 and
2018 carry six and four month-ends against twelve for a full year. Callers get a coverage
frame rather than a docstring sentence, because a gap that lives in prose is a gap that gets
forgotten at the point a brief is generated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import french
from ..factor import cleanlegs as cl

PANEL = Path("data/raw/ishares/IWV/panel")
#: The first year the holdings panel exists. Loading from here rather than from just before the
#: requested window is what makes the formation gate behave identically to the note scripts.
PANEL_FIRST_YEAR = 2006

#: Daily market-factor standard deviation, decimal returns. Anything outside this band is a
#: units error rather than a market regime -- a hundredfold error here once produced market
#: betas of -382 that wrote 134 well-formed rows without raising anything.
MKT_SD_BAND = (0.003, 0.03)

#: Minimum DISTINCT DATES a year must carry to support a daily-return loading. The iShares
#: panel is monthly before 2013 and daily afterwards -- twelve dates a year against roughly
#: two hundred and fifty -- and counting MONTHS instead of dates marks 2007 through 2012 as
#: complete when they hold twelve observations in total. A loading cannot be estimated from
#: twelve points, so dates are what the gate counts.
MIN_DATES_PER_YEAR = 120


def _factors(rates: pd.Series | None = None) -> pd.DataFrame:
    m = french.market()[["Mkt-RF", "SMB", "HML"]]
    sd = float(m["Mkt-RF"].std())
    lo, hi = MKT_SD_BAND
    if not lo < sd < hi:
        raise ValueError(
            f"Mkt-RF daily sd is {sd:.6f}, outside {lo}-{hi}. These are not decimal returns; "
            "check units before any loading is estimated.")
    return m if rates is None else pd.concat([m, rates.rename("d10y")], axis=1)


def ten_year_change() -> pd.Series:
    d = pd.read_csv("data/raw/fred/DGS10.csv")
    d.columns = ["date", "y10"]
    s = pd.to_numeric(d["y10"], errors="coerce").set_axis(pd.to_datetime(d["date"])).dropna()
    return s.diff()


def load_panel(years) -> tuple[pd.DataFrame, pd.Series]:
    px, sec = {}, {}
    for y in years:
        f = PANEL / f"IWV_{y}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["as_of", "ticker", "sector", "price"])
        d = d[d.price.notna() & (d.price > 0)]
        px[y] = d.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
        sec[y] = d.drop_duplicates("ticker").set_index("ticker")["sector"]
    if not px:
        return pd.DataFrame(), pd.Series(dtype=object)
    P = pd.concat(px.values()).sort_index()
    S = pd.concat(sec.values())
    return P[~P.index.duplicated()], S[~S.index.duplicated()]


def coverage() -> pd.DataFrame:
    """Which years the holdings panel can support, as a frame a caller can act on.

    Reports dates and months separately because they diverge: the panel switches from monthly
    snapshots to daily around 2013, and a year with twelve months can hold twelve
    observations. `usable` gates on dates, since that is what a daily-return loading consumes.
    """
    rows = []
    for f in sorted(PANEL.glob("IWV_*.parquet")):
        y = int(f.stem.split("_")[1])
        a = pd.read_parquet(f, columns=["as_of"])["as_of"]
        n_d, n_m = int(a.nunique()), int(a.dt.to_period("M").nunique())
        rows.append({"year": y, "dates": n_d, "months": n_m,
                     "cadence": "daily" if n_d >= MIN_DATES_PER_YEAR else "monthly",
                     "usable": bool(n_d >= MIN_DATES_PER_YEAR)})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def first_usable_date() -> pd.Timestamp | None:
    """Earliest date the panel can support a loading, derived rather than hardcoded."""
    c = coverage()
    ok = c[c.usable]
    return pd.Timestamp(f"{int(ok.year.min())}-01-01") if len(ok) else None


def _loadings(r: pd.Series, X: pd.DataFrame, min_obs: int = 40) -> dict:
    d = pd.concat([r.rename("y"), X], axis=1).dropna()
    if len(d) < min_obs:
        return {}
    A = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(A, d["y"].to_numpy(), rcond=None)
    return dict(zip(["alpha", *X.columns], beta))


def conferred_at(asof: pd.Timestamp, P: pd.DataFrame, S: pd.Series, X: pd.DataFrame,
                 *, decile: float = 0.10, realised: bool = False) -> dict:
    """Tilt conferred by the sort at one date. Set realised=True for the after-the-fact block.

    The realised block is stamped available_at three months after the date it describes,
    because that is when the window it uses has closed. Nothing that reads available_at can
    then pick it up as if it were known at formation.
    """
    mom = cl.momentum(P, asof, sectors=S)
    mom = mom[(mom > -0.95) & (mom < 5.0)]
    if len(mom) < 300:
        return {}
    win, los = cl.legs(mom, decile)
    R = P.pct_change().where(lambda r: r.abs() < cl.JUMP)
    if realised:
        w = (R.index > asof) & (R.index <= asof + pd.DateOffset(months=3))
        avail = asof + pd.DateOffset(months=3)
    else:
        w = (R.index > asof - pd.DateOffset(months=12)) & (R.index <= asof)
        avail = asof
    rw = R.loc[w, [t for t in win if t in R.columns]].mean(axis=1)
    rl = R.loc[w, [t for t in los if t in R.columns]].mean(axis=1)
    lw, ll = _loadings(rw, X), _loadings(rl, X)
    if not lw or not ll:
        return {}
    pre = "realised_" if realised else "conferred_"
    out = {"as_of": asof.normalize(), "observed_at": asof.normalize(),
           "available_at": pd.Timestamp(avail).normalize(),
           "estimand": "realised_after_formation" if realised else "conferred_in_formation",
           "n_leg": len(win)}
    for k in X.columns:
        out[f"{pre}{k}_w"] = lw[k]
        out[f"{pre}{k}_l"] = ll[k]
        out[f"{pre}{k}_spread"] = lw[k] - ll[k]
    return out


def build(start: str = "2013-09-30", end: str = "2025-12-31",
          *, realised: bool = False) -> pd.DataFrame:
    """The panel. Rows carry their own availability stamp and estimand label."""
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    P, S = load_panel(range(a.year - 1, b.year + 2))
    if P.empty:
        return pd.DataFrame()
    P, _ = cl.correct_splits(P)
    X = _factors(ten_year_change())
    rows = []
    for d in pd.date_range(a, b, freq="ME"):
        if (P.index <= d).sum() < 260:
            continue
        r = conferred_at(d, P, S, X, realised=realised)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def percentile_of(value: float, panel: pd.DataFrame, col: str) -> float:
    """Where one date's tilt sits in the distribution the monitor has seen. NaN if unusable."""
    s = panel[col].dropna()
    return float((s < value).mean()) if len(s) >= 24 else float("nan")


# --------------------------------------------------------------------------------------
# The panel, promoted from scripts/build_conferred_panel.py
# --------------------------------------------------------------------------------------

#: The two estimands, kept apart on purpose. They are different WINDOWS and, because the
#: holdings panel is not daily throughout, they also require different estimators. The
#: formation block is a four-factor regression of the leg portfolio's returns over the twelve
#: months ending at the sort date. The post-formation block cannot use that fit: the three
#: months after a 2019 sort hold 31 panel snapshots against a forty-observation minimum, so it
#: uses the median of per-name univariate betas against the panel's own cross-sectional median,
#: which is the estimator the published post-formation figures were produced with.
ESTIMANDS = {
    "conferred_in_formation": "4-factor portfolio beta over the 12m formation window",
    "realised_after_formation": "median per-name beta vs the panel median, 3m after formation",
}


def panel(start: str = "2013-09-30", end: str = "2025-12-31",
          years: range | None = None) -> pd.DataFrame:
    """Both estimands per month-end, on one frame, each labelled.

    Promoted so the pipeline and the scripts read one implementation. The formation column is
    asserted elsewhere to reproduce data/processed/conferred_tilt.csv exactly; the promotion is
    only worth having if it is the same number.
    """
    from ..factor import placebo as pb
    from ..factor import sectors as sx

    a, b = pd.Timestamp(start), pd.Timestamp(end)
    # Load from the panel's own beginning, not from one year before the first requested date.
    # The formation gate counts SNAPSHOTS at or before a date, and this panel is monthly before
    # 2013, so a narrow load fails that gate on dates whose formation window is in fact well
    # populated: at 2013-09-30 a 2012-onward load sees 199 snapshots against a gate of 260 and
    # drops the date, while its twelve-month window holds 190 observations against a minimum of
    # 40. Three dates were lost that way and the note script kept them.
    yrs = years or range(PANEL_FIRST_YEAR, b.year + 2)
    P, S_raw = load_panel(yrs)
    if P.empty:
        return pd.DataFrame()
    P, _ = cl.correct_splits(P)
    S = sx.normalise(S_raw)
    R = P.pct_change().where(lambda r: r.abs() < cl.JUMP)
    X = _factors(ten_year_change())
    rows = []
    for d in pd.date_range(a, b, freq="ME"):
        if (P.index <= d).sum() < 260:
            continue
        book = pb.real_book(P, d, sectors=S)
        if book is None:
            continue
        form = pb.spread_for(book, R, X, d)
        if not form:
            continue
        post = pb.median_beta_spread(book, R, d, realised=True)
        rows.append({
            "asof": d.normalize(), "n_leg": form["n_w"],
            "beta_w": form["Mkt-RF_w"], "beta_l": form["Mkt-RF_l"],
            "beta_spread": form["beta_spread"],
            "d10y_spread": form["d10y_spread"], "hml_spread": form["HML_spread"],
            "smb_spread": form["SMB_spread"],
            "estimand": "conferred_in_formation",
            "available_at": form["available_at"],
            "post_beta_w": post.get("beta_w", float("nan")),
            "post_beta_l": post.get("beta_l", float("nan")),
            "post_beta_spread": post.get("beta_spread", float("nan")),
            "post_estimand": "realised_after_formation",
            "post_available_at": post.get("available_at", pd.NaT),
            "post_n_obs": post.get("n_obs", 0),
        })
    return pd.DataFrame(rows)
