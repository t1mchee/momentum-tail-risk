"""Cboe index history and options volume ratios.

Free, no key, no auth, no bot gate -- the best history/latency/cost tradeoff of any
source in this project. Data is available at roughly T+1.

Two things to know before using any of it:

* **Dates are ``MM/DD/YYYY``**, unlike French (``YYYYMMDD``) and FRED (ISO). Three
  sources, three date formats.
* **Several series start too late to see the crashes that matter.** VIX3M begins
  2009-09-18, so the VIX term-structure signal is blind to the 2008-09 momentum crash --
  the single largest tail event in the modern WML record. DSPX begins 2014-06. Any
  feature built on these must be missing-aware rather than silently backfilled, or the
  model will "learn" that crashes only happen post-2009.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from .. import pit
from ..config import RAW, USER_AGENT

INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
PC_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{name}pc.csv"

#: name -> first available date, as verified 2026-08-21. Used to assert that a series
#: really does start where we think, so a silent upstream change is caught.
#:
#: NOTE, and it corrects an earlier claim in this project: substantial free historical
#: IMPLIED-volatility data does exist here, including implied CORRELATION back to 2006 and
#: single-stock implied vol back to 2011. The project previously asserted "no free
#: historical single-stock IV exists" on the strength of a survey summary, without probing
#: the endpoint. Probing it took one command.
INDEX_START: dict[str, str] = {
    # --- volatility
    "VIX": "1990-01-02",
    "VIX9D": None,
    "VIX3M": "2009-09-18",
    "VIX6M": None,
    "VVIX": "2006-03-06",
    "SKEW": None,
    "DSPX": "2014-06-19",
    # --- implied correlation, the forward-looking crowding-adjacent series
    "COR1M": "2006-01-03",
    "COR3M": "2006-01-03",
    "ICJ": "2008-06-10",   # legacy Cboe implied correlation, ends 2021-11-19
    "JCJ": "2008-06-04",   # legacy, ends 2021-11-19
    "KCJ": "2009-11-24",   # legacy, ends 2021-11-19
    # --- single-stock implied vol (only five names, all megacaps)
    "VXAPL": "2011-01-07",
    "VXAZN": "2011-01-07",
    "VXGOG": "2011-01-07",
    "VXIBM": "2011-01-07",
    "VXGS": "2011-01-07",
    # --- size / style / cross-asset implied vol
    "VXN": "2009-09-14",    # Nasdaq-100
    "RVX": "2009-09-16",    # Russell 2000 — the small-cap tilt of the short leg
    "VXD": "2009-09-18",    # Dow
    "VXTLT": "2004-01-02",  # long bonds — the duration axis
    "VXEEM": "2011-03-16",
    "VXGDX": "2011-03-16",
    "VXSLV": "2011-03-16",
    "EVZ": "2009-09-18",
}

#: Series that have been discontinued upstream. Downloading them still works but the
#: history stops, so a naive forward-fill would carry a stale value indefinitely.
DISCONTINUED: dict[str, str] = {
    "ICJ": "2021-11-19",
    "JCJ": "2021-11-19",
    "KCJ": "2021-11-19",
    "VXXLE": "2022-02-11",
    "OIV": "2022-11-07",
    "EVZ": "2025-03-11",
}

#: Cboe changed methodology twice in 2012: index volume switched from OCC *cleared* to
#: *preliminary reported* volume after 2012-05-31, and equity volume began excluding ETP
#: volume on 2012-06-11. Do not z-score across these breaks.
PC_BREAKS = (pd.Timestamp("2012-05-31"), pd.Timestamp("2012-06-11"))


def _fetch(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=90)
    resp.raise_for_status()
    return resp.text


def _cache(kind: str, name: str, text: str):
    d = RAW / "cboe" / kind
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.csv").write_text(text, encoding="utf-8")


def index(name: str = "VIX", *, refresh: bool = False) -> pd.DataFrame:
    """Load one Cboe index history, indexed by date.

    Returns whatever columns the file carries -- OHLC for the VIX family and correlation
    indices, close-only for DSPX/VVIX/SKEW.
    """
    path = RAW / "cboe" / "index" / f"{name}.csv"
    if refresh or not path.exists():
        text = _fetch(INDEX_URL.format(name=name))
        _cache("index", name, text)
    else:
        text = path.read_text(encoding="utf-8")

    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    date_col = next(c for c in df.columns if "DATE" in c)
    df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"

    expected = INDEX_START.get(name)
    if expected and df.index.min() > pd.Timestamp(expected) + pd.Timedelta(days=5):
        raise ValueError(
            f"{name} starts {df.index.min().date()}, expected ~{expected}. "
            "Upstream history may have been truncated."
        )
    return df.apply(pd.to_numeric, errors="coerce")


def close(name: str = "VIX", *, refresh: bool = False) -> pd.Series:
    """Close-level series for one index, whatever the file calls that column."""
    df = index(name, refresh=refresh)
    col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
    return df[col].rename(name.lower()).dropna()


def term_structure(*, refresh: bool = False) -> pd.DataFrame:
    """VIX curve slope -- the classic stress/complacency gauge.

    ``vix3m/vix`` below 1 is backwardation: near-term fear exceeds medium-term, which is
    the shape associated with acute deleveraging. Starts 2009-09 and is therefore blind
    to 2008; left as NaN before then rather than backfilled.
    """
    vix = close("VIX", refresh=refresh)
    vix3m = close("VIX3M", refresh=refresh)
    df = pd.concat([vix, vix3m], axis=1)
    df["ts_ratio"] = df["vix3m"] / df["vix"]
    df["backwardation"] = df["ts_ratio"] < 1.0
    return df


def put_call(name: str = "equity", *, refresh: bool = False) -> pd.DataFrame:
    """Cboe put/call volume ratio. ``equity`` is the positioning-relevant one.

    Index P/C is dominated by institutional hedging flow and says more about index
    overlay demand than about single-stock positioning. Adds a ``regime`` column marking
    the 2012 methodology breaks so downstream code cannot accidentally normalise across
    them.
    """
    path = RAW / "cboe" / "pc" / f"{name}.csv"
    if refresh or not path.exists():
        text = _fetch(PC_URL.format(name=name))
        _cache("pc", name, text)
    else:
        text = path.read_text(encoding="utf-8")

    # A few lines of preamble precede the header on these files.
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.upper().startswith("DATE"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [c.strip().upper().replace("P/C RATIO", "PC_RATIO") for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["DATE"]).set_index("DATE").sort_index()
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")

    df["regime"] = 0
    for i, b in enumerate(PC_BREAKS, start=1):
        df.loc[df.index > b, "regime"] = i

    # These files are ABANDONED, not merely lagged: the equity ratio stops on
    # 2019-10-04. The download succeeds, the parse succeeds, and the series just ends.
    # Forward-filling it onto a 2026 calendar would silently carry a 2019 value for
    # seven years. Warn loudly; downstream code must treat it as missing after the end.
    staleness = (pd.Timestamp.today().normalize() - df.index.max()).days
    if staleness > 30:
        import warnings

        warnings.warn(
            f"Cboe {name} put/call ends {df.index.max().date()} ({staleness} days stale) "
            f"-- this file is no longer maintained. Usable for the DESIGN tier (2007, "
            f"2019) but NOT available live. Do not forward-fill past the end date.",
            stacklevel=2,
        )
    return df


def panel(*, refresh: bool = False) -> pd.DataFrame:
    """Assemble the Cboe market-stress block used by the feature layer.

    Point-in-time note: these are close-of-day values published the following morning, so
    ``available_at`` is the next business day. Missing early history is left as NaN.
    """
    out = pd.concat(
        [close(n, refresh=refresh) for n in ("VIX", "VIX3M", "VVIX", "SKEW", "COR1M")],
        axis=1,
    )
    out["vix_ts"] = out["vix3m"] / out["vix"]
    pc = put_call("equity", refresh=refresh)
    if "PC_RATIO" in pc.columns:
        out["equity_pc"] = pc["PC_RATIO"]
    return out.sort_index()


def panel_pit(*, refresh: bool = False) -> pd.DataFrame:
    """`panel()` with point-in-time stamps (available the next business day)."""
    df = panel(refresh=refresh).reset_index()
    avail = df["date"].apply(lambda d: pit.us_business_days(d, 1))
    return pit.stamp(df, observed_at="date", available_at=avail, name="cboe:panel")


# --------------------------------------------------------------------------------------
# Forward-looking implied panel
# --------------------------------------------------------------------------------------

#: Series assembled into the implied channel. Chosen for mechanism, not availability:
#: implied correlation is the crowding-adjacent construct, RVX/VIX is the size tilt of the
#: momentum short leg, VXN/VIX is growth concentration, and VXTLT is the duration axis
#: that drove Sept 2019.
IMPLIED_SERIES: tuple[str, ...] = ("COR1M", "COR3M", "VIX", "VIX3M", "RVX", "VXN", "VXTLT")


def implied_panel(*, refresh: bool = False) -> pd.DataFrame:
    """Forward-looking implied-vol and implied-correlation state variables.

    MEASURED LIMITATION, and the reason this panel is reported rather than relied on:
    index-level implied correlation does not see crowding concentrated in a *subset* of
    names. On 2019-09-06, COR1M sat at the **26th percentile** of its 2006-2019 history
    while the momentum book's own realised comomentum was at the **87th**. Adding this
    panel to the severity model fails the same gate every other channel fails -- pinball
    0.00409 against a vol-scaled 0.00372 over 102 refits.

    That is a specific, testable reason to want *basket-level* implied correlation (MTUM
    iv30 against its constituents' iv30), which has no free history and is therefore a
    forward-capture-only component. It is a far stronger argument than an absence claim.
    """
    cols = {}
    for name in IMPLIED_SERIES:
        try:
            s = close(name, refresh=refresh)
        except Exception:  # noqa: BLE001 - a discontinued or renamed series must not abort
            continue
        end = DISCONTINUED.get(name)
        if end:
            s = s.loc[: pd.Timestamp(end)]
        cols[name.lower()] = s

    out = pd.DataFrame(cols).sort_index()
    if {"cor1m", "cor3m"}.issubset(out.columns):
        # Backwardation in implied correlation: near-term dislocation priced above
        # medium-term, the correlation analogue of an inverted VIX curve.
        out["cor_term_structure"] = out["cor3m"] / out["cor1m"]
    if {"rvx", "vix"}.issubset(out.columns):
        out["rvx_vix"] = out["rvx"] / out["vix"]
    if {"vxn", "vix"}.issubset(out.columns):
        out["vxn_vix"] = out["vxn"] / out["vix"]
    if {"vix3m", "vix"}.issubset(out.columns):
        out["vix_term_structure"] = out["vix3m"] / out["vix"]
    return out


def implied_panel_pit(*, refresh: bool = False) -> pd.DataFrame:
    """`implied_panel()` with point-in-time stamps (available next business day)."""
    df = implied_panel(refresh=refresh).reset_index()
    avail = df["date"].apply(lambda d: pit.us_business_days(d, 1))
    return pit.stamp(df, observed_at="date", available_at=avail, name="cboe:implied")
