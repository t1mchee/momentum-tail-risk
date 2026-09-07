"""Short-side constraint: daily short volume and Reg SHO threshold lists.

Why these two, and why now
--------------------------
Free stock-level securities-lending data does not exist and will not for years. The SEC's
exemptive order of 3 Dec 2025 pushed Rule 10c-1a public dissemination (FINRA SLATE) to
**29 March 2029** and Form SHO to **2 Jan 2028**. So borrow fees and utilisation are
unavailable for the foreseeable future, and these proxies are a permanent part of the
architecture rather than a stopgap.

Both sources are **perishable**:

* FINRA's short-volume CDN holds roughly a trailing 8 years and silently drops older
  files (2018-08-01 returns 200; 2018-07-02 returns 403). Nominal history starts 2009 but
  is not obtainable from FINRA today.
* Nasdaq's threshold lists have short retention and are not archived anywhere public.

Three caveats that change how the short-volume data must be used
-----------------------------------------------------------------
1. **Volumes are fractional now.** A real row from 2026-08-19::

       20260819|A|656799.067061|7|1138358.086029|B,Q,N

   Retail fractional-share flow is in the file. An integer parser truncates silently, and
   pre-2021 vs post-2021 ratios compare different populations -- fractional flow
   concentrates in exactly the large-cap names momentum holds.

2. **"Short volume" is a trade-reporting attribute, not positioning.** A market maker
   selling to a buyer it must later cover flags the sale short, which puts a structural
   floor around 40-50% for liquid names. Use it only in *differences against each stock's
   own trailing baseline*, never in levels and never cross-sectionally.

3. **The denominator moves.** These files cover TRF/ADF/ORF (off-exchange) prints only, so
   as off-exchange share of total volume drifts, the ratio drifts with it for reasons
   having nothing to do with shorting.

``short_exempt_volume`` is a separate and much better-behaved column: it spikes on Rule
201 circuit-breaker names and is a genuine distress marker. Almost nobody uses it.

Threshold-list membership is the sharpest of the three signals: it requires fails >= 10,000
shares AND >= 0.5% of shares outstanding for five consecutive settlement days, and it
triggers a mandatory T+13 buy-in. That buy-in is a *forced covering event* -- the closest
thing to a free real-time squeeze-pressure indicator that exists.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .. import pit
from ..config import RAW, USER_AGENT

SHORT_VOL_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
THRESHOLD_URL = "https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{ymd}.txt"

#: Earliest date the FINRA CDN still serves, verified by binary search 2026-08-21.
#: This moves forward over time as old files are dropped.
SHORT_VOL_CDN_FLOOR = dt.date(2018, 8, 1)


class NoDataForDate(RuntimeError):
    """No file published for this date (weekend, holiday, or aged off the CDN)."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=15),
    retry=retry_if_exception_type(requests.RequestException),
)
def _get(url: str, session: requests.Session | None = None) -> str:
    s = session or requests.Session()
    resp = s.get(url, headers={"User-Agent": USER_AGENT}, timeout=45, allow_redirects=False)
    # Nasdaq 302s to a 404 page rather than returning 404 for missing dates.
    if resp.status_code in (302, 403, 404):
        raise NoDataForDate(f"{url} -> HTTP {resp.status_code}")
    resp.raise_for_status()
    return resp.text


def _dir(kind: str):
    d = RAW / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store(kind: str, day: dt.date, text: str) -> None:
    with gzip.open(_dir(kind) / f"{day:%Y%m%d}.txt.gz", "wt", encoding="utf-8") as fh:
        fh.write(text)


def _load_stored(kind: str, day: dt.date) -> str | None:
    p = _dir(kind) / f"{day:%Y%m%d}.txt.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------------------
# FINRA daily short volume
# --------------------------------------------------------------------------------------


def parse_short_volume(text: str) -> pd.DataFrame:
    """Parse a consolidated short-volume file. Volumes are floats, not ints."""
    df = pd.read_csv(io.StringIO(text), sep="|")
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns or "ShortVolume" not in df.columns:
        raise NoDataForDate("unrecognised short-volume schema")

    df = df.rename(
        columns={
            "Date": "date",
            "Symbol": "ticker",
            "ShortVolume": "short_volume",
            "ShortExemptVolume": "short_exempt_volume",
            "TotalVolume": "total_volume",
            "Market": "market",
        }
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    for c in ("short_volume", "short_exempt_volume", "total_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "ticker"])
    df["short_ratio"] = df["short_volume"] / df["total_volume"].replace(0, pd.NA)
    df["exempt_ratio"] = df["short_exempt_volume"] / df["total_volume"].replace(0, pd.NA)
    return df


def short_volume(day: dt.date | str, *, session=None, use_cache: bool = True) -> pd.DataFrame:
    """One day of consolidated short volume, cached locally on first fetch."""
    d = pd.Timestamp(day).date()
    text = _load_stored("finra_shortvol", d) if use_cache else None
    if text is None:
        text = _get(SHORT_VOL_URL.format(ymd=f"{d:%Y%m%d}"), session)
        _store("finra_shortvol", d, text)
    return parse_short_volume(text)


# --------------------------------------------------------------------------------------
# Reg SHO threshold lists
# --------------------------------------------------------------------------------------


def parse_threshold(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="|")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security Name": "name",
            "Market Category": "market_category",
            "Reg SHO Threshold Flag": "threshold_flag",
        }
    )
    return df.dropna(subset=["ticker"])[
        [c for c in ["ticker", "name", "market_category", "threshold_flag"] if c in df]
    ]


def threshold_list(day: dt.date | str, *, session=None, use_cache: bool = True) -> pd.DataFrame:
    """Nasdaq Reg SHO threshold securities for one date."""
    d = pd.Timestamp(day).date()
    text = _load_stored("regsho_threshold", d) if use_cache else None
    if text is None:
        text = _get(THRESHOLD_URL.format(ymd=f"{d:%Y%m%d}"), session)
        _store("regsho_threshold", d, text)
    out = parse_threshold(text)
    out["date"] = pd.Timestamp(d)
    return out


# --------------------------------------------------------------------------------------
# Capture and aggregation
# --------------------------------------------------------------------------------------


def capture(day: dt.date | str | None = None) -> dict[str, int]:
    """Daily job: pull and store both sources for one date."""
    d = pd.Timestamp(day).date() if day else dt.date.today()
    session = requests.Session()
    out: dict[str, int] = {}
    for label, fn in (("short_volume", short_volume), ("threshold_list", threshold_list)):
        try:
            out[label] = len(fn(d, session=session, use_cache=False))
        except NoDataForDate:
            out[label] = 0
    return out


def backfill(
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    *,
    pause: float = 0.2,
) -> pd.DataFrame:
    """Archive both series over a date range.

    Defaults to the full window the FINRA CDN still serves. Files already stored locally
    are skipped, so this is resumable and cheap to re-run.
    """
    import time

    start = pd.Timestamp(start).date() if start else SHORT_VOL_CDN_FLOOR
    end = pd.Timestamp(end).date() if end else dt.date.today()
    session = requests.Session()

    rows = []
    for day in pd.bdate_range(start, end):
        d = day.date()
        got = {"date": day, "short_volume": 0, "threshold": 0}
        for key, kind, fn in (
            ("short_volume", "finra_shortvol", short_volume),
            ("threshold", "regsho_threshold", threshold_list),
        ):
            if _load_stored(kind, d) is not None:
                got[key] = -1  # already cached
                continue
            try:
                got[key] = len(fn(d, session=session, use_cache=False))
            except (NoDataForDate, requests.RequestException, ValueError):
                got[key] = 0
            time.sleep(pause)
        rows.append(got)
    return pd.DataFrame(rows)


def aggregate_over(tickers: list[str], day: dt.date | str) -> pd.Series:
    """Short-side pressure aggregated over a named basket for one date.

    ``tickers`` would typically be the momentum long or short leg. Reported as
    volume-weighted ratios plus threshold-list overlap, which is the binding-constraint
    measure rather than a flow measure.
    """
    sv = short_volume(day)
    sub = sv[sv["ticker"].isin(tickers)]
    if not len(sub):
        return pd.Series(dtype=float)

    tv = sub["total_volume"].sum()
    try:
        th = set(threshold_list(day)["ticker"])
    except NoDataForDate:
        th = set()

    return pd.Series(
        {
            "date": pd.Timestamp(day),
            "n_matched": len(sub),
            "short_ratio_vw": sub["short_volume"].sum() / tv if tv else float("nan"),
            "exempt_ratio_vw": sub["short_exempt_volume"].sum() / tv if tv else float("nan"),
            "n_on_threshold": len(th & set(tickers)),
            "threshold_share": len(th & set(tickers)) / len(tickers) if tickers else float("nan"),
        }
    )


def market_breadth(day: dt.date | str) -> pd.Series:
    """Market-wide short-side stress for one date.

    Threshold-list *count* is the headline: a rising count means more names are hitting
    the mandatory-buy-in constraint at once, which is the systemic version of the squeeze
    mechanism rather than the single-name version.
    """
    sv = short_volume(day)
    try:
        n_th = len(threshold_list(day))
    except NoDataForDate:
        n_th = 0
    tv = sv["total_volume"].sum()
    return pd.Series(
        {
            "date": pd.Timestamp(day),
            "n_symbols": len(sv),
            "short_ratio_mkt": sv["short_volume"].sum() / tv if tv else float("nan"),
            "exempt_ratio_mkt": sv["short_exempt_volume"].sum() / tv if tv else float("nan"),
            "n_threshold_securities": n_th,
        }
    )


def load_breadth_pit(days: list[dt.date] | None = None) -> pd.DataFrame:
    """Market-breadth series over whatever dates are cached locally, PIT-stamped.

    Short-volume files post after the close, so ``available_at`` is the next business day.
    """
    stored = sorted(_dir("finra_shortvol").glob("*.txt.gz"))
    dates = days or [dt.datetime.strptime(p.stem.split(".")[0], "%Y%m%d").date() for p in stored]
    rows = [market_breadth(d) for d in dates]
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    avail = df["date"].apply(lambda d: pit.us_business_days(d, 1))
    return pit.stamp(df, observed_at="date", available_at=avail, name="finra:shortvol")
