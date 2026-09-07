"""Book-specific earnings calendar from SEC 8-K Item 2.02 acceptance timestamps.

Why this exists, and why it is not a repeat of the FOMC refutation
------------------------------------------------------------------
Refutation R2 killed the scheduled-catalyst hypothesis using FOMC dates: across a 3x3 grid
of horizons and thresholds, median lift 1.13x and zero cells surviving Benjamini-Hochberg.
But the diagnosis matters more than the verdict. **FOMC dates are common to every book.**
Every portfolio in the market faces the same meeting on the same day, so the variable
cannot discriminate between a fragile book and a robust one — it has no cross-sectional
content at all.

Earnings dates of the *actual holdings* do. A week in which the momentum long leg reports
heavily and the short leg does not is a one-sided risk week for that specific book, and no
common calendar can express it. This is the same hypothesis family tested with the same
machinery on a variable that can actually vary across books — which is what makes the pair
informative rather than a second bite.

Source
------
``data.sec.gov/submissions/CIK##########.json`` carries, per filing: ``form``, ``items``
(8-K item codes), and ``acceptanceDateTime`` to the second. Item **2.02** is "Results of
Operations and Financial Condition" — the earnings release. Verified on MSFT: 8-Ks tagged
``2.02,9.01`` at 20:04 UTC on 2026-07-29, 2026-04-29, 2026-01-28, i.e. quarterly, after
the close.

Point-in-time
-------------
Two distinct facts with different availability, and conflating them would be lookahead:

* **That a company will report** in the next N days is knowable in advance in practice
  (companies pre-announce dates), but EDGAR only records the filing after the fact. This
  module therefore builds the *realised* calendar and uses it to construct a forward
  density; the memo must state that a live system would source scheduled dates from an
  investor-relations feed, and that using realised dates as a proxy for scheduled dates is
  an approximation which is optimistic by roughly the pre-announcement lead time.
* **The content** of a release is available only at ``acceptanceDateTime``, and filings
  accepted after 17:30 ET are deemed filed the next business day.

The density feature uses only the first fact. No release content is read.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import RAW, USER_AGENT

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

#: 8-K item code for "Results of Operations and Financial Condition".
EARNINGS_ITEM = "2.02"

#: SEC fair-access limit is 10 requests/second across all endpoints, and a descriptive
#: User-Agent is mandatory (requests without one get 403).
_MIN_INTERVAL = 0.12
_last = 0.0


def _throttle() -> None:
    global _last
    wait = _MIN_INTERVAL - (time.monotonic() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=20),
    retry=retry_if_exception_type(requests.RequestException),
)
def _get_json(url: str) -> dict:
    _throttle()
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
    resp.raise_for_status()
    return resp.json()


def _dir():
    d = RAW / "edgar"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ticker_cik_map(*, refresh: bool = False) -> pd.DataFrame:
    """Ticker -> CIK, from the SEC's own mapping file."""
    p = _dir() / "company_tickers.json"
    if refresh or not p.exists():
        p.write_text(json.dumps(_get_json(TICKER_MAP_URL)), encoding="utf-8")
    raw = json.loads(p.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw.values())
    return df.rename(columns={"cik_str": "cik", "title": "name"}).drop_duplicates("ticker")


def earnings_dates(ticker: str, cik: int, *, refresh: bool = False) -> pd.DataFrame:
    """All 8-K Item 2.02 acceptance timestamps for one company.

    Only the ``recent`` block is read. It holds ~1,000 filings, which at a typical 15-25
    filings a year covers several decades — ample for a 2013+ panel. Companies exceeding
    it expose older blocks under ``filings.files``; those are not fetched, and the
    resulting truncation would show up as missing early history rather than wrong dates.
    """
    p = _dir() / f"sub_{cik:010d}.json"
    if refresh or not p.exists():
        try:
            p.write_text(json.dumps(_get_json(SUBMISSIONS_URL.format(cik=cik))), encoding="utf-8")
        except (requests.RequestException, ValueError):
            return pd.DataFrame(columns=["ticker", "accepted_at", "filing_date", "items"])

    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        r = d["filings"]["recent"]
    except (KeyError, ValueError):
        return pd.DataFrame(columns=["ticker", "accepted_at", "filing_date", "items"])

    n = len(r.get("form", []))
    items = r.get("items", [""] * n)
    rows = [
        {
            "ticker": ticker,
            "accepted_at": r["acceptanceDateTime"][i],
            "filing_date": r["filingDate"][i],
            "items": items[i] if i < len(items) else "",
        }
        for i in range(n)
        if r["form"][i] == "8-K" and EARNINGS_ITEM in (items[i] if i < len(items) else "")
    ]
    out = pd.DataFrame(rows)
    if len(out):
        out["accepted_at"] = pd.to_datetime(out["accepted_at"], errors="coerce", utc=True)
        out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce")
    return out


def build_calendar(tickers: list[str], *, refresh: bool = False) -> pd.DataFrame:
    """Earnings calendar for a ticker universe. Resumable; caches one file per company."""
    m = ticker_cik_map().set_index("ticker")["cik"].to_dict()
    frames = []
    for t in tickers:
        cik = m.get(t)
        if cik is None:
            continue
        df = earnings_dates(t, int(cik), refresh=refresh)
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "accepted_at", "filing_date", "items"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "filing_date"])


def save_calendar(cal: pd.DataFrame, name: str = "earnings_calendar") -> None:
    cal.to_parquet(_dir() / f"{name}.parquet", index=False)


def load_calendar(name: str = "earnings_calendar") -> pd.DataFrame:
    return pd.read_parquet(_dir() / f"{name}.parquet")


def forward_density(
    calendar: pd.DataFrame,
    members: pd.Index | list[str],
    as_of: pd.Timestamp,
    *,
    horizon: int = 10,
) -> float:
    """Fraction of a leg's names reporting within ``horizon`` calendar days of ``as_of``.

    Count-based rather than weight-based deliberately: the project has now twice found
    that weight-weighting destroys a signal that is present by count (the SSR U-shape
    collapsed from a 7.8x lift to 3.17%/0.00% of leg dollars). Both views belong in the
    monitor; the count view is the one used for the statistical test.
    """
    if not len(calendar) or not len(members):
        return float("nan")
    lo, hi = as_of, as_of + pd.Timedelta(days=horizon)
    win = calendar[(calendar["filing_date"] > lo) & (calendar["filing_date"] <= hi)]
    reporting = set(win["ticker"]) & set(members)
    return len(reporting) / len(members)


def density_series(
    calendar: pd.DataFrame,
    leg_members: dict[pd.Timestamp, dict[str, list[str]]],
    *,
    horizon: int = 10,
) -> pd.DataFrame:
    """Forward earnings density per leg at each formation date, plus the spread.

    ``spread`` is winner-leg minus loser-leg density. That is the genuinely book-specific
    quantity: a week where the winners report and the losers do not is one-sided risk, and
    it is exactly what a market-wide calendar cannot represent.
    """
    rows = []
    for as_of, legs in sorted(leg_members.items()):
        w = forward_density(calendar, legs.get("winners", []), as_of, horizon=horizon)
        l = forward_density(calendar, legs.get("losers", []), as_of, horizon=horizon)
        rows.append(
            {
                "as_of": as_of,
                "density_winners": w,
                "density_losers": l,
                "density_spread": w - l,
                "density_both": (w + l) / 2 if pd.notna(w) and pd.notna(l) else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("as_of").sort_index()
