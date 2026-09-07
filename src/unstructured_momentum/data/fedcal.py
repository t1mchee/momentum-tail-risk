"""FOMC statement calendar: the scheduled-catalyst hypothesis.

Why this exists
---------------
News coverage of factor unwinds is reactive -- the Sept 2019 evidence pack is dominated by
articles published *after* the unwind. But a reversal does not need to be forecast from
narrative if its *trigger* is on a calendar. Two of the sharpest momentum reversals in the
record were policy events:

* **2001-01-03** -- an unscheduled intermeeting cut. The single worst vol-standardized
  momentum move in the modern sample (-11.65 sigma across all twelve constructions), with
  the market flat over the window.
* **2022-11-10** -- a CPI print, and one of the ten worst single days in a century of WML.

If reversals cluster on policy dates, then a *catalyst calendar* is genuinely
forward-looking information: it is known weeks ahead, it is not a forecast, and it tells a
PM when the distribution of outcomes widens. That is a far more defensible claim than
"news sentiment predicts crashes", and it is testable.

Source
------
Statement dates are recoverable from the Federal Reserve's own URL scheme,
``/newsevents/pressreleases/monetary{YYYYMMDD}a.htm``, linked from the annual FOMC
calendar pages. Scheduled meetings and unscheduled intermeeting actions both appear, and
the distinction matters: an unscheduled action is by definition a surprise, so it is
*not* forward-looking information and must be excluded when testing whether a known
calendar helps. `scheduled_only()` enforces that.

Statements are released at **14:00 ET**, so the tradable reaction begins that afternoon.
"""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .. import pit
from ..config import RAW, USER_AGENT

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"

#: The Fed has used three different URL schemes for FOMC materials, and each covers a
#: different era. Scraping only the modern one silently truncates the calendar to 2011+,
#: which would drop every episode this project cares about before 2011.
#:
#:   2011+        /newsevents/pressreleases/monetary{YYYYMMDD}a.htm
#:   2008-2010    /monetarypolicy/files/fomcminutes{YYYYMMDD}.pdf
#:   2007         /fomc/minutes/{YYYYMMDD}.htm
#:   pre-2006     /boarddocs/press/{kind}/{YYYY}/{YYYYMMDD}/
#:
#: Four schemes across four eras. Handling only the modern one truncates the calendar to
#: 2011+; handling three still leaves a 2008-2010 hole that happens to contain the Mar 2009
#: design episode. Beige Book paths also carry dates and are deliberately not matched.
#:
#: Minutes are published three weeks after the meeting but are *named* for the meeting
#: date, so the date extracted is the statement date, not the publication date.
_STATEMENT_PATTERNS = (
    re.compile(r"monetary(\d{8})a\.htm"),
    re.compile(r"/monetarypolicy/files/fomcminutes(\d{8})\.pdf"),
    re.compile(r"/fomc/minutes/(\d{8})\.htm"),
    re.compile(r"/boarddocs/press/[a-z]+/\d{4}/(\d{8})"),
)

#: FOMC statements are released at 14:00 ET.
RELEASE_HOUR, RELEASE_MINUTE = 14, 0

#: Statement-date coverage begins here; earlier decisions were not announced this way.
COVERAGE_START = dt.date(1994, 2, 4)


class PageMissing(RuntimeError):
    """Calendar page does not exist (the Fed rolls historical pages off)."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=20),
    retry=retry_if_exception_type(requests.ConnectionError),
)
def _get(url: str) -> str:
    """Fetch a calendar page.

    Retries only on connection errors. A 404 is a permanent fact about the Fed's site --
    annual ``fomchistorical{year}`` pages are rolled off after roughly five years, with
    recent years living only on the main calendars page -- so retrying one wastes three
    round-trips to reach the same answer.
    """
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    if resp.status_code == 404:
        raise PageMissing(url)
    resp.raise_for_status()
    return resp.text


def _cache_path():
    d = RAW / "fed"
    d.mkdir(parents=True, exist_ok=True)
    return d / "fomc_statement_dates.parquet"


def scrape(start_year: int = 1994, end_year: int | None = None) -> pd.DataFrame:
    """Collect every FOMC statement date from the Fed's calendar pages."""
    end_year = end_year or dt.date.today().year
    found: set[str] = set()

    for page in [CALENDAR_URL] + [
        HISTORICAL_URL.format(year=y) for y in range(start_year, end_year + 1)
    ]:
        try:
            html = _get(page)
            for rx in _STATEMENT_PATTERNS:
                found.update(rx.findall(html))
        except (PageMissing, requests.RequestException, Exception):
            continue

    dates = sorted({dt.datetime.strptime(s, "%Y%m%d").date() for s in found})

    # Two-day meetings surface as consecutive dates (1999-06-29/30, 2012-10-23/24). The
    # statement is released on the final day, so collapse runs of adjacent dates to their
    # last. Left in, they would double-count meetings and, worse, be mistaken for
    # unscheduled intermeeting actions by the gap heuristic below.
    collapsed = [
        d for i, d in enumerate(dates) if i + 1 == len(dates) or (dates[i + 1] - d).days > 1
    ]

    df = pd.DataFrame({"date": pd.to_datetime(collapsed)})
    df["release_at"] = df["date"] + pd.Timedelta(hours=RELEASE_HOUR, minutes=RELEASE_MINUTE)
    return df


def _flag_unscheduled(df: pd.DataFrame) -> pd.DataFrame:
    """Mark statements that were almost certainly unscheduled intermeeting actions.

    The FOMC meets eight times a year at roughly six-week intervals. A statement within
    21 days of another is therefore very unlikely to be a regular meeting, which picks up
    the intermeeting cuts (Jan 2001, Sept 2001, Aug 2007, Jan 2008, Mar 2020) that matter
    most here. Heuristic and imperfect -- it is used only to *exclude* dates from the
    forward-looking test, so erring toward flagging is the safe direction.
    """
    out = df.sort_values("date").copy()
    gap_prev = out["date"].diff().dt.days
    gap_next = out["date"].diff(-1).dt.days.abs()
    out["unscheduled"] = (gap_prev < 21) | (gap_next < 21)
    # An eight-meeting year has ~45-day gaps; the first row has no predecessor.
    out.loc[out.index[0], "unscheduled"] = False
    return out


def load(*, refresh: bool = False) -> pd.DataFrame:
    """FOMC statement dates with release timestamps, cached locally."""
    p = _cache_path()
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    df = _flag_unscheduled(scrape())
    df.to_parquet(p, index=False)
    return df


def scheduled_only(*, refresh: bool = False) -> pd.DataFrame:
    """Only pre-announced meetings -- the genuinely forward-looking subset.

    Unscheduled actions are surprises by construction. Including them in a test of whether
    a *known calendar* helps would be circular: it would credit the calendar with
    predicting events the calendar did not contain.
    """
    df = load(refresh=refresh)
    return df[~df["unscheduled"]].reset_index(drop=True)


def load_pit(*, refresh: bool = False) -> pd.DataFrame:
    """Statement dates stamped point-in-time.

    ``available_at`` is the 14:00 ET release, so any feature reading the *content* of a
    statement is correctly barred until then. The *date* of a scheduled meeting is known
    far earlier; that asymmetry is the whole point of `days_to_next_scheduled`.
    """
    df = load(refresh=refresh)
    return pit.stamp(df, observed_at="date", available_at="release_at", name="fomc")


def days_to_next_scheduled(dates: pd.DatetimeIndex, *, refresh: bool = False) -> pd.Series:
    """Trading-calendar-agnostic days until the next scheduled FOMC statement.

    This is legitimately forward-looking with no leakage: the meeting schedule is published
    roughly a year ahead, so on any given day a PM genuinely knows how far away the next
    decision is.
    """
    sched = pd.DatetimeIndex(scheduled_only(refresh=refresh)["date"])
    idx = pd.DatetimeIndex(dates)
    pos = sched.searchsorted(idx, side="left")
    nxt = pd.Series(
        [sched[p] if p < len(sched) else pd.NaT for p in pos], index=idx, dtype="datetime64[ns]"
    )
    return (nxt - idx.to_series()).dt.days.rename("days_to_fomc")


def near_any(dates, window_days: int = 3, *, scheduled: bool = False) -> pd.Series:
    """Whether each date falls within ``window_days`` of an FOMC statement."""
    cal = pd.DatetimeIndex((scheduled_only() if scheduled else load())["date"])
    idx = pd.DatetimeIndex(dates)
    out = []
    for d in idx:
        deltas = (cal - d).days if hasattr(cal - d, "days") else (cal - d).to_series().dt.days
        out.append(bool((abs(pd.Series(deltas)) <= window_days).any()))
    return pd.Series(out, index=idx, name=f"near_fomc_{window_days}d")
