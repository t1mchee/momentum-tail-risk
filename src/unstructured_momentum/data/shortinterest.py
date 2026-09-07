"""FINRA consolidated short interest: the size of the short position, point-in-time.

Why this source
---------------
The 13F work established that institutional long holdings cannot distinguish the momentum
legs, and said why: Form 13F reports long US equity and nothing else, so it never sees the
leg momentum is short or the leverage that turns crowding into forced selling. This is the
complement. Short interest is the position itself, reported twice a month for every US
equity security, and ``daysToCoverQuantity`` -- short interest over average daily volume --
is the mechanism stated as a number, because it is literally how long the crowd needs to get
out.

The availability trap, which is the whole reason this is a module
----------------------------------------------------------------
The API returns ``settlementDate`` and no publication date at all. Settlement is when the
position was MEASURED. FINRA disseminates roughly eight business days later, so a monitor
using settlement dates would hold positions more than a week before anyone could see them,
and any signal built that way would look predictive because it was reading the future. This
is the same fault that shifted an entire 13F panel one quarter early, and it is easier to
make here because the field is right there and looks like a date you can use.

So availability is computed as settlement plus ten business days -- deliberately LATER than
FINRA's stated lag. An availability stamp that is too late costs a little signal; one that
is too early invents it. Only one of those errors is recoverable.

What the numbers do and do not mean
-----------------------------------
* ``currentShortPositionQuantity`` is shares short. Divided by shares outstanding it is the
  usual short-interest ratio, but shares outstanding is not in this feed and has to come
  from elsewhere.
* ``daysToCoverQuantity`` is FINRA's own short-interest-over-volume figure. It saturates at
  999.99 for securities with no reported volume, which is a sentinel and not a number; those
  rows are dropped rather than averaged.
* This is a POSITION, unlike the daily short-volume feed in ``shortside.py``, which is FLOW
  and is contaminated by market-maker hedging. The two are easy to confuse and answer
  different questions.
* Short interest is reported by broker-dealers on a settlement basis and nets nothing across
  accounts, so it is a gross figure.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

USER_AGENT = "unstructured-momentum research chee.timothy@gmail.com"
URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
ROOT = Path("data/raw/shortinterest")

#: The API refuses larger pages; one settlement date is roughly 16,000 securities.
PAGE = 5000

#: Earliest settlement date the API serves, established by bisection rather than assumed.
FIRST_SETTLEMENT = dt.date(2018, 1, 12)

#: Business days added to a settlement date to get the date a monitor could act on it.
#: FINRA's own lag is about eight; ten is used so the stamp errs late. A stamp that is too
#: late loses a little signal, one that is too early manufactures it.
DISSEMINATION_LAG_BDAYS = 10

#: daysToCoverQuantity takes this value when a security reports no average daily volume.
#: It is a sentinel, not a duration, and averaging it silently would put every illiquid
#: name at the top of any crowding ranking.
DTC_SENTINEL = 999.99


class NoData(Exception):
    """The API returned no rows for this settlement date."""


def available_at(settlement: dt.date | str) -> pd.Timestamp:
    """When a monitor could first act on a settlement date's short interest."""
    s = pd.Timestamp(settlement)
    return (s + pd.tseries.offsets.BDay(DISSEMINATION_LAG_BDAYS)).normalize()


def _post(body: dict, session: requests.Session | None = None) -> requests.Response:
    s = session or requests
    r = s.post(
        URL,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        data=json.dumps(body),
        timeout=120,
    )
    r.raise_for_status()
    return r


def fetch_window(start: str, end: str, *, session=None) -> pd.DataFrame:
    """Every row whose settlement date falls in a window, paged.

    Fetching by window rather than by settlement date is deliberate. An earlier version
    enumerated the distinct dates first, which meant paging through the entire feed once to
    learn what was in it and then paging through it again to retrieve it -- twice the
    requests for the same bytes. Settlement dates are recovered from the rows themselves,
    which is also the only way that survives a mid-month date moving for a holiday.
    """
    frames, offset, total = [], 0, None
    while True:
        r = _post(
            {
                "limit": PAGE,
                "offset": offset,
                "compareFilters": [
                    {"fieldName": "settlementDate", "fieldValue": start, "compareType": "GTE"},
                    {"fieldName": "settlementDate", "fieldValue": end, "compareType": "LTE"},
                ],
            },
            session=session,
        )
        if r.status_code == 204 or not r.text.strip():
            break
        frames.append(pd.read_csv(io.StringIO(r.text)))
        total = int(r.headers.get("Record-Total", 0))
        offset += PAGE
        if offset >= total:
            break
        time.sleep(0.15)
    if not frames:
        raise NoData(f"no rows for {start}..{end}")
    d = pd.concat(frames, ignore_index=True)
    if total is not None and len(d) != total:
        raise NoData(f"{start}..{end}: got {len(d)} rows, API reported {total}")
    return d


def annotate(d: pd.DataFrame) -> pd.DataFrame:
    """Attach the point-in-time stamps and the sentinel flag to raw API rows."""
    d = d.copy()
    d["settlement_date"] = pd.to_datetime(d["settlementDate"])
    d["available_at"] = d["settlement_date"].map(available_at)
    d["dtc_is_sentinel"] = d["daysToCoverQuantity"].round(2).eq(DTC_SENTINEL)
    return d


def store(d: pd.DataFrame) -> list[str]:
    """Split an annotated window by settlement date and write one file each."""
    ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    for sd, g in d.groupby("settlement_date"):
        stamp = pd.Timestamp(sd).strftime("%Y%m%d")
        g.to_parquet(ROOT / f"{stamp}.parquet")
        written.append(str(pd.Timestamp(sd).date()))
    return sorted(written)


def fetch(settlement: str, *, session=None, use_cache: bool = True) -> pd.DataFrame:
    """One settlement date, all securities, cached on first fetch.

    Carries ``available_at`` alongside ``settlementDate`` so the two can never be confused
    downstream, and a ``dtc_is_sentinel`` flag so the saturated rows are visible rather than
    quietly averaged.
    """
    out = ROOT / f"{str(settlement).replace('-', '')}.parquet"
    if use_cache and out.exists():
        return pd.read_parquet(out)

    frames, offset, total = [], 0, None
    while True:
        r = _post(
            {
                "limit": PAGE,
                "offset": offset,
                "compareFilters": [
                    {"fieldName": "settlementDate", "fieldValue": str(settlement), "compareType": "EQUAL"}
                ],
            },
            session=session,
        )
        if r.status_code == 204 or not r.text.strip():
            break
        frames.append(pd.read_csv(io.StringIO(r.text)))
        total = int(r.headers.get("Record-Total", 0))
        offset += PAGE
        if offset >= total:
            break
        time.sleep(0.2)

    if not frames:
        raise NoData(f"no rows for settlement {settlement}")
    d = pd.concat(frames, ignore_index=True)

    if total is not None and len(d) != total:
        # The API states its own record count; a short read is a paging fault, not a small
        # date, and it must not pass as one.
        raise NoData(f"{settlement}: got {len(d)} rows, API reported {total}")

    d["settlement_date"] = pd.to_datetime(d["settlementDate"])
    d["available_at"] = available_at(settlement)
    d["dtc_is_sentinel"] = d["daysToCoverQuantity"].round(2).eq(DTC_SENTINEL)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out)
    return d


def clean(d: pd.DataFrame, *, min_adv: int = 1000) -> pd.DataFrame:
    """Rows usable for a crowding statistic, with the sentinel and the illiquid removed.

    Dropping is deliberate rather than flagging. A metric that only reports has to be read
    by someone at the right moment, and that has already failed in this project: an
    implausible-value share was printed at 80 percent and then at 99.99 percent while the
    pipeline carried on, because nothing consumed the number.
    """
    ok = (
        ~d["dtc_is_sentinel"]
        & d["averageDailyVolumeQuantity"].ge(min_adv)
        & d["currentShortPositionQuantity"].gt(0)
        & d["daysToCoverQuantity"].gt(0)
    )
    return d[ok].copy()


def coverage(d: pd.DataFrame) -> dict:
    """What a pull actually contains, in the terms that have caught faults before."""
    cl = clean(d)
    tot_short = float(d["currentShortPositionQuantity"].sum())
    return {
        "rows": int(len(d)),
        "rows_usable": int(len(cl)),
        "sentinel_rows": int(d["dtc_is_sentinel"].sum()),
        "sentinel_short_share": (
            float(d.loc[d["dtc_is_sentinel"], "currentShortPositionQuantity"].sum() / tot_short)
            if tot_short
            else 0.0
        ),
        "exchanges": d["marketClassCode"].value_counts().to_dict(),
        "median_dtc": float(cl["daysToCoverQuantity"].median()) if len(cl) else float("nan"),
        "settlement_date": str(d["settlement_date"].iloc[0].date()),
        "available_at": str(d["available_at"].iloc[0].date()),
    }
