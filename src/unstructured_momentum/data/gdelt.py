"""GDELT DOC 2.0: timestamped news retrieval with point-in-time discipline.

Why GDELT
---------
Free, keyless, permissively licensed ("unlimited and unrestricted use ... without fee",
attribution only), and it carries an explicit publication-side timestamp. Of the text
sources surveyed it has the lowest ToS risk and the best time resolution in the modern era.

The timestamp is a *feature*
----------------------------
GDELT's ``seendate`` is the crawl/ingest slot, not the article's publication moment, and it
lags publication by minutes to hours. That is usually treated as a defect. Here it is
exactly right: crawl time is a conservative ``available_at``. We could not have read an
article before GDELT saw it, so using ``seendate`` cannot leak, whereas using a
publisher-claimed publication time might.

Hard coverage limits, not worked around
----------------------------------------
* The DOC 2.0 article API starts **2017-01-01**. It covers Sept 2019 and everything after,
  and covers none of Jan 2001, Aug 2007 or Mar 2009.
* ``maxrecords`` caps at **250** with no deep pagination, so this API cannot bulk-harvest
  history. Long backfills need the raw 15-minute GKG files at
  ``data.gdeltproject.org/gdeltv2/``; this module deliberately does not pretend otherwise
  and instead slices queries by time window.
* GDELT distributes metadata and URLs, never article bodies. Titles and domains are what
  we get; full text would require fetching each URL, with the copyright questions that
  implies.

Queries target the mechanism, not sentiment
--------------------------------------------
Generic news tone is weakly related to factor reversals and would dilute the signal. The
query set in `MECHANISM_QUERIES` goes after the specific things that precede an unwind:
explicit crowding talk, factor-rotation narrative, and forced-deleveraging language.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .. import pit
from ..config import CORPUS, USER_AGENT

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

#: The DOC 2.0 article index does not reach earlier than this.
COVERAGE_START = dt.date(2017, 1, 1)

#: Hard cap imposed by the API. There is no deep pagination past it.
MAX_RECORDS = 250

#: Mechanism-targeted queries. Each maps to one of the channels the system monitors.
#:
#: OR'd terms MUST be parenthesised -- GDELT rejects an unparenthesised OR with
#: "Queries containing OR'd terms must be surrounded by ()" as a plain-text 200, not a
#: JSON error, so an unwrapped query fails at parse time rather than returning nothing.
MECHANISM_QUERIES: dict[str, str] = {
    "crowding": '("crowded trade" OR "crowded positioning" OR "consensus positioning")',
    "factor_unwind": (
        '("factor unwind" OR "momentum unwind" OR "quant unwind" OR "factor rotation")'
    ),
    "deleveraging": '("deleveraging" OR "degrossing" OR "forced selling" OR "margin call")',
    "momentum_narrative": '("momentum stocks" OR "momentum trade" OR "momentum factor")',
    "rotation": '("rotation into value" OR "value rotation" OR "growth to value")',
}


#: Outlets whose factor/market commentary is written by people, and which a PM would
#: plausibly read. Not exhaustive, and deliberately conservative -- the point is to
#: measure *professional* narrative, so a source that mostly syndicates does not belong
#: here even if it occasionally carries good material.
TIER1_DOMAINS: frozenset[str] = frozenset(
    {
        "reuters.com", "in.reuters.com", "uk.reuters.com",
        "wsj.com", "ft.com", "bloomberg.com", "barrons.com",
        "marketwatch.com", "cnbc.com", "economist.com",
        "institutionalinvestor.com", "pionline.com", "risk.net",
        "hedgeweek.com", "efinancialnews.com", "investmentweek.co.uk",
        "ftadviser.com", "citywire.com", "pensionsandinvestments.com",
    }
)

#: Domains empirically dominated by templated, machine-generated filler on these queries.
#: Identified from the Sept-2019 pull: 202 of 246 pre-event "crowding" matches came from
#: news.yahoo.com alone, nearly all syndicated Simply Wall St shareholder-composition
#: pieces ("What Kind Of Investor Owns Most Of ..."), against 4 from Reuters.
#: Counting these as crowding chatter measures a publishing schedule, not a market.
BOILERPLATE_DOMAINS: frozenset[str] = frozenset(
    {"news.yahoo.com", "finance.yahoo.com", "simplywall.st", "nasdaq.com", "zacks.com"}
)

#: Title patterns for the same templated content, since the syndication also reaches
#: domains not on the denylist.
BOILERPLATE_TITLE_PATTERNS: tuple[str, ...] = (
    "what kind of investor",
    "what kind of shareholder",
    "who owns most",
    "shares do institutions own",
    "investor composition",
    "insiders own",
    "institutional investors",
)


class GdeltCoverageError(RuntimeError):
    """Requested window predates the DOC 2.0 article index."""


def classify_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each article as tier-1, boilerplate, or other.

    Kept as a tag rather than a hard filter so the discarded volume stays visible: a
    feature that silently drops 80% of its input should say so.
    """
    if not len(df):
        return df
    out = df.copy()
    dom = out["domain"].astype(str).str.lower()
    title = out["title"].astype(str).str.lower()

    is_boiler = dom.isin(BOILERPLATE_DOMAINS) | title.apply(
        lambda t: any(p in t for p in BOILERPLATE_TITLE_PATTERNS)
    )
    out["source_tier"] = "other"
    out.loc[dom.isin(TIER1_DOMAINS), "source_tier"] = "tier1"
    out.loc[is_boiler, "source_tier"] = "boilerplate"
    return out


#: GDELT's documented ceiling, quoted verbatim from its own 429 body: "Please limit
#: requests to one every 5 seconds". Enforced here rather than left to each caller,
#: because a caller who forgets does not get an error -- they get a 429 that burns the
#: budget for everyone and looks like a transient failure. Set well above the stated
#: 5s because 5s empirically still trips the limiter on consecutive queries.
MIN_REQUEST_INTERVAL = 12.0
_last_request: float = 0.0


def _throttle() -> None:
    global _last_request
    import time

    wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=10, max=90),
    retry=retry_if_exception_type(requests.RequestException),
)
def _get(params: dict) -> dict:
    _throttle()
    resp = requests.get(DOC_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    if resp.status_code == 429:
        raise requests.RequestException("rate limited by GDELT (1 req / 5s)")
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError:
        # GDELT returns a plain-text error page for malformed queries.
        raise ValueError(f"non-JSON response: {resp.text[:200]}") from None


def search(
    query: str,
    start: dt.date | str,
    end: dt.date | str,
    *,
    max_records: int = MAX_RECORDS,
    country: str | None = "US",
    language: str | None = "english",
) -> pd.DataFrame:
    """Article search over a time window.

    ``start``/``end`` are inclusive dates. Results carry ``seendate`` (crawl time), which
    becomes ``available_at``.
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s.date() < COVERAGE_START:
        raise GdeltCoverageError(
            f"GDELT DOC 2.0 begins {COVERAGE_START}; {s.date()} is not covered. "
            "For earlier episodes use EDGAR, the NYT Archive API, or BIS/IMF publications."
        )

    q = query
    if country:
        q += f" sourcecountry:{country}"
    if language:
        q += f" sourcelang:{language}"

    # Cache on disk keyed by the exact request. GDELT's limiter is far stricter in practice
    # than its stated 1-req/5s, and a research workflow re-runs the same queries constantly
    # while iterating. Caching is both politeness and reproducibility: an analysis re-run
    # months later sees the same corpus rather than a silently re-crawled one.
    key = hashlib.sha256(
        f"{q}|{s.date()}|{e.date()}|{max_records}".encode()
    ).hexdigest()[:20]
    cache_dir = CORPUS / "gdelt" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{key}.json"
    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        arts = payload.get("articles", [])
        if not arts:
            return pd.DataFrame(columns=["title", "url", "domain", "seendate", "language"])
        df = pd.DataFrame(arts)
        df["seendate"] = pd.to_datetime(
            df["seendate"], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce"
        )
        return df.dropna(subset=["seendate"]).reset_index(drop=True)

    payload = _get(
        {
            "query": q,
            "mode": "artlist",
            "format": "json",
            "maxrecords": min(max_records, MAX_RECORDS),
            "startdatetime": s.strftime("%Y%m%d000000"),
            "enddatetime": e.strftime("%Y%m%d235959"),
            "sort": "datedesc",
        }
    )

    cached.write_text(json.dumps(payload), encoding="utf-8")

    arts = payload.get("articles", [])
    if not arts:
        return pd.DataFrame(columns=["title", "url", "domain", "seendate", "language"])

    df = pd.DataFrame(arts)
    df["seendate"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    return df.dropna(subset=["seendate"]).reset_index(drop=True)


def _windows(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cur = start
    while cur <= end:
        stop = min(cur + pd.Timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + pd.Timedelta(days=1)


def harvest(
    start: dt.date | str,
    end: dt.date | str,
    *,
    queries: dict[str, str] | None = None,
    window_days: int = 3,
    pause: float = 1.5,
) -> pd.DataFrame:
    """Retrieve all mechanism queries over a period, sliced to work around the 250 cap.

    The API returns at most 250 records per call with no pagination, so a long period is
    split into short windows. ``window_days`` of 3 keeps most windows below the cap during
    normal news flow; windows that hit exactly 250 are flagged as truncated rather than
    silently under-reported.
    """
    import time

    queries = queries or MECHANISM_QUERIES
    s, e = pd.Timestamp(start), pd.Timestamp(end)

    frames = []
    for label, q in queries.items():
        for w_start, w_end in _windows(s, e, window_days):
            try:
                df = search(q, w_start, w_end)
            except (requests.RequestException, ValueError):
                continue
            if len(df):
                df["mechanism"] = label
                df["query"] = q
                df["truncated"] = len(df) >= MAX_RECORDS
                frames.append(df)
            time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["url", "mechanism"]).reset_index(drop=True)


def to_pit(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp articles for point-in-time retrieval.

    Both timestamps are the crawl time. We do not have a trustworthy publication instant,
    and inventing one by parsing the page would create an ``available_at`` earlier than the
    moment we could actually have seen the article.
    """
    if not len(df):
        return df
    out = df.copy()
    return pit.stamp(out, observed_at="seendate", available_at="seendate", name="gdelt")


def store(df: pd.DataFrame, name: str) -> None:
    """Persist a harvested corpus slice."""
    d = CORPUS / "gdelt"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{name}.parquet", index=False)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(CORPUS / "gdelt" / f"{name}.parquet")


def daily_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Article counts per mechanism per day -- the crowding-chatter feature.

    Counts are the raw construct. They are *not* normalised here because GDELT's source
    coverage grows over time, so a level comparison across years is meaningless; the
    feature layer should z-score against a trailing window rather than against history.
    """
    if not len(df):
        return pd.DataFrame()
    d = df.copy()
    d["date"] = d["seendate"].dt.tz_convert(pit.NY).dt.normalize()
    return (
        d.groupby(["date", "mechanism"]).size().unstack("mechanism").fillna(0).sort_index()
    )
