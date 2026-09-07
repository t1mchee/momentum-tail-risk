"""iShares ETF holdings: point-in-time daily history and live harvest (MTUM primary).

What this gives us
------------------
Daily holdings *and* shares outstanding for MTUM back to inception (2013-04-16), as-of any
trading date, free and without authentication. Differencing shares outstanding yields
implied creation/redemption flow -- a genuine positioning series, not a return-based proxy.

Concretely, this reconstructs what the momentum portfolio actually owned on the eve of
every modern reversal episode. Verified spot checks::

    2019-09-06  88,000,000 shares   (Friday before the Sept 9 momentum unwind)
    2020-11-06  82,450,000 shares   (Friday before vaccine day)
    2021-01-26  88,950,000 shares   (eve of the squeeze peak)

Endpoint archaeology -- read before "simplifying" any URL here
--------------------------------------------------------------
Three endpoints exist. Two are traps.

1. ``.../{fund}/1467271812596.ajax?fileType=csv&...`` -- **DEAD.** Returns the product
   page HTML for every request, including with ``asOfDate``. Verified 2026-08-21. This is
   the endpoint every older tutorial and community scraper uses, so anything built on it
   is silently broken *right now*.
2. ``.../{fund}/latest-holdings.csv`` -- works, but current snapshot only, no history.
   Kept as a fallback canary.
3. ``blackrock.com/varnish-api/.../get-fund-document?...&asOfDate=YYYYMMDD`` -- **the one
   that works.** Different host, undocumented, no SLA. Full daily history.

Three silent-failure modes, none of which a status-code check catches
---------------------------------------------------------------------
* The dead ``.ajax`` endpoint responds ``200`` with ``content-type: text/csv`` and
  ``content-disposition: attachment; filename=MTUM_holdings.csv`` -- and 1.4 MB of HTML
  in the body. Every header lies. A ``HEAD`` request is worse: it advertises
  ``content-length: 1``.
* Non-trading dates and pre-inception dates return ``200`` with a ~303-byte stub whose
  fields are the literal string ``"-"``.
* Numerics are quoted with thousands separators (``"1,627,770,127.10"``).

So: validate the *body*, never the headers, and refuse to persist anything that does not
parse into plausible holdings.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from functools import lru_cache
import fcntl
import io
import re
from dataclasses import dataclass

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .. import pit
from ..config import RAW

#: Undocumented private API. No SLA, subject to change without notice -- as the death of
#: the `.ajax` endpoint demonstrates. `canary_check()` exists to detect the next break.
HIST_API = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data"
    "/product-data/api/v1/get-fund-document"
)
LATEST_BASE = "https://www.ishares.com/us/products"

#: A browser-like UA is required; the default requests/curl UA gets an HTML interstitial.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0 Safari/537.36"
)


@dataclass(frozen=True)
class Fund:
    ticker: str
    portfolio_id: str
    slug: str
    inception: dt.date
    #: BlackRock's archive is month-end only before roughly this date, and daily after.
    #: Probing non-month-end dates earlier just burns requests on `"-"` stubs.
    daily_from: dt.date | None = None
    #: Earliest date served at all (month-end granularity).
    archive_from: dt.date | None = None

    @property
    def latest_url(self) -> str:
        return f"{LATEST_BASE}/{self.portfolio_id}/{self.slug}/latest-holdings.csv"


#: Coverage boundaries verified by probing the API on 2026-08-21.
#:
#: The broad-market funds matter more than MTUM for factor work: IWV carries ~2,900
#: Russell 3000 constituents *with prices*, point-in-time and survivorship-free (a name is
#: present while it is in the index, including names later acquired or delisted). That is
#: the free stock-level price panel the market-data survey concluded does not exist, and
#: it is what makes it possible to construct the momentum *short* leg -- the past-loser
#: decile -- rather than proxying it. The short leg is where Daniel-Moskowitz's mechanism
#: lives, so this is not a cosmetic improvement.
FUNDS: dict[str, Fund] = {
    "MTUM": Fund(
        "MTUM", "251614", "ishares-msci-usa-momentum-factor-etf",
        dt.date(2013, 4, 16), daily_from=dt.date(2013, 4, 16),
    ),
    "IWV": Fund(
        "IWV", "239714", "ishares-russell-3000-etf",
        dt.date(2000, 5, 22), daily_from=dt.date(2013, 1, 1),
        archive_from=dt.date(2006, 9, 1),
    ),
    "IWM": Fund(
        "IWM", "239710", "ishares-russell-2000-etf",
        dt.date(2000, 5, 22), daily_from=dt.date(2013, 1, 1),
        archive_from=dt.date(2006, 9, 1),
    ),
    "IWB": Fund(
        "IWB", "239707", "ishares-russell-1000-etf",
        dt.date(2000, 5, 15), daily_from=dt.date(2013, 1, 1),
        archive_from=dt.date(2006, 9, 1),
    ),
}

#: Upstream holes in BlackRock's archive, not fetch failures. Verified for IWV and
#: consistent with MTUM's otherwise-unexplained 187-day gap at 2017-07-05. Skipped rather
#: than retried, so a backfill does not spend hours re-requesting known-empty dates.
ARCHIVE_GAPS: tuple[tuple[dt.date, dt.date], ...] = (
    (dt.date(2014, 12, 20), dt.date(2015, 1, 25)),
    (dt.date(2017, 1, 1), dt.date(2017, 7, 31)),
)


def in_archive_gap(day: dt.date) -> bool:
    return any(a <= day <= b for a, b in ARCHIVE_GAPS)


#: Rows to drop when isolating the equity sleeve.
#:
#: These are EXCLUSION lists, never whitelists, because BlackRock's field *semantics* drift
#: across eras and every whitelist written against one era has silently emptied another:
#:
#:   Asset Class   "Equity" (2013) -> "Other" (2016) -> "Equity" (2019+)
#:   Sector        "-" for every row in 2006-era broad-market files
#:
#: Note "-" is absent from the sector list on purpose: in 2006 it means "not populated",
#: not "not an equity", and denylisting it dropped all 2,956 Russell 3000 constituents.
#: The real invariants are a positive price and a plausible row count, which is what
#: `_parse_positions` actually enforces.
_NON_EQUITY_CLASSES = {"Cash", "Money Market", "Futures", "Cash Collateral", "Derivatives", "-"}
_NON_EQUITY_SECTORS = {"Cash and/or Derivatives", "Cash", "Derivatives"}


class HarvestError(RuntimeError):
    """Response was not a genuine holdings file."""


class NotATradingDate(HarvestError):
    """The `"-"` stub: a weekend, holiday, or pre-inception date."""


@dataclass
class Holdings:
    """One point-in-time snapshot of a fund's book."""

    ticker: str
    as_of: dt.date
    fetched_at: pd.Timestamp
    shares_outstanding: float | None
    positions: pd.DataFrame
    raw_text: str

    @property
    def n_positions(self) -> int:
        return len(self.positions)


# --------------------------------------------------------------------------------------
# Fetch + validate
# --------------------------------------------------------------------------------------


# Deliberately does NOT retry NotATradingDate -- a Sunday will still be a Sunday.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=20),
    retry=retry_if_exception_type(requests.RequestException),
)
def _get(url: str, params: dict | None = None, session: requests.Session | None = None) -> str:
    s = session or requests.Session()
    resp = s.get(url, params=params, headers={"User-Agent": _UA}, timeout=60)
    resp.raise_for_status()
    return resp.text


def _validate_body(text: str, ticker: str, as_of: dt.date | None) -> None:
    """Reject HTML-masquerading-as-CSV and the `"-"` stub. Body only; headers lie."""
    head = text.lstrip()[:200].lower()
    if head.startswith(("<!doctype", "<html")) or "<head>" in head:
        raise HarvestError(
            f"{ticker}: HTML page returned while claiming CSV ({len(text):,} bytes). "
            f"This is the blocked-endpoint signature, not a transient error -- the URL "
            f"pattern has probably changed again. See module docstring."
        )
    if 'Fund Holdings as of,"-"' in text or len(text) < 1_000:
        raise NotATradingDate(
            f"{ticker}: no holdings for {as_of} (weekend, market holiday, or "
            f"pre-inception). Server returned a {len(text):,}-byte stub with HTTP 200."
        )
    if "Fund Holdings as of" not in text:
        raise HarvestError(
            f"{ticker}: unrecognised format, missing 'Fund Holdings as of' "
            f"({len(text):,} bytes). Refusing to persist."
        )


def _parse_header(text: str) -> tuple[dt.date, float | None]:
    m = re.search(r'Fund Holdings as of,\s*"?([A-Za-z]{3} \d{1,2}, \d{4})"?', text)
    if not m:
        raise HarvestError("could not locate 'Fund Holdings as of' date")
    as_of = dt.datetime.strptime(m.group(1), "%b %d, %Y").date()

    so: float | None = None
    m = re.search(r'Shares Outstanding,\s*"?([\d,]+(?:\.\d+)?)"?', text)
    if m:
        so = float(m.group(1).replace(",", ""))
    return as_of, so


def _parse_positions(text: str) -> pd.DataFrame:
    """Parse the holdings table following the preamble.

    Locates the header row by content rather than a fixed ``skiprows``, because the
    preamble length varies across funds and has changed over time.
    """
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Ticker,") and "Weight (%)" in ln),
        None,
    )
    if start is None:
        raise HarvestError("could not locate the holdings table header row")

    body: list[str] = []
    for ln in lines[start + 1 :]:
        if not ln.strip():  # disclaimer prose follows a blank line
            break
        body.append(ln)

    df = pd.read_csv(io.StringIO("\n".join([lines[start], *body])), thousands=",")
    df.columns = [c.strip() for c in df.columns]

    numeric = {
        "Weight (%)": "weight_pct",
        "Quantity": "quantity",
        "Market Value": "market_value",
        "Price": "price",
    }
    for src, dst in numeric.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(
                df[src].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )

    df = df.rename(
        columns={"Ticker": "ticker", "Name": "name", "Sector": "sector", "Asset Class": "asset_class"}
    )
    keep = [c for c in ["ticker", "name", "sector", "asset_class", *numeric.values()] if c in df]
    out = df[keep]

    # Identify equities by EXCLUSION, not by whitelisting `Asset Class`. BlackRock has
    # relabelled that field over time: it reads "Equity" in 2013, "Other" through part of
    # 2016, and "Equity" again from 2019. Whitelisting "Equity" silently drops every row
    # in the middle window -- which is how the first backfill died at 2016-05-05.
    if "asset_class" in out.columns:
        out = out.loc[~out["asset_class"].astype(str).str.strip().isin(_NON_EQUITY_CLASSES)]
    if "sector" in out.columns:
        out = out.loc[~out["sector"].astype(str).str.strip().isin(_NON_EQUITY_SECTORS)]
    if "price" in out.columns:
        out = out.loc[out["price"].fillna(0) > 0]

    if len(out) < 20:
        raise HarvestError(f"parsed only {len(out)} equity positions; expected O(100)")
    return out.reset_index(drop=True)


def fetch(
    ticker: str = "MTUM",
    as_of: dt.date | str | None = None,
    *,
    session: requests.Session | None = None,
) -> Holdings:
    """Fetch holdings for one fund, as of a date (default: latest published)."""
    fund = FUNDS[ticker]
    fetched_at = pd.Timestamp.now(tz="UTC")

    if as_of is None:
        text = _get(fund.latest_url, session=session)
    else:
        d = pd.Timestamp(as_of).date()
        text = _get(
            HIST_API,
            params={
                "appType": "PRODUCT_PAGE",
                "appSubType": "ISHARES",
                "targetSite": "us-ishares",
                "locale": "en_US",
                "portfolioId": fund.portfolio_id,
                "userType": "individual",
                "component": "holdings",
                "asOfDate": f"{d:%Y%m%d}",
            },
            session=session,
        )

    _validate_body(text, ticker, pd.Timestamp(as_of).date() if as_of else None)
    parsed_as_of, shares = _parse_header(text)
    return Holdings(ticker, parsed_as_of, fetched_at, shares, _parse_positions(text), text)


def canary_check(ticker: str = "MTUM") -> bool:
    """Cheap health check that the undocumented API still serves real data.

    Run this before any backfill and in the daily job. When BlackRock next changes the
    URL pattern, this is what tells us -- rather than a quietly corrupted dataset.
    """
    try:
        h = fetch(ticker, as_of=dt.date(2019, 9, 6))
    except HarvestError:
        return False
    return h.shares_outstanding == 88_000_000.0 and h.n_positions > 20


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def _raw_dir(ticker: str):
    d = RAW / "ishares" / ticker
    d.mkdir(parents=True, exist_ok=True)
    return d


def summary_path(ticker: str):
    return _raw_dir(ticker) / f"{ticker}_daily_summary.parquet"


def positions_path(ticker: str):
    return _raw_dir(ticker) / f"{ticker}_positions.parquet"


def persist(h: Holdings) -> pd.Series:
    """Write the immutable raw file and return the day's summary row.

    Raw bytes are retained because there is no archive of what BlackRock published when.
    If they restate, our snapshot is the only evidence the earlier version existed.
    """
    path = _raw_dir(h.ticker) / f"{h.ticker}_{h.as_of:%Y%m%d}.csv"
    if not path.exists():
        path.write_text(h.raw_text, encoding="utf-8")

    top10 = (
        h.positions.nlargest(10, "weight_pct")["weight_pct"].sum()
        if "weight_pct" in h.positions
        else float("nan")
    )
    return pd.Series(
        {
            "ticker": h.ticker,
            "as_of": pd.Timestamp(h.as_of),
            "fetched_at": h.fetched_at,
            "shares_outstanding": h.shares_outstanding,
            "n_positions": h.n_positions,
            "top10_weight": top10,
            "raw_path": str(path),
        }
    )


@contextlib.contextmanager
def _lock(path):
    """Exclusive advisory lock around a read-modify-write.

    Two backfill processes were once run concurrently against the same parquet files;
    their interleaved writes corrupted the positions table ("Couldn't deserialize
    thrift") and left a 187-day hole. The raw CSVs survived and `rebuild_from_raw()`
    recovered everything, but the append path should not have been racy in the first
    place.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _append(path, new: pd.DataFrame, keys: list[str]) -> None:
    with _lock(path):
        combined = (
            pd.concat([pd.read_parquet(path), new], ignore_index=True) if path.exists() else new
        )
        combined = (
            combined.drop_duplicates(subset=keys, keep="last")
            .sort_values(keys)
            .reset_index(drop=True)
        )
        # Write-then-rename so a crash mid-write cannot leave a half-written parquet.
        tmp = path.with_suffix(path.suffix + ".tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(path)


def panel_path(ticker: str, year: int):
    d = _raw_dir(ticker) / "panel"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker}_{year}.parquet"


def build_panel(ticker: str, year: int, *, force: bool = False) -> pd.DataFrame:
    """Parse one year of raw holdings into a tidy year-partitioned panel.

    Partitioned by year because a broad-market fund is ~2,900 names/day: appending to a
    single table per day is quadratic and would reach ~10M rows for IWV. Raw CSVs stay the
    source of truth; these are derived caches that can be deleted and rebuilt.
    """
    out = panel_path(ticker, year)
    if out.exists() and not force:
        return pd.read_parquet(out)

    frames = []
    for path in sorted(_raw_dir(ticker).glob(f"{ticker}_{year}*.csv")):
        text = path.read_text(encoding="utf-8")
        try:
            _validate_body(text, ticker, None)
            as_of, _ = _parse_header(text)
            pos = _parse_positions(text)
        except HarvestError:
            continue
        pos["as_of"] = pd.Timestamp(as_of)
        frames.append(pos[["as_of", "ticker", "sector", "weight_pct", "price", "quantity"]])

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(out, index=False)
    return panel


def _stored_years(ticker: str) -> list[int]:
    """Years with holdings on disk, from the raw CSVs or, failing those, the derived panel.

    The raw CSVs are the source of truth and are what a full working copy holds. A trimmed
    distribution ships only the derived year panels, which ``build_panel`` already returns
    unchanged when they exist, so discovering years from them makes the same code run against
    either copy. Raw CSVs win where both are present, so a working copy still rebuilds from
    source rather than from its own cache.
    """
    raw = {int(p.stem.split("_")[1][:4]) for p in _raw_dir(ticker).glob(f"{ticker}_*.csv")}
    if raw:
        return sorted(raw)
    d = _raw_dir(ticker) / "panel"
    return sorted({int(p.stem.split("_")[1][:4]) for p in d.glob(f"{ticker}_*.parquet")})


def price_panel(
    ticker: str = "IWV", start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Wide date x symbol price matrix assembled from stored holdings.

    Survivorship-free by construction: a name appears while it was actually in the index,
    including names later acquired or delisted. That is the property free price APIs
    destroy, and it is the one that matters for the momentum short leg, whose constituents
    disproportionately stop existing.
    """
    years = _stored_years(ticker)
    frames = [build_panel(ticker, y) for y in years]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise FileNotFoundError(f"no stored holdings for {ticker}")

    long = pd.concat(frames, ignore_index=True)
    if start:
        long = long[long["as_of"] >= pd.Timestamp(start)]
    if end:
        long = long[long["as_of"] <= pd.Timestamp(end)]
    return long.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")


@lru_cache(maxsize=4)
def _membership(ticker: str) -> tuple:
    """Every (as_of, frozenset of tickers) pair in the stored holdings, cached."""
    years = _stored_years(ticker)
    frames = [build_panel(ticker, y) for y in years]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise FileNotFoundError(f"no stored holdings for {ticker}")
    long = pd.concat(frames, ignore_index=True)
    return tuple((d, frozenset(g["ticker"])) for d, g in long.groupby("as_of"))


def members_at(as_of, ticker: str = "IWV") -> frozenset:
    """Point-in-time index membership: the holdings file at or before ``as_of``.

    This is the UNIVERSE DEFINITION, not a filter. The price panel is assembled from the
    holdings files, so a name carries prices only while it is a constituent; ranking every
    ticker that has ever appeared in the fund therefore sorts over names that could not have
    been held. Deletions look like halts and new entrants have no formation history, both of
    which are properties of the source rather than of the market.
    """
    as_of = pd.Timestamp(as_of)
    prior = [(d, m) for d, m in _membership(ticker) if d <= as_of]
    return prior[-1][1] if prior else frozenset()


def rebuild_from_raw(ticker: str = "MTUM") -> pd.DataFrame:
    """Regenerate both parquet tables from the immutable raw CSVs on disk.

    The raw files are the source of truth; the parquets are derived caches. This is the
    recovery path for a corrupted or partially-written cache, and it is also how a parser
    fix (such as the asset-class relabelling) gets applied retroactively without
    re-hitting the API 1,300 times.
    """
    rows, frames = [], []
    for path in sorted(_raw_dir(ticker).glob(f"{ticker}_*.csv")):
        text = path.read_text(encoding="utf-8")
        try:
            _validate_body(text, ticker, None)
            as_of, shares = _parse_header(text)
            positions = _parse_positions(text)
        except HarvestError:
            continue

        top10 = positions.nlargest(10, "weight_pct")["weight_pct"].sum()
        rows.append(
            {
                "ticker": ticker,
                "as_of": pd.Timestamp(as_of),
                "fetched_at": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC"),
                "shares_outstanding": shares,
                "n_positions": len(positions),
                "top10_weight": top10,
                "raw_path": str(path),
            }
        )
        pos = positions.copy()
        pos["as_of"] = pd.Timestamp(as_of)
        pos["fund"] = ticker
        frames.append(pos)

    summary = pd.DataFrame(rows).sort_values("as_of").reset_index(drop=True)
    summary.to_parquet(summary_path(ticker), index=False)
    positions = pd.concat(frames, ignore_index=True)
    positions.to_parquet(positions_path(ticker), index=False)
    return summary


def backfill(
    ticker: str = "MTUM",
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    *,
    store_positions: bool = True,
    pause: float = 0.4,
) -> pd.DataFrame:
    """Walk business days and accumulate holdings history.

    Skips non-trading dates via the `"-"` stub rather than a holiday calendar, so the
    fund's own publication schedule is the authority. Resumable: dates already present in
    the summary are not refetched.
    """
    import time

    fund = FUNDS[ticker]
    start = pd.Timestamp(start).date() if start else (fund.archive_from or fund.inception)
    end = pd.Timestamp(end).date() if end else dt.date.today()

    done: set[pd.Timestamp] = set()
    sp = summary_path(ticker)
    if sp.exists():
        done = set(pd.read_parquet(sp)["as_of"])

    # Before `daily_from` the archive only serves month-ends, so requesting every business
    # day there wastes thousands of round-trips on `"-"` stubs.
    daily_from = fund.daily_from or fund.inception
    bdays = pd.bdate_range(start, end)
    month_ends = set(pd.bdate_range(start, end, freq="BME"))
    candidates = [d for d in bdays if d.date() >= daily_from or d in month_ends]

    session = requests.Session()
    rows: list[pd.Series] = []
    for day in candidates:
        if day in done or in_archive_gap(day.date()):
            continue
        try:
            h = fetch(ticker, as_of=day.date(), session=session)
        except NotATradingDate:
            continue
        rows.append(persist(h))

        if store_positions:
            pos = h.positions.copy()
            pos["as_of"] = pd.Timestamp(h.as_of)
            pos["fund"] = ticker
            _append(positions_path(ticker), pos, ["fund", "as_of", "ticker"])

        _append(sp, pd.DataFrame([rows[-1]]), ["ticker", "as_of"])
        time.sleep(pause)

    return pd.DataFrame(rows)


def harvest(tickers: list[str] | None = None) -> pd.DataFrame:
    """Daily job: fetch the latest published file for each fund and append."""
    session = requests.Session()
    rows = []
    for t in tickers or list(FUNDS):
        h = fetch(t, session=session)
        rows.append(persist(h))
        _append(summary_path(t), pd.DataFrame([rows[-1]]), ["ticker", "as_of"])
    return pd.DataFrame(rows)


def load_flows(ticker: str = "MTUM") -> pd.DataFrame:
    """Shares-outstanding series with implied creation/redemption, PIT-stamped.

    ``observed_at`` is the fund's as-of date. ``available_at`` is that date plus one
    business day: BlackRock publishes holdings for date *t* on *t+1*. We deliberately do
    not use our own ``fetched_at`` for backfilled rows, since that would claim 2026
    availability for a 2013 observation.
    """
    sp = summary_path(ticker)
    if not sp.exists():
        raise FileNotFoundError(f"No harvest history for {ticker}. Run backfill() or harvest().")

    df = pd.read_parquet(sp).sort_values("as_of").copy()
    df["shares_chg"] = df["shares_outstanding"].diff()
    df["shares_chg_pct"] = df["shares_outstanding"].pct_change()
    # 20-day implied net flow as a share of base: the crowding-relevant aggregate.
    df["flow_20d_pct"] = df["shares_outstanding"].pct_change(20)

    avail = df["as_of"].apply(lambda d: pit.us_business_days(d, 1))
    return pit.stamp(df, observed_at="as_of", available_at=avail, name=f"ishares:{ticker}")
