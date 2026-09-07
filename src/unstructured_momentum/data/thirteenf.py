"""Institutional equity holdings from Form 13F, structured and point-in-time.

Why this exists as a module rather than a script
------------------------------------------------
Crowding is the mechanism behind the August 2007 momentum unwind, and for most of this
project it was treated as unobservable. That was wrong. It was unobservable in the corpus
being searched -- company filings, which describe what a firm is exposed to and never who
owns it -- and the emptiness got generalised to all data. Form 13F is the record of who owns
what, it is free, and it reaches back to 2001, which covers thirteen of the fifteen
registered episodes including the quant quake.

The format punishes assumptions, so the defences live here rather than in the caller:

* **The index quarter is the FILING quarter, not the holdings quarter.** A 13F accepted in
  May 2014 reports holdings as of 31 March 2014. Reading the index quarter as the position
  date shifts the entire panel one quarter early, which is a leak: it would place March
  positions in the hands of a monitor that could not have seen them until May. Every row here
  carries ``period_of_report`` taken from the filing header and ``available_at`` taken from
  the acceptance timestamp, and the two are never conflated.
* **Values are in thousands of dollars until they suddenly are not.** The SEC amendment moved
  Form 13F to whole dollars for filings accepted from early 2023, but filers transitioned
  raggedly: in the first affected quarter 94 percent had switched and the rest had not. A
  date-based rule is therefore wrong for six percent of a quarter, silently and by a factor
  of a thousand. Units are detected per filing from the distribution of value-per-share.
* **The parse validates itself against a price.** Median value-per-share across the holdings
  of a large filer is an estimate of the market price of a stock it holds, and that estimate
  can be checked against the actual close. When this was first run it disagreed by 5.8x,
  which looked like a units bug and was really the filing-quarter error above. A parse that
  cannot reproduce a known price is not trusted, and the check is cheap enough to keep.
* **Two document formats.** Filings from 2013 onward carry an XML information table, with or
  without a namespace prefix depending on the filer's software. Earlier ones are whitespace
  aligned text whose column positions vary between filers. Both are handled, and the text
  parser reports the share of lines it could not read rather than returning what it managed.

Layout
------
``data/raw/thirteenf/``
    ``index/{year}q{n}.parquet``    one row per 13F-HR accepted that quarter
    ``holdings/{period}.parquet``   all holdings reported for that position date
    ``raw/{accession}.txt.gz``      the dissemination document as fetched
    ``manifest.json``               what was requested, what arrived, failures by reason

Raw documents are kept because re-parsing is free and re-fetching is not. The 13F population
is large -- roughly three thousand filers a quarter in 2007 and nine thousand now -- and a
full sweep is measured in hours, so the corpus is built once and queried many times.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

USER_AGENT = "unstructured-momentum research chee.timothy@gmail.com"
ROOT = Path("data/raw/thirteenf")

#: SEC asks for ten requests a second; eight leaves headroom for a shared bucket.
RATE_LIMIT = 8.0
_BUCKET_LOCK = threading.Lock()
_next_slot = 0.0

#: Electronic 13F-HR begins here. Earlier episodes get no positioning layer at all.
FIRST_YEAR = 2001

#: A filing whose median value-per-share is below this is reporting thousands of dollars.
#: Common stock priced under one dollar a share is rare enough in a whole-portfolio median
#: that the gap between the two regimes is three orders of magnitude wide.
UNITS_CUTOFF = 1.0

class ParseFailure(Exception):
    """The document was fetched but no holdings could be read from it."""


def _acquire() -> None:
    """One global token bucket, shared by every worker."""
    global _next_slot
    with _BUCKET_LOCK:
        now = time.monotonic()
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + 1.0 / RATE_LIMIT
    if wait:
        time.sleep(wait)


def _get(url: str, session: requests.Session | None = None) -> str:
    _acquire()
    s = session or requests
    r = s.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------- index

_IDX_ROW = re.compile(
    r"^13F-HR(?:/A)?\s+(?P<name>.+?)\s{2,}(?P<cik>\d+)\s+(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<path>\S+)\s*$"
)


def index(year: int, quarter: int, *, refresh: bool = False) -> pd.DataFrame:
    """Every 13F-HR accepted in a calendar quarter.

    The date column is the acceptance date, which is when a monitor could first have seen
    the filing. It is not the position date; see ``period_of_report`` on the holdings.
    """
    out = ROOT / "index" / f"{year}q{quarter}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    txt = _get(
        f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"
    )
    rows = []
    for line in txt.splitlines():
        if not line.startswith("13F-HR"):
            continue
        m = _IDX_ROW.match(line)
        if m:
            d = m.groupdict()
            d["amended"] = line.startswith("13F-HR/A")
            rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ParseFailure(f"no 13F-HR rows parsed from {year} QTR{quarter}")
    df["cik"] = df["cik"].astype(int)
    df["accepted_at"] = pd.to_datetime(df["date"])
    df["accession"] = df["path"].str.rsplit("/", n=1).str[-1].str.replace(".txt", "", regex=False)
    df = df.drop(columns=["date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


# ---------------------------------------------------------------- document parse

_PERIOD = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")
_ACCEPTED = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")
_INFO_TABLE = re.compile(r"<(?:\w+:)?infoTable>(.*?)</(?:\w+:)?infoTable>", re.S)

#: Anchor on the CUSIP, which is the one field with a fixed shape, then read the two
#: numbers that follow it and the share/principal flag. Filer column positions vary; the
#: order of these fields does not.
_TEXT_ROW = re.compile(
    r"(?P<cusip>[0-9A-Z]{8}[0-9A-Z])\s+"
    r"\$?(?P<value>[\d,]+(?:\.\d+)?)\s+"
    r"(?P<shares>[\d,]+(?:\.\d+)?)\s+"
    r"(?P<kind>SH|PRN)\b"
)

#: A second pre-2013 layout: CUSIP first, then the issuer name, then two numbers, and no
#: share/principal token at all. Roughly a quarter of 2006 filings use it, and under the
#: layout above they parse to nothing. The two numbers appear in the OPPOSITE order to the
#: first layout, so a regex permissive enough to match both would read shares as value and
#: value as shares on every one of them -- silently, since both are positive integers of
#: plausible magnitude. Which order is right is settled per filing by implied price, never
#: by the pattern.
_TEXT_ROW_CUSIP_FIRST = re.compile(
    r"(?P<cusip>[0-9A-Z]{8}[0-9A-Z])\s+"
    r"(?P<issuer>[A-Z][A-Za-z0-9 &.,'/\-]{2,60}?)\s+"
    r"\$?(?P<a>[\d,]+(?:\.\d+)?)\s+"
    r"\$?(?P<b>[\d,]+(?:\.\d+)?)\s+"
    r"(?=SOLE|SHARED|DEFINED|SH\b|PRN\b|N/A|OTHER|\d)"
)

#: A fourth pre-2013 layout: the issuer name, then the CUSIP, then value and share count,
#: with NO share-or-principal token anywhere -- these filings mark discretion with bare "X"
#: and "NA" columns instead. Requiring SH or PRN rejected them entirely, and they are not
#: small: their documents run to the same length as the ones that parsed. As with the
#: CUSIP-first layout, which of the two numbers is the value cannot be read off the pattern
#: and is settled against reference prices.
_TEXT_ROW_NOTOKEN = re.compile(
    r"(?P<cusip>[0-9A-Z]{8}[0-9A-Z])\s+"
    r"\$?(?P<a>[\d,]+(?:\.\d+)?)\s+"
    r"\$?(?P<b>[\d,]+(?:\.\d+)?)(?:\s|$)"
)


#: A third pre-2013 layout: fixed-width columns with no separator at all, so the CUSIP runs
#: straight into the market value as "0003611054049". Where the field boundary falls is
#: undecidable from the text, but not from the identifier: only one split point yields a
#: valid CUSIP check digit, so the checksum locates the boundary that whitespace does not
#: mark. This is the same principle as trp-32 -- the arbiter comes from outside the bytes --
#: applied to a column boundary rather than a column order.
_TEXT_ROW_GLUED = re.compile(
    r"(?P<blob>[0-9A-Z]{10,32})\s+(?P<shares>[\d,]+(?:\.\d+)?)\s+(?P<kind>SH|PRN)\b"
)


#: Median implied price a correctly-ordered, correctly-scaled holdings table should produce.
#: The upper bound has to clear Berkshire Hathaway A, which traded near 118,510 dollars in
#: September 2007 and was flagged implausible by an earlier bound of 100,000. A guard that
#: rejects the most expensive real security in the market is measuring its own calibration.
_SANE_PRICE = (0.05, 1_000_000.0)


def valid_cusip(code: str) -> bool:
    """Whether a nine-character token is a real CUSIP, by its own check digit.

    The pattern ``[0-9A-Z]{9}`` matches plenty of things that are not identifiers. In one
    2007 filing it matched a fragment of an issuer name, "IFIEDREIT" out of DIVERSIFIED
    REIT, and the row attached to it carried 2.5e13 dollars. Two such filings accounted for
    80 percent of a quarter's total value while the quarter's anchor price check passed.

    CUSIP carries a modulus-ten check digit computed over its first eight characters, so the
    identifier can be asked whether it is one. This rejects name fragments outright, since
    their final character is usually not even a digit.
    """
    if not isinstance(code, str) or len(code) != 9:
        return False
    total = 0
    for i, ch in enumerate(code[:8]):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch.upper()) - ord("A") + 10
        elif ch == "*":
            v = 36
        elif ch == "@":
            v = 37
        elif ch == "#":
            v = 38
        else:
            return False
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    if not code[8].isdigit():
        return False
    return (10 - (total % 10)) % 10 == int(code[8])


#: Security-class descriptors sitting between the issuer name and the CUSIP. Stripped so
#: that two filers describing the same issuer as "COM" and "COM SH BEN INT" agree.
_CLASS_TAIL = re.compile(
    r"\s+(COM|CL\s*[A-Z]|COMMON|ORD|ADR|SPON\s*ADR|SH\s*BEN\s*INT|PFD|NOTE|"
    r"UNIT|WT|WTS|RIGHTS?|PUT|CALL|DEP\s*RCPT|PAR\s*[\d.]+|USD[\d.]*|"
    r"\$[\d.]+|SBI|TR\s*UNIT|BEN\s*INT)\b.*$",
    re.I,
)


def _clean_issuer(prefix: str) -> str | None:
    """The issuer name from whatever precedes the CUSIP on a holdings line."""
    s = re.sub(r"\s+", " ", prefix).strip(" .,-")
    s = _CLASS_TAIL.sub("", s).strip(" .,-")
    return s.upper() or None


def _xml_field(block: str, name: str) -> str | None:
    m = re.search(rf"<(?:\w+:)?{name}>([^<]*)</(?:\w+:)?{name}>", block)
    return m.group(1).strip() if m else None


def _num(x: str | None) -> float:
    if x is None:
        return float("nan")
    return float(x.replace(",", "").replace("$", "").strip() or "nan")


def parse_holdings(doc: str, reference: dict | None = None) -> pd.DataFrame:
    """Holdings from either document format, with the unread share reported.

    Returns columns ``cusip``, ``issuer``, ``value_raw``, ``shares``, ``kind``. The value is
    left in whatever units the filer used; call :func:`scale_values` to normalise.
    """
    blocks = _INFO_TABLE.findall(doc)
    if blocks:
        rows = []
        for b in blocks:
            rows.append(
                {
                    "cusip": (_xml_field(b, "cusip") or "").upper().strip(),
                    "issuer": _xml_field(b, "nameOfIssuer"),
                    "value_raw": _num(_xml_field(b, "value")),
                    "shares": _num(_xml_field(b, "sshPrnamt")),
                    "kind": (_xml_field(b, "sshPrnamtType") or "SH").upper(),
                }
            )
        df = pd.DataFrame(rows)
        n_before = len(df)
        ok = df["cusip"].map(valid_cusip)
        df = df[ok]
        px = df["value_raw"] / df["shares"].replace(0, np.nan)
        lo, hi = _SANE_PRICE
        sane = ((px >= lo) & (px <= hi)) | ((px * 1000 >= lo) & (px * 1000 <= hi))
        keep = (df["kind"] != "SH") | sane.fillna(False)
        n_impossible = int((~keep).sum())
        df = df[keep]
        df.attrs["format"] = "xml"
        df.attrs["unread_share"] = 0.0
        df.attrs["impossible_price_rows"] = n_impossible
        df.attrs["bad_cusip_rows"] = int(n_before - len(df) - n_impossible)
        if df.empty:
            raise ParseFailure("no rows survived CUSIP check-digit validation")
        return df

    # Pre-2013 text table. Matched line by line rather than across the whole document, so
    # that whatever precedes the CUSIP on the same line can be kept as the issuer name.
    # The name is the only bridge from a CUSIP to a ticker without a paid identifier
    # crosswalk, so discarding it would make the panel unjoinable to any index membership.
    candidates = sum(
        1 for ln in doc.splitlines() if re.search(r"\b[0-9A-Z]{8}[0-9A-Z]\b\s+[\d,$]", ln)
    )
    variants = [
        ("name-first", _parse_name_first(doc)),
        ("glued-columns", _parse_glued(doc)),
        ("cusip-first-value-shares", _parse_cusip_first(doc, value_first=True)),
        ("cusip-first-shares-value", _parse_cusip_first(doc, value_first=False)),
        ("no-token-value-shares", _parse_notoken(doc, value_first=True)),
        ("no-token-shares-value", _parse_notoken(doc, value_first=False)),
    ]
    # Choose by implied price, not by which pattern matched most rows. Two of these layouts
    # differ only in which of two positive integers is the value, so row count cannot
    # separate them and only the price they imply can.
    stated = stated_totals(doc)
    scored = []
    for name, df in variants:
        if df is None or not len(df):
            continue
        agree = _totals_agreement(df, stated)
        # The filing's own declared totals outrank every heuristic when present, because
        # they are the only signal that can see over-matching: a spurious row carrying a
        # plausible price is invisible to a price check and obvious in an entry count.
        scored.append((name, df, agree if agree is not None else _price_sanity(df, reference)))
    if not scored:
        raise ParseFailure("no holdings rows matched in any known text layout")
    layout, df, score = max(scored, key=lambda t: (t[2], len(t[1])))
    if score < 0.5:
        raise ParseFailure(
            f"best layout {layout!r} implies sane prices for only {score:.0%} of rows"
        )
    n_before = len(df)
    df = df[df["cusip"].map(valid_cusip)]
    if df.empty:
        raise ParseFailure(f"no rows survived CUSIP validation under layout {layout!r}")

    # Drop share-denominated rows whose value and share count imply an impossible price
    # under EITHER unit scaling. Such a row is a parse error whatever produced it -- glued
    # digits, a bond principal read as a share count, a column boundary in the wrong place --
    # and reporting it was not enough. One filing survived every other check with a Duke
    # Energy bond at six billion dollars a share and carried 99.99 percent of a quarter's
    # total value. Principal rows are exempt, because a principal amount is not a price.
    # NOTE: only a coarse pre-filter here, admitting either unit scaling, because which one
    # applies is not known until the whole table has been seen. The binding check happens in
    # scale_values once units are fixed.
    px = df["value_raw"] / df["shares"].replace(0, np.nan)
    lo, hi = _SANE_PRICE
    sane = ((px >= lo) & (px <= hi)) | ((px * 1000 >= lo) & (px * 1000 <= hi))
    keep = (df["kind"] != "SH") | sane.fillna(False)
    n_impossible = int((~keep).sum())
    df = df[keep]
    if df.empty:
        raise ParseFailure(f"every row implied an impossible price under layout {layout!r}")
    df = df[["cusip", "issuer", "value_raw", "shares", "kind"]]
    df.attrs["format"] = "text"
    df.attrs["bad_cusip_rows"] = int(n_before - len(df) - n_impossible)
    df.attrs["impossible_price_rows"] = n_impossible
    df.attrs["layout"] = layout
    df.attrs["layout_score"] = score
    df.attrs["stated_entries"], df.attrs["stated_value"] = stated
    df.attrs["unread_share"] = (
        0.0 if candidates == 0 else max(0.0, 1.0 - len(df) / candidates)
    )
    return df


#: Form 13F asks the filer to declare how many entries the table holds and what they are
#: worth in total. That is a checksum the document carries about itself, and it beats every
#: heuristic in this module: it settles layout, column order and units in one comparison,
#: and it catches over-matching, which no price check can see because a spurious row with a
#: plausible price looks exactly like a real one.
_ENTRY_TOTAL = re.compile(r"Entry\s+Total:?\s*\$?\s*([\d,]+)", re.I)
_VALUE_TOTAL = re.compile(r"Value\s+Total:?\s*\$?\s*([\d,]+)", re.I)


def stated_totals(doc: str) -> tuple[int | None, float | None]:
    """The entry count and value total the filing declares for itself."""
    e = _ENTRY_TOTAL.search(doc)
    v = _VALUE_TOTAL.search(doc)
    return (
        int(e.group(1).replace(",", "")) if e else None,
        float(v.group(1).replace(",", "")) if v else None,
    )


def _totals_agreement(df: pd.DataFrame, stated: tuple[int | None, float | None]) -> float | None:
    """How well a candidate reading reproduces the filing's declared totals, or None."""
    n_stated, v_stated = stated
    if not n_stated and not v_stated:
        return None
    scores = []
    if n_stated:
        scores.append(max(0.0, 1.0 - abs(len(df) - n_stated) / max(n_stated, 1)))
    if v_stated:
        got = float(df["value_raw"].sum())
        # The declared total is in the filer's own units, so compare against both scalings.
        best = max(
            1.0 - min(abs(got / v_stated - 1), 1.0),
            1.0 - min(abs(got / 1000 / v_stated - 1), 1.0),
            1.0 - min(abs(got * 1000 / v_stated - 1), 1.0),
        )
        scores.append(max(0.0, best))
    return sum(scores) / len(scores)


def _price_sanity(df: pd.DataFrame, reference: dict | None = None) -> float:
    """How well a candidate reading of a text table agrees with prices known from elsewhere.

    A plausibility range cannot arbitrate between the two column orders and it was wrong to
    think it could. Transposing value and shares inverts the implied price, and an inverse
    price is usually still a plausible price: one filing read the wrong way put Iomega at
    401 dollars a share when it closed near 2.50, and both readings sat comfortably inside
    any range wide enough to admit real equities. Worse, the wrong reading scored 100
    percent, because every row was equally and identically wrong.

    So the arbiter is external. ``reference`` maps CUSIP to a price established from filings
    whose layout is unambiguous -- the XML era, and the pre-2013 tables that carry an
    explicit SH or PRN token. A candidate reading is scored by the share of its overlapping
    CUSIPs whose implied price lands within ten percent of that consensus, trying both unit
    scalings. With no overlap the score falls back to bare plausibility, which is weak, and
    the caller is told so by the returned coverage.
    """
    eq = df[(df["kind"] == "SH") & (df["shares"] > 0) & (df["value_raw"] > 0)]
    eq = eq[eq["cusip"].map(valid_cusip)]
    if eq.empty:
        return 0.0
    r = eq["value_raw"] / eq["shares"]
    if reference:
        ref = eq["cusip"].map(reference)
        ok = ref.notna()
        if ok.sum() >= 3:
            best = 0.0
            for scale in (1.0, 1000.0):
                agree = ((r[ok] * scale / ref[ok] - 1).abs() <= 0.10).mean()
                best = max(best, float(agree))
            return best
    lo, hi = _SANE_PRICE
    return float(max(((r >= lo) & (r <= hi)).mean(), ((r * 1000 >= lo) & (r * 1000 <= hi)).mean()))


def _parse_name_first(doc: str) -> pd.DataFrame | None:
    rows = []
    for ln in doc.splitlines():
        m = _TEXT_ROW.search(ln)
        if not m:
            continue
        d = m.groupdict()
        d["issuer"] = _clean_issuer(ln[: m.start()])
        rows.append(d)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["value_raw"] = df["value"].map(_num)
    df["shares"] = df["shares"].map(_num)
    return df


def _split_glued(blob: str, shares: float) -> tuple[str, float] | None:
    """Find where the CUSIP ends inside a run with no separators.

    The checksum alone is not enough. A truncated class name can run into the identifier --
    "AMERICAN DEPOS86468610022000" -- and a nine-character window starting inside that name
    passes the check digit roughly one time in ten, which put a Suez ADR position at 21
    million dollars a share. So every offset that yields a valid CUSIP with an all-digit
    remainder is a candidate, and the one whose implied price is plausible wins.
    """
    best = None
    for i in range(len(blob) - 8):
        head, tail = blob[i : i + 9], blob[i + 9 :]
        if not tail.isdigit() or not tail or not valid_cusip(head):
            continue
        value = float(tail)
        if shares <= 0 or value <= 0:
            continue
        px = value / shares
        lo, hi = _SANE_PRICE
        score = max(int(lo <= px <= hi), int(lo <= px * 1000 <= hi))
        if score and (best is None or value < best[1]):
            # Prefer the smaller value: a split too far left absorbs digits from the class
            # name and inflates the amount, never deflates it.
            best = (head, value)
    return best


def _parse_glued(doc: str) -> pd.DataFrame | None:
    """Fixed-width rows where the CUSIP abuts the value, boundary found by check digit."""
    rows = []
    for ln in doc.splitlines():
        m = _TEXT_ROW_GLUED.search(ln)
        if not m:
            continue
        shares = _num(m.group("shares"))
        got = _split_glued(m.group("blob"), shares)
        if not got:
            continue
        cusip, value = got
        rows.append(
            {
                "cusip": cusip,
                "issuer": _clean_issuer(ln[: m.start()]),
                "value_raw": value,
                "shares": shares,
                "kind": m.group("kind"),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def _parse_notoken(doc: str, *, value_first: bool) -> pd.DataFrame | None:
    """Name, CUSIP, then two numbers, with no share-or-principal token to anchor on."""
    rows = []
    for ln in doc.splitlines():
        m = _TEXT_ROW_NOTOKEN.search(ln)
        if not m or not valid_cusip(m.group("cusip")):
            continue
        a, b = _num(m.group("a")), _num(m.group("b"))
        rows.append(
            {
                "cusip": m.group("cusip"),
                "issuer": _clean_issuer(ln[: m.start()]),
                "value_raw": a if value_first else b,
                "shares": b if value_first else a,
                "kind": "SH",
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def _parse_cusip_first(doc: str, *, value_first: bool) -> pd.DataFrame | None:
    rows = []
    for ln in doc.splitlines():
        m = _TEXT_ROW_CUSIP_FIRST.search(ln)
        if not m:
            continue
        d = m.groupdict()
        a, b = _num(d.pop("a")), _num(d.pop("b"))
        d["value_raw"], d["shares"] = (a, b) if value_first else (b, a)
        d["issuer"] = _clean_issuer(d["issuer"])
        d["kind"] = "SH"
        rows.append(d)
    if not rows:
        return None
    return pd.DataFrame(rows)


def detect_units(holdings: pd.DataFrame) -> tuple[str, float]:
    """Whether this filer reported thousands or whole dollars, and the median it rests on.

    Uses only share-denominated lines, because a principal amount on a bond is not a share
    count and its ratio carries no price information.

    Deliberately not gated on a minimum holding count. An earlier version required eight
    equity lines, which sent a six-line filer to the quarter's modal units and multiplied
    its positions by a thousand into twenty-four billion dollar holdings. The two regimes
    sit three orders of magnitude apart -- a value-per-share ratio is a price, and prices
    are tens of dollars while thousands-scaled ratios are hundredths -- so a handful of
    consistent ratios settles the question and a count threshold only discards the evidence.
    ``unknown`` is now returned only when there is no usable line at all.
    """
    eq = holdings[(holdings["kind"] == "SH") & (holdings["shares"] > 0) & (holdings["value_raw"] > 0)]
    if eq.empty:
        return "unknown", float("nan")
    med = float(np.median(eq["value_raw"] / eq["shares"]))
    return ("thousands" if med < UNITS_CUTOFF else "dollars"), med


#: One bound, used everywhere. Two once existed and drifted: the Berkshire ceiling was
#: raised in the parser's copy and not in the reporting copy, so a quarter read 35 percent
#: implausible against one constant and 0.009 percent against the other, for the same data.
PLAUSIBLE_PRICE = _SANE_PRICE


#: How far one filer's implied price may sit from what every other filer reports for the
#: same security on the same date, before the row is treated as a misparse.
CONSENSUS_TOLERANCE = 5.0


def drop_against_consensus(panel: pd.DataFrame, *, tolerance: float = CONSENSUS_TOLERANCE):
    """Remove rows whose implied price contradicts the other filers holding that security.

    No fixed price bound can do this job. The ceiling has to clear Berkshire Hathaway A,
    which really did trade near 118,510 dollars in 2007, and a ceiling that admits Berkshire
    also admits a misparsed ConocoPhillips at 370,770 when fifty other filers put it at 88.
    Absolute plausibility cannot separate those two cases; agreement between filers can.

    This is the same instrument as the layout arbitration in trp-32 -- a referent from
    outside the document -- applied across filings rather than within one. The redundancy is
    free: a widely held security is reported by hundreds of managers on the same date, and
    they cannot all be wrong in the same direction.
    """
    if panel.empty:
        return panel, {"consensus_rows_dropped": 0, "consensus_value_dropped": 0.0}
    eq = (panel["kind"] == "SH") & (panel["shares"] > 0) & (panel["value_usd"] > 0)
    px = panel["value_usd"] / panel["shares"].where(eq)
    med = px.groupby(panel["cusip"]).transform("median")
    n_rep = px.groupby(panel["cusip"]).transform("count")
    # Only judge a row when several filers report the security; a single-filer name has no
    # consensus to contradict and is left to the absolute bound.
    ratio = (px / med).where(med > 0)
    bad = eq & (n_rep >= 3) & ((ratio > tolerance) | (ratio < 1 / tolerance))
    bad = bad.fillna(False)
    dropped_value = float(panel.loc[bad, "value_usd"].sum())
    total_value = float(panel.loc[eq, "value_usd"].sum())
    out = panel[~bad].reset_index(drop=True)
    return out, {
        "consensus_rows_dropped": int(bad.sum()),
        "consensus_value_dropped": dropped_value / total_value if total_value else 0.0,
    }


def implausible_share(holdings: pd.DataFrame) -> float:
    """Share of scaled equity value sitting at an impossible implied price."""
    eq = holdings[(holdings["kind"] == "SH") & (holdings["shares"] > 0) & (holdings["value_usd"] > 0)]
    if eq.empty:
        return 0.0
    px = eq["value_usd"] / eq["shares"]
    bad = (px < PLAUSIBLE_PRICE[0]) | (px > PLAUSIBLE_PRICE[1])
    return float(eq.loc[bad, "value_usd"].sum() / eq["value_usd"].sum())


def scale_values(holdings: pd.DataFrame, *, fallback: str | None = None) -> pd.DataFrame:
    """Add ``value_usd``, normalised to whole dollars, using per-filing unit detection.

    A filing holding one or two names is too thin to type from its own median. Those return
    ``unknown``, and without a fallback they would be scaled as whole dollars -- which
    understates a thousands-era filing by a factor of a thousand and looks like a very small
    manager rather than like a parse failure. ``fallback`` should be the modal units of the
    filings in the same quarter that could be typed; :func:`quarter` supplies it.
    """
    units, med = detect_units(holdings)
    effective = units if units != "unknown" else (fallback or "dollars")
    out = holdings.copy()
    out["value_usd"] = out["value_raw"] * (1000.0 if effective == "thousands" else 1.0)

    # The binding plausibility check, applied once units are known. Running it on the raw
    # ratio is not enough and quietly failed: a Conoco row whose raw value over shares came
    # to 370 passed as a dollar price, then the filing was typed thousands and the same row
    # became 370,770 dollars a share. The pre-filter admits either scaling by necessity,
    # since units are a property of the whole table; only here is the question decidable.
    px = out["value_usd"] / out["shares"].replace(0, np.nan)
    lo, hi = PLAUSIBLE_PRICE
    keep = (out["kind"] != "SH") | ((px >= lo) & (px <= hi)).fillna(False)
    n_dropped = int((~keep).sum())
    out = out[keep]
    out.attrs = dict(holdings.attrs)
    out.attrs["scaled_price_drops"] = n_dropped
    out.attrs["units"] = units
    out.attrs["units_effective"] = effective
    out.attrs["units_median"] = med
    return out


@dataclass
class Filing:
    accession: str
    cik: int
    name: str
    period_of_report: dt.date
    available_at: pd.Timestamp
    holdings: pd.DataFrame
    doc_format: str
    units: str
    unread_share: float
    stated_entries: int | None = None


def fetch_filing(path: str, *, session=None, use_cache: bool = True) -> str:
    """The dissemination document, cached on first fetch."""
    accession = path.rsplit("/", 1)[-1].replace(".txt", "")
    cache = ROOT / "raw" / f"{accession}.txt.gz"
    if use_cache and cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    doc = _get(f"https://www.sec.gov/Archives/{path}", session=session)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as fh:
        fh.write(doc)
    return doc


def read_filing(
    row,
    *,
    session=None,
    use_cache: bool = True,
    fallback_units: str | None = None,
    reference: dict | None = None,
) -> Filing:
    """One filing, parsed, with its position date and availability separated."""
    doc = fetch_filing(row["path"], session=session, use_cache=use_cache)
    m = _PERIOD.search(doc)
    if not m:
        raise ParseFailure(f"{row['accession']}: no CONFORMED PERIOD OF REPORT")
    period = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
    a = _ACCEPTED.search(doc)
    available = (
        pd.to_datetime(a.group(1), format="%Y%m%d%H%M%S")
        if a
        else pd.Timestamp(row["accepted_at"])
    )
    h = scale_values(parse_holdings(doc, reference), fallback=fallback_units)
    return Filing(
        accession=row["accession"],
        cik=int(row["cik"]),
        name=row["name"],
        period_of_report=period,
        available_at=available,
        holdings=h,
        doc_format=h.attrs.get("format", "?"),
        units=h.attrs.get("units", "unknown"),
        unread_share=float(h.attrs.get("unread_share", 0.0)),
        stated_entries=h.attrs.get("stated_entries"),
    )


def quarter(
    year: int,
    quarter_no: int,
    *,
    limit: int | None = None,
    session=None,
    validate_cusip: str | None = None,
    validate_price: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Every readable filing accepted in a quarter, as one holdings panel plus a report.

    Units are typed per filing, and filings too thin to type inherit the modal units of
    those that could be -- so the fallback is calibrated from the quarter itself rather than
    from a hardcoded amendment date that six percent of filers ignored.

    Pass ``validate_cusip`` and ``validate_price`` to gate the panel on reproducing a known
    close. A quarter that cannot recover a price it should know is not returned as data.
    """
    ix = index(year, quarter_no)
    if limit:
        ix = ix.head(limit)

    # Two passes. The first reads every filing with no external referent, which succeeds for
    # the XML era and for text tables carrying an explicit SH or PRN token -- layouts whose
    # column order is unambiguous. Those establish a consensus price per CUSIP. The second
    # pass re-reads whatever failed or landed on an ambiguous layout, using that consensus to
    # decide which of two positive integers is the value and which the share count. Re-reading
    # is free because the raw documents are cached; the network cost is paid once.
    parsed, failures, retry = [], {}, []
    for _, r in ix.iterrows():
        try:
            f = read_filing(r, session=session)
            if f.doc_format == "text" and f.holdings.attrs.get("layout", "").startswith("cusip-first"):
                retry.append(r)
            parsed.append(f)
        except Exception as exc:  # noqa: BLE001 - reason is recorded, not swallowed
            failures[type(exc).__name__] = failures.get(type(exc).__name__, 0) + 1
            retry.append(r)

    unambiguous = [
        f for f in parsed
        if f.doc_format == "xml" or not f.holdings.attrs.get("layout", "").startswith("cusip-first")
    ]
    reference = {}
    if unambiguous:
        ref_rows = pd.concat([f.holdings.assign(_c=f.holdings["cusip"]) for f in unambiguous])
        ref_rows = ref_rows[(ref_rows["kind"] == "SH") & (ref_rows["shares"] > 0) & (ref_rows["value_usd"] > 0)]
        reference = (ref_rows["value_usd"] / ref_rows["shares"]).groupby(ref_rows["_c"]).median().to_dict()

    if reference and retry:
        by_accession = {f.accession: i for i, f in enumerate(parsed)}
        recovered = 0
        for r in retry:
            try:
                f = read_filing(r, session=session, reference=reference)
            except Exception:  # noqa: BLE001 - already counted in failures
                continue
            if f.accession in by_accession:
                parsed[by_accession[f.accession]] = f
            else:
                parsed.append(f)
                recovered += 1
        failures["recovered_on_second_pass"] = recovered
        for k in list(failures):
            if k != "recovered_on_second_pass":
                failures[k] = max(0, failures[k] - recovered)

    typed = [f.units for f in parsed if f.units != "unknown"]
    modal = pd.Series(typed).mode().iloc[0] if typed else "dollars"

    frames = []
    for f in parsed:
        h = f.holdings
        if f.units == "unknown":
            h = scale_values(h.drop(columns=["value_usd"]), fallback=modal)
        h = h.assign(
            cik=f.cik,
            filer=f.name,
            accession=f.accession,
            period_of_report=f.period_of_report,
            available_at=f.available_at,
            units=f.units,
        )
        frames.append(h)

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    panel, consensus_report = drop_against_consensus(panel)
    report = {
        "index_quarter": f"{year}Q{quarter_no}",
        "period_of_report": str(panel["period_of_report"].mode().iloc[0]) if len(panel) else None,
        "filings_indexed": int(len(ix)),
        "filings_read": len(parsed),
        "failures": failures,
        "modal_units": modal,
        "units_breakdown": pd.Series([f.units for f in parsed]).value_counts().to_dict(),
        "untyped_filings": sum(1 for f in parsed if f.units == "unknown"),
        "untyped_value_share": (
            float(panel.loc[panel["units"] == "unknown", "value_usd"].sum() / panel["value_usd"].sum())
            if len(panel) and panel["value_usd"].sum() > 0
            else 0.0
        ),
        "mean_unread_share": float(np.mean([f.unread_share for f in parsed])) if parsed else 0.0,
        "implausible_value_share": implausible_share(panel) if len(panel) else 0.0,
        **consensus_report,
        # Agreement with the entry count each filing declares for itself. The only check in
        # this module that can see over-matching, since a spurious row with a plausible
        # price is invisible to every price-based test.
        "filings_declaring_entries": sum(1 for f in parsed if f.stated_entries),
        "entry_count_exact": (
            sum(1 for f in parsed if f.stated_entries and len(f.holdings) == f.stated_entries)
            / max(sum(1 for f in parsed if f.stated_entries), 1)
        ),
        "holdings_rows": int(len(panel)),
    }
    if validate_cusip and validate_price and len(panel):
        got = implied_price(panel, validate_cusip)
        report["validation"] = {
            "cusip": validate_cusip,
            "implied": got,
            "expected": validate_price,
            "passed": bool(abs(got / validate_price - 1) < 0.02) if got == got else False,
        }
    return panel, report


# ---------------------------------------------------------------- validation

def implied_price(holdings: pd.DataFrame, cusip: str) -> float:
    """The price this filing implies for one security, in whole dollars.

    The check that caught the filing-quarter error. If a quarter's filings imply a price
    that does not match the close on ``period_of_report``, something upstream is wrong and
    the panel should not be used.
    """
    h = holdings[
        holdings["cusip"].str.startswith(cusip[:8]) & (holdings["kind"] == "SH")
    ]
    h = h[(h["shares"] > 0) & (h["value_usd"] > 0)]
    if h.empty:
        return float("nan")
    return float(np.median(h["value_usd"] / h["shares"]))
