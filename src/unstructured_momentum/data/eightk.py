"""8-K bodies: the highest-frequency issuer text that exists, and the one we never fetched.

This project has used 8-K METADATA since early on -- Item 2.02 acceptance timestamps drive the
earnings calendar -- while the bodies were never downloaded. That is a real gap for anything
asking when language moves. Every text source here is filings: lawyer-mediated, annually
recycled, released on a compliance calendar that has nothing to do with when a risk became real.
Item 1A updates quarterly at best. An 8-K is filed within four business days of a material event
and says what happened.

What the item codes buy
-----------------------
2.02 is the earnings release and is already used for its timestamp. The rest is where material
events live, and they are the ones worth reading: 2.05 and 2.06 are impairments and exit costs,
1.01 and 1.02 are material agreements entered and terminated, 2.03 and 2.04 are debt obligations
and acceleration events, 5.02 is executive departure, 7.01 and 8.01 are the regulation FD and
other-events buckets where guidance withdrawals appear.

Point-in-time
-------------
``acceptanceDateTime`` is the second-level moment the regulator took receipt and is the most
precise availability marker in the project. It is carried through unchanged; nothing here infers
an earlier availability from a document date, because a document date is when someone typed it.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from ..config import RAW, USER_AGENT

ROOT = RAW / "edgar_8k"
TEXT = ROOT / "text"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
INDEX = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/index.json"

#: SEC asks for no more than ten requests a second; eight leaves headroom and matches the
#: filing loader, so the two cannot collectively exceed the limit when run together.
RATE_LIMIT = 8.0
_next_slot = 0.0

#: Already consumed for its timestamp by the earnings calendar; excluded here by default so the
#: body pull is about material events rather than re-reading earnings releases.
EARNINGS_ITEM = "2.02"

#: Item codes worth the bytes. Everything else is administrative.
MATERIAL_ITEMS = (
    "1.01", "1.02", "1.03",          # material agreements, termination, bankruptcy
    "2.03", "2.04", "2.05", "2.06",  # debt obligations, acceleration, exit costs, impairment
    "4.02",                          # non-reliance on previously issued financials
    "5.02",                          # departure of directors or principal officers
    "7.01", "8.01",                  # Regulation FD, other events -- where guidance moves
)


def _acquire() -> None:
    global _next_slot
    now = time.monotonic()
    wait = _next_slot - now
    if wait > 0:
        time.sleep(wait)
    _next_slot = max(now, _next_slot) + 1.0 / RATE_LIMIT


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def strip_html(html: str) -> str:
    from .filings import strip_html as _s
    return _s(html)


@dataclass
class PullReport:
    requested: int = 0
    filings: int = 0
    failures: dict = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.failures[reason] = self.failures.get(reason, 0) + 1


def filing_list(cik: int, s: requests.Session, *, start: str, end: str,
                items: tuple[str, ...] = MATERIAL_ITEMS,
                include_earnings: bool = False) -> list[dict]:
    """8-K rows for one company, filtered to the item codes worth reading.

    Only the ``recent`` block is read, as the earnings calendar does. That holds about a
    thousand filings, which at a typical fifteen to twenty-five 8-Ks a year covers decades. A
    company exceeding it would show as missing EARLY history rather than as wrong dates, which
    is the failure mode to prefer.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / f"sub_{cik:010d}.json"
    if not p.exists():
        _acquire()
        try:
            r = s.get(SUBMISSIONS.format(cik=cik), timeout=30)
            r.raise_for_status()
            p.write_text(r.text, encoding="utf-8")
        except (requests.RequestException, ValueError):
            return []
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        rec = d["filings"]["recent"]
    except (KeyError, ValueError):
        return []

    n = len(rec.get("form", []))
    got = rec.get("items", [""] * n)
    out = []
    for i in range(n):
        if rec["form"][i] != "8-K":
            continue
        codes = got[i] if i < len(got) else ""
        if not include_earnings and codes.strip() == EARNINGS_ITEM:
            continue
        if items and not any(c in codes for c in items):
            continue
        fd = rec["filingDate"][i]
        if not (start <= fd <= end):
            continue
        out.append({"accession": rec["accessionNumber"][i], "filing_date": fd,
                    "accepted_at": rec["acceptanceDateTime"][i], "items": codes,
                    "primary_doc": rec.get("primaryDocument", [""] * n)[i]})
    return out


#: The 8-K's own cover page is mostly registrant boilerplate and inline-XBRL tags. The material
#: content is almost always in an exhibit -- 99.1 is the press release, 99.2 the slides.
_EXHIBIT = re.compile(r"ex(?:hibit)?[-_]?99", re.I)

#: Inline XBRL leaves a run of identifiers and repeated dates at the top of the stripped text.
#: A document is prose from the first occurrence of the form heading onward.
_PROSE_START = re.compile(r"FORM\s+8-K|CURRENT\s+REPORT", re.I)


def _fetch_doc(cik: int, acc: str, doc: str, s: requests.Session) -> str:
    _acquire()
    try:
        r = s.get(ARCHIVE.format(cik=cik, acc_nodash=acc, doc=doc), timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        return ""
    return strip_html(r.text)


def fetch_body(cik: int, row: dict, s: requests.Session, *, with_exhibits: bool = True) -> str:
    """The filing's readable content: cover page PLUS its 99-series exhibits.

    Fetching only the primary document was the first version and it would have built a corpus
    of COVER SHEETS. An 8-K's primary document is the registrant boilerplate -- name, address,
    commission file number, a checkbox list -- wrapped in inline XBRL, and it typically closes
    by pointing at an exhibit. Microsoft's September 2020 filing is 28 kB of cover and 16 kB of
    actual press release in a separate file. The event is in the exhibit.

    The XBRL preamble is dropped by starting at the form heading, which is the first prose in
    the document and cheap to find.
    """
    acc = row["accession"].replace("-", "")
    parts = []
    doc = row.get("primary_doc") or ""
    if doc:
        body = _fetch_doc(cik, acc, doc, s)
        m = _PROSE_START.search(body)
        parts.append(body[m.start():] if m else body)
    if with_exhibits:
        _acquire()
        try:
            j = s.get(INDEX.format(cik=cik, acc_nodash=acc), timeout=30).json()
            names = [i["name"] for i in j.get("directory", {}).get("item", [])]
        except (requests.RequestException, ValueError):
            names = []
        for name in names:
            if _EXHIBIT.search(name) and name.lower().endswith((".htm", ".html", ".txt")):
                parts.append(_fetch_doc(cik, acc, name, s))
    return "\n\n".join(p for p in parts if p).strip()


def pull(universe: list[str], cik_map: dict[str, int], *, start: str = "2016-01-01",
         end: str = "2024-12-31", log_every: int = 25, workers: int = 4) -> PullReport:
    """Resumable body pull. Already-downloaded accessions are skipped without a request.

    Threaded, sharing ONE token bucket, so adding workers raises utilisation of the allowed rate
    rather than exceeding it -- the same design the filing loader uses. Written single-threaded
    first, which ran at 2.1 requests a second against an 8 per second budget: each body needs an
    index fetch, a primary document and its exhibits, all sequential, so latency rather than the
    rate limit set the pace. That was a 45-hour job where the arithmetic floor is 12.

    The floor is worth stating plainly: total requests divided by the SEC's stated ten per
    second. No amount of hardware moves it, and a GPU is irrelevant -- this is HTTP and tag
    stripping, with no tensor work anywhere in the path.
    """
    from concurrent.futures import ThreadPoolExecutor

    TEXT.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in TEXT.glob("*.json.gz")}
    rep = PullReport(requested=len(universe))
    rows: list[dict] = []
    lock = __import__("threading").Lock()

    def one(tk: str) -> None:
        cik = cik_map.get(tk)
        if not cik:
            with lock:
                rep.note("no filer identifier for ticker")
            return
        s = session()
        for f in filing_list(cik, s, start=start, end=end):
            key = f["accession"].replace("-", "")
            if key in have:
                continue
            body = fetch_body(cik, f, s)
            if len(body) < 200:
                with lock:
                    rep.note("no usable body")
                continue
            (TEXT / f"{key}.json.gz").write_bytes(gzip.compress(json.dumps({
                "ticker": tk, "cik": cik, "accession": f["accession"],
                "filing_date": f["filing_date"], "accepted_at": f["accepted_at"],
                "items": f["items"], "body": body}).encode()))
            with lock:
                rows.append({"ticker": tk, "cik": cik, **f, "n_chars": len(body)})
                rep.filings += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, _ in enumerate(ex.map(one, universe), 1):
            if k % log_every == 0:
                print(f"  {k}/{len(universe)} tickers, {rep.filings} bodies", flush=True)
    if rows:
        idx = pd.DataFrame(rows)
        idx["accepted_at"] = pd.to_datetime(idx["accepted_at"], errors="coerce", utc=True)
        idx["filing_date"] = pd.to_datetime(idx["filing_date"], errors="coerce")
        # Merge with whatever is already indexed. The first version overwrote here, so a
        # caller pulling in ticker chunks kept only its last chunk -- 44k bodies on disk,
        # 698 rows addressable. An index must be a running total, never a call report.
        prior = load_index()
        if not prior.empty:
            idx = pd.concat([prior, idx], ignore_index=True)
            idx = idx.drop_duplicates(subset="accession", keep="last")
        idx.to_parquet(ROOT / "index.parquet")
    return rep


def rebuild_index(log_every: int = 5000) -> pd.DataFrame:
    """Reconstruct the index from the bodies on disk.

    Every ``text/*.json.gz`` carries its own metadata, so the index is derivable state.
    This exists because the overwrite bug above orphaned tens of thousands of bodies:
    the recovery is to treat the store, not the index, as the source of truth.
    """
    rows = []
    files = sorted(TEXT.glob("*.json.gz"))
    for k, f in enumerate(files, 1):
        try:
            d = json.loads(gzip.decompress(f.read_bytes()))
        except (OSError, ValueError):
            continue
        rows.append({"ticker": d.get("ticker"), "cik": d.get("cik"),
                     "accession": d.get("accession"), "filing_date": d.get("filing_date"),
                     "accepted_at": d.get("accepted_at"), "items": d.get("items"),
                     "primary_doc": d.get("primary_doc", ""),
                     "n_chars": len(d.get("body", ""))})
        if log_every and k % log_every == 0:
            print(f"  {k}/{len(files)} bodies read", flush=True)
    idx = pd.DataFrame(rows)
    if not idx.empty:
        idx["accepted_at"] = pd.to_datetime(idx["accepted_at"], errors="coerce", utc=True)
        idx["filing_date"] = pd.to_datetime(idx["filing_date"], errors="coerce")
        idx = idx.drop_duplicates(subset="accession", keep="last")
        idx.to_parquet(ROOT / "index.parquet")
    return idx


def load_index() -> pd.DataFrame:
    p = ROOT / "index.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_body(accession: str) -> dict:
    f = TEXT / f"{accession.replace('-', '')}.json.gz"
    return json.loads(gzip.decompress(f.read_bytes())) if f.exists() else {}
