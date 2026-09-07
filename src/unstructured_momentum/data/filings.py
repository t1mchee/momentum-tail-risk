"""A reusable corpus of SEC filing sections, structured and point-in-time.

Why this exists as a module rather than a script
------------------------------------------------
Four ad-hoc pulls have been run in this project and each one repeated a mistake the previous
one had already found: a section that is really a cross-reference, a filing index that
returns only its most recent block, a slice that silently collapses to a fragment, a
universe that does not intersect the book it was meant to describe. Each fix lived in the
script that discovered it and travelled no further.

So the extraction rules live here, once, with the defences attached:

* **Cross-references are resolved.** A section reading "these disclosures appear in
  Management's Discussion" is a pointer, not content. Banks satisfy Item 7A this way and an
  earlier pull read the pointer, silently returning "no interest-rate exposure" for half the
  financial companies in a book.
* **The filing index is paged.** ``filings.recent`` holds roughly the last thousand filings
  and older ones sit in separate files. A pull that ignores them comes back looking like a
  growing panel rather than a truncated one.
* **Extractions are size-checked against their own neighbours.** A section that is a third
  or three times the length of the same company's previous one is a parsing failure. In a
  change measure a single bad extraction corrupts two comparisons, not one.
* **A manifest records what was asked for**, not only what arrived, because a capture that
  quietly collects the wrong population looks exactly like progress.

Layout
------
``data/raw/filings/``
    ``index.parquet``      one row per filing: identifiers, dates, section lengths, flags
    ``text/{accession}.json.gz``  extracted sections AND the stripped full document
    ``manifest.json``      universe requested, coverage achieved, failures by reason

The full stripped document is kept alongside the sections, at roughly 100 kilobytes
compressed. That is the difference between a corpus that answers one question and one that
answers the next: re-extracting a different section costs seconds, re-fetching it costs
hours. Any future work wanting Item 3, or the management discussion, or the whole document
for an embedding, needs no network at all.

The index is small enough to load whole and join against anything. The text is addressed by
accession so nothing depends on a filename convention.

Point-in-time
-------------
Every row carries ``accepted_at``, the second-level timestamp at which the regulator took
receipt. That, not the period covered and not the filing date, is when the text became
knowable. Filings accepted after 17:30 Eastern are deemed filed the next business day.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from ..config import RAW, USER_AGENT

ROOT = RAW / "filings"
TEXT = ROOT / "text"
INDEX = ROOT / "index.parquet"
MANIFEST = ROOT / "manifest.json"

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

#: SEC fair access is 10 requests a second across ALL endpoints, so the limit is a global
#: rate, not a per-request delay. Sleeping after each fetch wastes most of the budget: a
#: document takes about 0.11 seconds to retrieve, so a 0.13 second sleep yields 4.2 requests
#: a second against a ceiling of 10. A shared token bucket lets several workers share one
#: budget and roughly doubles throughput without going over.
RATE_LIMIT = 8.0
_BUCKET_LOCK = threading.Lock()
_next_slot = 0.0


def _acquire() -> None:
    """One global token bucket, shared by every worker."""
    global _next_slot
    with _BUCKET_LOCK:
        now = time.monotonic()
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + 1.0 / RATE_LIMIT
    if wait:
        time.sleep(wait)

#: A section whose opening points elsewhere is a reference, not content.
XREF = re.compile(
    r"(are|is)\s+(set forth|included|incorporated|contained|presented)"
    r"|\bsee\s+(Item|Management|Note|Part)"
    r"|incorporated (herein )?by reference"
    r"|refer(ence|red)?\s+to\s+(Item|Management|Note)", re.I)

RATE_ISH = re.compile(r"interest rate|LIBOR|basis point|variable[- ]rate|floating", re.I)

#: Extractions outside this band relative to a neighbour are parsing failures.
LEN_BAND = (1 / 3, 3.0)

#: Section boundaries differ by form and getting this wrong is silent. In an annual report
#: Item 1A is followed by Item 1B or Properties. In a quarterly report the risk-factor
#: update sits in Part II and is followed by Unregistered Sales. Using the annual
#: terminators on a quarterly filing extracted 65,740 characters of Microsoft's financial
#: statements and called it a risk-factor section.
#: Stripping tags collapses "<b>Bu</b>siness" to "Bu siness", so a heading word can carry an
#: internal space that no amount of separator-class widening will match. Four of the five
#: Item 1 misses audited on 2026-08-25 were this: "Bu siness", "Ri sk Factors", "FA CTORS",
#: "Unres olved". The anchor therefore allows whitespace BETWEEN LETTERS of the keyword,
#: bounded at three characters because the collapse leaves single spaces and an unbounded
#: gap would let a word match across a table.
def _loose(word: str) -> str:
    """A regex matching `word` even when tag boundaries have split it."""
    return r"\s{0,3}".join(re.escape(c) for c in word)


#: Separators seen between the item number and its title. The colon cost MSI its business
#: description outright: "Item 1: Business" was the only body heading in the filing, and the
#: class had no colon, so the sole surviving candidate was the contents page.
_SEP = r"[\.\:\s\-\u2013\u2014]"


SECTIONS = {
    ("10-K", "item_1a"): ([rf"Item{_SEP}{{0,3}}1A{_SEP}{{0,3}}" + _loose("Risk Factors")],
                          [rf"Item{_SEP}{{0,3}}1B", rf"Item{_SEP}{{0,3}}2{_SEP}{{0,3}}" + _loose("Propert"),
                           rf"Item{_SEP}{{0,3}}6{_SEP}{{0,3}}" + _loose("Exhibit")]),
    ("10-K", "item_7a"): ([rf"Item{_SEP}{{0,3}}7A{_SEP}{{0,3}}" + _loose("Quantitative and Qualitative")],
                          [rf"Item{_SEP}{{0,3}}8{_SEP}{{0,3}}" + _loose("Financial Statements")]),
    ("10-Q", "item_1a"): ([rf"Item{_SEP}{{0,3}}1A{_SEP}{{0,3}}" + _loose("Risk Factors")],
                          [rf"Item{_SEP}{{0,3}}2{_SEP}{{0,3}}" + _loose("Unregistered"),
                           rf"Item{_SEP}{{0,3}}3{_SEP}{{0,3}}" + _loose("Default"),
                           rf"Item{_SEP}{{0,3}}4{_SEP}{{0,3}}" + _loose("Mine"),
                           rf"Item{_SEP}{{0,3}}5{_SEP}{{0,3}}" + _loose("Other"),
                           rf"Item{_SEP}{{0,3}}6{_SEP}{{0,3}}" + _loose("Exhibit"), r"SIGNATURES?"]),
    ("10-Q", "item_7a"): ([rf"Item{_SEP}{{0,3}}3{_SEP}{{0,3}}" + _loose("Quantitative and Qualitative")],
                          [rf"Item{_SEP}{{0,3}}4{_SEP}{{0,3}}" + _loose("Controls")]),
    # Item 1 is the section the product-similarity literature uses -- the Hoberg-Phillips
    # lineage builds its entire text-based industry classification from the business
    # description, and every published result that recovers shared economic exposure from
    # filings uses it rather than the risk factors. It was never extracted here.
    ("10-K", "item_1"): ([rf"Item{_SEP}{{1,3}}1{_SEP}{{1,3}}" + _loose("Business") + r"\b"],
                         [rf"Item{_SEP}{{0,3}}1A{_SEP}{{0,3}}" + _loose("Risk Factors"),
                          rf"Item{_SEP}{{0,3}}2{_SEP}{{0,3}}" + _loose("Propert")]),
    # Item 7 is management's own discussion, where forward-looking language lives. Item 7A,
    # already held, is the narrow market-risk subsection of it.
    ("10-K", "item_7"): ([rf"Item{_SEP}{{1,3}}7{_SEP}{{1,3}}" + _loose("Management") + r".{0,25}" + _loose("Discussion")],
                         [rf"Item{_SEP}{{0,3}}7A{_SEP}{{0,3}}" + _loose("Quantitative"),
                          rf"Item{_SEP}{{0,3}}8{_SEP}{{0,3}}" + _loose("Financial")]),
    ("10-Q", "item_2"): ([rf"Item{_SEP}{{1,3}}2{_SEP}{{1,3}}" + _loose("Management") + r".{0,25}" + _loose("Discussion")],
                         [rf"Item{_SEP}{{0,3}}3{_SEP}{{0,3}}" + _loose("Quantitative"),
                          rf"Item{_SEP}{{0,3}}4{_SEP}{{0,3}}" + _loose("Controls")]),
}

#: A quarterly filing that says the risk factors are unchanged is not a failed extraction.
#: It is the informative case: the published strategy is long the companies that leave a
#: section alone. Recorded as a flag rather than discarded as a short section.
NO_CHANGE = re.compile(
    r"no material (changes?|updates?)[^.]{0,120}risk factors"
    r"|risk factors[^.]{0,120}(have )?not materially changed"
    r"|there have been no material changes[^.]{0,80}risk factors", re.I)

SECTION_CAP = 200_000

#: Plausible section sizes by form, in characters. Filing layouts vary more than any regex
#: can track, so the goal is not perfect extraction but DETECTABLE failure: an annual
#: risk-factor section running to the character cap, or a quarterly update the size of a
#: whole document, is flagged rather than silently stored and used.
PLAUSIBLE = {
    ("10-K", "item_1a"): (4_000, 180_000),
    ("10-K", "item_7a"): (400, 60_000),
    ("10-Q", "item_1a"): (200, 40_000),
    ("10-Q", "item_7a"): (200, 40_000),
    # Bands for the three sections added later, set from the measured 2nd and 98th percentiles
    # of 500 sampled filings rather than by eye, with headroom. Anything outside is a missed
    # terminator or a cross-reference stub, not a short section.
    ("10-K", "item_1"): (3_000, 160_000),
    ("10-K", "item_7"): (6_000, 190_000),
    ("10-Q", "item_2"): (8_000, 160_000),
}


def implausible(form: str, name: str, n: int) -> bool:
    lo, hi = PLAUSIBLE.get((form, name), (0, SECTION_CAP))
    return not (lo <= n <= hi)

def _get(url: str, session: requests.Session, timeout: int = 120) -> str:
    _acquire()
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def strip_html(html: str) -> str:
    t = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", html)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&#\d+;|&[a-z]+;", " ", t)
    return re.sub(r"\s+", " ", t)


#: Words that should be dense in a real risk-factor or market-risk section within its first
#: couple of thousand characters. A slice that starts at a cross-reference or a contents page
#: contains page numbers and legalese instead.
_SECTION_PROSE = re.compile(
    r"\b(risk|risks|adverse|adversely|could|may|uncertain|exposure|fluctuat|"
    r"interest rate|volatil|decline|loss|losses)\b", re.I)

#: A real section reads as prose. A contents page reads as "Item 1B ... 45 Item 2 ... 45".
_TOC_ISH = re.compile(r"Item\s*\d+[A-B]?\.?\s+[A-Z][^.]{0,60}?\s+\d{1,3}\s+Item\s*\d", re.I)


#: Prose profiles PER SECTION. The original discriminator was written for risk factors and used
#: risk vocabulary -- risk, adverse, could, may, uncertain, loss. Reusing it for Item 1, which
#: describes products and markets, rejected 84 percent of valid business descriptions, and for
#: Item 7 it rejected 70 percent of valid MD&A. A section is recognised by ITS OWN register.
_PROSE_BY_SECTION = {
    # Broadened 2026-08-25. The original list was drawn from what a business description
    # discusses; it omitted the words a business description OPENS with. Chipotle's and
    # Entegris' Item 1 bodies were located correctly and then thrown away for scoring five
    # against a threshold of six, because their first 2,500 characters are corporate
    # self-introduction -- incorporated, founded, headquartered, stores -- before they reach
    # products and customers. The threshold stays at six; the register is what was wrong.
    "item_1": re.compile(
        r"\b(products?|services?|customers?|clients?|markets?|segments?|operations?|"
        r"competiti\w*|employees?|manufactur\w*|distribut\w*|brands?|subsidiar\w*|"
        r"revenues?|business|compan(y|ies)|incorporated|founded|headquarter\w*|stores?|"
        r"restaurants?|solutions?|technolog\w*|offer\w*|sell|sales)\b", re.I),
    "item_2": re.compile(
        r"\b(revenues?|increase[sd]?|decrease[sd]?|compared|quarter|margin|expenses?|cash|"
        r"operations?|results?|period|million|growth)\b", re.I),
    "item_7": re.compile(
        r"\b(revenues?|increase[sd]?|decrease[sd]?|compared|fiscal|margin|expenses?|cash flow|"
        r"liquidity|operations?|results?|million|growth)\b", re.I),
}


def _looks_like_section(body: str, name: str = "item_1a") -> bool:
    """Whether a candidate slice reads as the section rather than a pointer to it.

    The threshold is deliberately the same across sections; what changes is the vocabulary the
    section is expected to be written in.
    """
    head = body[:2500]
    if not head:
        return False
    if _TOC_ISH.search(head):
        return False
    pat = _PROSE_BY_SECTION.get(name, _SECTION_PROSE)
    return len(pat.findall(head)) >= 6


def slice_section(text: str, name: str, form: str = "10-K",
                  cap: int = SECTION_CAP) -> str:
    """Take the first heading occurrence that actually reads as the section.

    An earlier version took the LAST occurrence, documented as skipping the table of
    contents. It does that, and then fails on the far more common case: a later
    cross-reference reading "Item 1A - Risk Factors of this Form 10-K. Except to the extent
    required by applicable law". An audit found 24.7 percent of an embedded corpus starting
    at a cross-reference or a contents page, and the contents-page slices formed a
    near-duplicate boilerplate clique at internal cosine 0.477 against 0.012 for the pool.

    Every candidate heading is now tried in order and the first whose opening reads as prose
    is taken. Anything reaching the cap is refused rather than truncated, because a slice
    that long has run past its own terminator into the financial statements.
    """
    # dict.get evaluates its default EAGERLY, so the old one-liner computed
    # SECTIONS[("10-K", name)] before returning the (form, name) entry that existed -- raising
    # KeyError for every 10-Q Item 2 call, since there is no 10-K Item 2. The caller swallowed
    # it and recorded zero extractions, which read as a failed pattern rather than a crash.
    key = (form, name)
    if key not in SECTIONS:
        key = ("10-K", name)
    if key not in SECTIONS:
        raise KeyError(f"no section pattern for form {form!r}, section {name!r}")
    starts, ends = SECTIONS[key]
    pos = sorted(m.start() for p in starts for m in re.finditer(p, text, re.I))
    if not pos:
        return ""
    for start in pos:
        tail = text[start: start + 500_000]
        stop = [m.start() for p in ends for m in re.finditer(p, tail, re.I) if m.start() > 2000]
        body = tail[:min(stop)] if stop else tail
        if len(body) >= cap:
            continue          # ran past its terminator; not a section
        if _looks_like_section(body, name):
            return body[:cap]
    return ""


def resolve_reference(text: str, cap: int = 80_000) -> str:
    """Follow a cross-reference into the management discussion and keep the risk passages.

    Financial companies routinely satisfy Item 7A by pointing at Item 7. Reading the pointer
    returns a section that parses cleanly and says nothing.
    """
    pos = [m.start() for m in
           re.finditer(r"Item\s*7[\.\s\-—]*Management.{0,25}Discussion", text, re.I)]
    if not pos:
        return ""
    body = text[max(pos): max(pos) + 500_000]
    stop = [m.start() for m in re.finditer(r"Item\s*8[\.\s\-—]*Financial Statements", body, re.I)
            if m.start() > 3000]
    body = body[:min(stop)] if stop else body[:300_000]
    words = body.split()
    keep = [" ".join(words[i:i + 80]) for i in range(0, len(words), 80)
            if RATE_ISH.search(" ".join(words[i:i + 80]))]
    return " ".join(keep)[:cap]


def is_pointer(section: str) -> bool:
    if len(section) < 600:
        return True
    density = len(RATE_ISH.findall(section)) / max(len(section.split()) / 1000, 1)
    return bool(XREF.search(section[:400])) or density < 0.8


def filing_list(cik: int, s: requests.Session, *, forms=("10-K", "10-Q"),
                start: str = "2016-01-01", end: str = "2024-12-31") -> list[dict]:
    """Every matching filing, following the paged index rather than only the recent block."""
    out: list[dict] = []
    try:
        sub = json.loads(_get(SUBMISSIONS.format(cik=cik), s, timeout=45))
    except Exception:                                          # noqa: BLE001
        return out

    blocks = [sub.get("filings", {}).get("recent", {})]
    for extra in sub.get("filings", {}).get("files", []):
        # Older filings live in separate files. Ignoring them truncates history in a way
        # that reads as a growing panel rather than a missing one.
        if extra.get("filingTo", "") < start:
            continue
        try:
            blocks.append(json.loads(
                _get(f"https://data.sec.gov/submissions/{extra['name']}", s, timeout=45)))
        except Exception:                                      # noqa: BLE001
            continue

    for b in blocks:
        n = len(b.get("accessionNumber", []))
        for i in range(n):
            if b["form"][i] not in forms:
                continue
            if not (start <= b["filingDate"][i] <= end):
                continue
            out.append({
                "accession": b["accessionNumber"][i],
                "form": b["form"][i],
                "filing_date": b["filingDate"][i],
                "period": (b.get("reportDate") or [None] * n)[i],
                "accepted_at": (b.get("acceptanceDateTime") or [None] * n)[i],
                "primary_doc": b["primaryDocument"][i],
            })
    return out


@dataclass
class PullReport:
    requested: list[str] = field(default_factory=list)
    filings: int = 0
    failures: dict = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.failures[reason] = self.failures.get(reason, 0) + 1


def _fetch_one(tk: str, cik: int, f: dict, s: requests.Session) -> dict | None:
    """Fetch, strip and section one filing. Returns None on any per-item failure.

    Per-item isolation, not per-batch: one unparseable document must never end a run whose
    output cannot be regenerated cheaply.
    """
    acc = f["accession"].replace("-", "")
    try:
        raw = _get(ARCHIVE.format(cik=int(cik), acc=acc, doc=f["primary_doc"]), s)
    except Exception:                                          # noqa: BLE001
        return {"error": "document fetch failed"}
    txt = strip_html(raw)
    secs, flags = {}, []
    if NO_CHANGE.search(txt):
        flags.append("risk_factors_declared_unchanged")
    for name in ("item_1a", "item_7a"):
        body = slice_section(txt, name, form=f["form"])
        if name == "item_7a" and is_pointer(body):
            resolved = resolve_reference(txt)
            if len(resolved) > 600:
                body = resolved
                flags.append("item_7a_reference_resolved")
        secs[name] = body
        if body and implausible(f["form"], name, len(body)):
            flags.append(f"{name}_implausible_length")
    if (len(secs["item_1a"]) < 2000 and len(secs["item_7a"]) < 600
            and "risk_factors_declared_unchanged" not in flags):
        return {"error": "no usable section found"}
    with gzip.open(TEXT / f"{acc}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"ticker": tk, "accession": f["accession"], "form": f["form"],
                   "filing_date": f["filing_date"], "accepted_at": f["accepted_at"],
                   "full_text": txt, **secs}, fh)
    return {"ticker": tk, "cik": int(cik), "accession": f["accession"], "form": f["form"],
            "filing_date": f["filing_date"], "period": f["period"],
            "accepted_at": f["accepted_at"], "item_1a_chars": len(secs["item_1a"]),
            "item_7a_chars": len(secs["item_7a"]), "flags": ";".join(flags),
            "n_chars_doc": len(txt)}


def pull(universe: list[str], cik_map: dict[str, int], *, forms=("10-K", "10-Q"),
         start: str = "2016-01-01", end: str = "2024-12-31",
         workers: int = 4, log_every: int = 25) -> PullReport:
    """Fetch and extract for a universe. Resumable, threaded, and rate-limited globally.

    Workers share one token bucket, so adding threads increases utilisation of the allowed
    rate rather than exceeding it. Throughput is set by ``RATE_LIMIT``, not by ``workers``.
    """
    TEXT.mkdir(parents=True, exist_ok=True)
    s = session()
    rep = PullReport(requested=list(universe))
    have = {p.name.split(".")[0] for p in TEXT.glob("*.json.gz")}
    rows: list[dict] = pd.read_parquet(INDEX).to_dict("records") if INDEX.exists() else []

    def company(tk: str) -> list[dict]:
        cik = cik_map.get(tk)
        if cik is None:
            return [{"error": "no filer identifier for ticker"}]
        out = []
        for f in filing_list(int(cik), s, forms=forms, start=start, end=end):
            if f["accession"].replace("-", "") in have:
                out.append({"cached": True})
                continue
            out.append(_fetch_one(tk, int(cik), f, s))
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, res in enumerate(ex.map(company, universe), 1):
            for r in res:
                if r is None:
                    continue
                if r.get("cached"):
                    rep.filings += 1
                elif "error" in r:
                    rep.note(r["error"])
                else:
                    rows.append(r)
                    rep.filings += 1
            if n % log_every == 0:
                pd.DataFrame(rows).to_parquet(INDEX, index=False)
                print(f"  {n}/{len(universe)} companies, {rep.filings} filings", flush=True)

    pd.DataFrame(rows).to_parquet(INDEX, index=False)
    MANIFEST.write_text(json.dumps({
        "requested_companies": len(universe), "filings": rep.filings,
        "failures": rep.failures, "forms": list(forms), "start": start, "end": end,
        "rate_limit_per_second": RATE_LIMIT, "workers": workers,
    }, indent=1), encoding="utf-8")
    return rep


def load_index() -> pd.DataFrame:
    df = pd.read_parquet(INDEX)
    for c in ("filing_date", "period"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["accepted_at"] = pd.to_datetime(df["accepted_at"], errors="coerce", utc=True)
    return df


def load_text(accession: str) -> dict:
    with gzip.open(TEXT / f"{accession.replace('-', '')}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def flag_length_outliers(idx: pd.DataFrame, section: str = "item_1a") -> pd.DataFrame:
    """Mark extractions implausible against the same company's neighbouring filing.

    Both sides of a bad comparison are marked, not one: in a change measure a single failed
    extraction corrupts the pair ending on it and the pair starting from it.
    """
    col = f"{section}_chars"
    out = idx.sort_values(["ticker", "filing_date"]).copy()
    out["ratio_prev"] = out.groupby("ticker")[col].transform(lambda v: v / v.shift())
    bad = (out["ratio_prev"] < LEN_BAND[0]) | (out["ratio_prev"] > LEN_BAND[1])
    out["length_suspect"] = bad | bad.shift(-1).fillna(False)
    return out


def coverage(idx: pd.DataFrame, freq: str = "QE") -> pd.DataFrame:
    """Companies with a filing in force at each period end. Coverage is an output."""
    rows = []
    for q in pd.date_range(idx.filing_date.min(), idx.filing_date.max(), freq=freq):
        w = idx[(idx.filing_date <= q) & (idx.filing_date >= q - pd.Timedelta(days=400))]
        rows.append({"period": q, "companies": w.ticker.nunique(), "filings": len(w)})
    return pd.DataFrame(rows)
