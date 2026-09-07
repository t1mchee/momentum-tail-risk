"""The numeral canary: no number reaches the page unless a computed field produced it.

Section 2.1 states this as a principle -- deterministic code produces every statistic, and a
tokeniser rejects any page where a numeral in the prose fails to match a computed field. The
principle was asserted for some time before it was implemented, and the page it describes turned
out to be largely hand-typed. This is the implementation.

Two ideas and nothing more.

**A field knows where it came from.** Every number destined for the page is registered with the
value, the string it renders as, the file that produced it, the command that reproduces it and,
where the number rests on a document, the verbatim span and the accession that carries it. The
page is then formatted from those render strings rather than from literals.

**A numeral that matches no field fails the build.** After the page is composed, every numeral in
it is matched against the registry or against a small set of DEFINITIONAL constants -- the
horizon, the quantile, the thresholds that define the event -- each of which is declared here
with the reason it is definitional rather than claimed. Anything else is a hand-typed number and
the build stops.

The registry does a second job for free. A page whose every number knows its own provenance can
be rendered so a reader clicks a number and is shown the field behind it, which is what makes an
example output traceable evidence rather than a claim about traceability.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field as _field

#: Numerals that are part of the design's vocabulary rather than claims about the world. Each
#: carries the reason it is exempt, because an unexplained allowlist is how a tokeniser stops
#: being a check.
DEFINITIONAL: dict[str, str] = {
    "10": "the event horizon in trading days, fixed in the registration",
    "5": "the quantile the severity is reported at, fixed in the registration",
    "21": "the second horizon printed beside the first, fixed in the registration",
    "20": "one day in twenty; the plain-English restatement of the 5 percent quantile",
    "1": "ordinal, as in a one-in-twenty event or a ranked list position",
    "2": "ordinal, as in a ranked list position",
    "3": "ordinal, as in a ranked list position",
    "4": "ordinal, as in a ranked list position",
    "8": "the form number in '8-K', not a quantity",
}

#: A date is not a numeral in the sense that matters here, and neither is a name that happens
#: to contain digits. Both are stripped before the scan.
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NAMES = re.compile(
    r"\b8-K\b|COVID-19|\bMTUM\b|\bCOR1M\b|\bVIX\b|Russell 3000|\b12-1\b|\bItem 1A\b")
#: A decimal point counts only when digits follow it. Without that, a sentence-final numeral
#: swallows the full stop -- "on 0." -- and then matches no registered field, so the canary
#: rejects a page over punctuation.
_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


@dataclass
class Field:
    """One number on the page, and everything needed to check it."""

    key: str
    value: float | int | str
    text: str                      # exactly how it renders in the prose
    label: str                     # what a reader should be told it is
    source: str                    # the artifact that produced it
    reproduce: str                 # the command that regenerates that artifact
    span: dict | None = None       # {quote, ticker, accession, accepted_at} when documentary
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Registry:
    """Every field the page is allowed to print, keyed by the string it renders as."""

    fields: list[Field] = _field(default_factory=list)

    def add(self, key, value, text, label, source, reproduce, span=None, note="") -> str:
        """Register a number and return its render string, so a caller cannot print an
        unregistered one by accident: the only way to get the text is to register it."""
        self.fields.append(Field(key, value, str(text), label, source, reproduce, span, note))
        return str(text)

    # -- convenience formatters, each of which registers before it returns ----------------
    def pct(self, key, value, label, source, reproduce, *, dp=1, sign=False, span=None) -> str:
        t = f"{value:+.{dp}%}" if sign else f"{value:.{dp}%}"
        return self.add(key, value, t, label, source, reproduce, span)

    def num(self, key, value, label, source, reproduce, *, dp=0, comma=False, span=None) -> str:
        t = f"{value:,.{dp}f}" if comma else f"{value:.{dp}f}"
        return self.add(key, value, t, label, source, reproduce, span)

    def by_text(self) -> dict[str, list[Field]]:
        """Rendered text -> every registered field that renders it, in registration order.

        This used to return one field per text, which silently picked a winner: the page prints
        `10` as the event horizon, as the GICS sector count and as the loser-leg VaR's decade,
        and every one of them was labelled with whichever field was registered last. The values
        were right and the labels on repeated numerals were not, which is worse than an unlabelled
        number because it reads as traced.

        A text can legitimately map to several fields, and nothing in the page's own text can tell
        the occurrences apart. So the map keeps them all and the caller states the ambiguity rather
        than resolving it by luck.
        """
        out: dict[str, list[Field]] = {}
        for f in self.fields:
            seen = out.setdefault(f.text, [])
            if not any(g.key == f.key for g in seen):
                seen.append(f)
        return out


def strip_verbatim(page: str, reg: "Registry") -> str:
    """Remove every registered verbatim span before the scan.

    A number inside a quoted span belongs to the filer who wrote it, not to the system. "expects
    demand to remain suppressed and plateau at levels of around 50%, relative to 2019 levels"
    contains three numerals and the page asserts none of them: it asserts that a named company
    said this on a named date, and the quote gate has already checked that against the source.
    Requiring those numerals to match a computed field would be asking the wrong question, and
    allowlisting them one at a time would let a real hand-typed number hide among them.

    So spans are excised, and what remains is exactly the prose the system wrote in its own
    voice. That is the text the canary is for.
    """
    out = page
    for f in reg.fields:
        if not f.span:
            continue
        for key in ("rendered", "quote"):
            t = f.span.get(key)
            if t:
                out = out.replace(str(t), " ")
    return out


def numerals(page: str) -> list[str]:
    """Every numeral in the prose, with dates and digit-bearing names removed first."""
    clean = _NAMES.sub(" ", _DATE.sub(" ", page))
    return [m.group(0) for m in _NUM.finditer(clean)]


def _norm(s: str) -> str:
    return s.replace(",", "").replace("+", "").lstrip("0") or "0"


def _resolve(candidates: list["Field"]) -> dict:
    """The field dict for one numeral occurrence, or a stated ambiguity when several fit."""
    if len(candidates) == 1:
        return candidates[0].as_dict()
    d = candidates[0].as_dict()
    d["ambiguous"] = True
    d["label"] = (f"AMBIGUOUS: {len(candidates)} registered fields render this text; "
                  f"the page's wording cannot tell this occurrence apart from the others")
    d["candidates"] = [{"key": c.key, "label": c.label, "value": c.value,
                        "source": c.source, "reproduce": c.reproduce} for c in candidates]
    return d


def check(page: str, reg: Registry) -> dict:
    """Match every numeral against a registered field or a declared definitional constant.

    Returns the verdict and, for a page that passes, the provenance map that lets a reader click
    a number and see what produced it.
    """
    lookup = {_norm(t): fs for t, fs in reg.by_text().items()}
    seen, unmatched, matched = [], [], {}
    for n in numerals(strip_verbatim(page, reg)):
        seen.append(n)
        k = _norm(n)
        if k in lookup:
            matched[n] = _resolve(lookup[k])
        elif n.strip("+-%") in DEFINITIONAL:
            matched[n] = {"key": "definitional", "text": n,
                          "label": DEFINITIONAL[n.strip("+-%")],
                          "source": "the registration", "reproduce": "", "value": n}
        else:
            unmatched.append(n)
    return {
        "numerals": len(seen),
        "distinct": len(set(seen)),
        "registered_fields": len(reg.fields),
        "matched": len(set(seen)) - len(set(unmatched)),
        "unmatched": sorted(set(unmatched)),
        "passes": not unmatched,
        "provenance": matched,
    }


def _excluded(page: str, reg: "Registry") -> list[tuple[int, int]]:
    """Character ranges the scan must not look inside: verbatim spans, dates, digit-bearing names."""
    out: list[tuple[int, int]] = []
    for f in reg.fields:
        if not f.span:
            continue
        for key in ("rendered", "quote"):
            t = f.span.get(key)
            if not t:
                continue
            start = page.find(str(t))
            while start != -1:
                out.append((start, start + len(str(t))))
                start = page.find(str(t), start + 1)
    for rx in (_DATE, _NAMES):
        out += [(m.start(), m.end()) for m in rx.finditer(page)]
    return out


def segments(page: str, reg: "Registry") -> list[dict]:
    """Split the page into consecutive spans, each either plain text or one traced numeral.

    This is the ONLY thing a renderer should use. An earlier viewer highlighted numerals by
    replacing each matched string throughout the document, which put a span on the `10` inside
    `2020-10-31`, and then on digits inside its own placeholder tokens; the page rendered as
    garbage while the canary reported PASS, because the canary checked the source text and the
    reader was looking at the DOM. Offsets cannot do that, and `test_segments_round_trip` asserts
    that reassembling the segments reproduces the page character for character, which makes the
    whole class of fault unrepresentable rather than merely fixed.
    """
    lookup = {_norm(t): fs for t, fs in reg.by_text().items()}
    skip = _excluded(page, reg)

    def inside(a: int, b: int) -> bool:
        return any(s <= a and b <= e for s, e in skip)

    out, cursor = [], 0
    for m in _NUM.finditer(page):
        a, b = m.span()
        if inside(a, b):
            continue
        tok = m.group(0)
        cands = lookup.get(_norm(tok))
        if cands is None:
            if tok.strip("+-%") not in DEFINITIONAL:
                continue
            field = {"key": "definitional", "text": tok, "value": tok,
                     "label": DEFINITIONAL[tok.strip("+-%")],
                     "source": "the registration", "reproduce": "", "span": None}
        else:
            field = _resolve(cands)
        if a > cursor:
            out.append({"text": page[cursor:a]})
        out.append({"text": tok, "field": field})
        cursor = b
    if cursor < len(page):
        out.append({"text": page[cursor:]})
    return out


def document_marks(page: str, reg: "Registry") -> list[dict]:
    """Where each evidence document's ticker appears on its own evidence line, by offset."""
    marks = []
    for f in reg.fields:
        sp = f.span
        if not sp or not sp.get("ticker"):
            continue
        m = re.search(rf"(?m)^(\s*)({re.escape(sp['ticker'])})(\s+accepted)", page)
        if m:
            marks.append({"start": m.start(2), "end": m.end(2), "ticker": sp["ticker"],
                          "document": sp})
    return marks


class NumeralCanary(AssertionError):
    """Raised when a numeral on the page matches no computed field. The build stops."""


def enforce(page: str, reg: Registry) -> dict:
    r = check(page, reg)
    if not r["passes"]:
        raise NumeralCanary(
            f"{len(r['unmatched'])} numeral(s) on the page match no computed field and no "
            f"declared definitional constant: {r['unmatched']}. Either register the field that "
            f"produces each, or state why it is definitional in numerals.DEFINITIONAL. A "
            f"hand-typed number on this page is the fault this check exists to catch.")
    return r
