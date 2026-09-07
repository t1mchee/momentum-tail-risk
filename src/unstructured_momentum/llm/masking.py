"""Structural masking for the narrative-comparison layer, with its residue counted.

This project measured that removing dates does not hide the date: a model placed forty of
forty historical windows correctly from text with dates stripped. Masking here is therefore
structural rather than cosmetic, and -- following the placebo-redaction precedent in
`llm.exposure` -- it returns a SURVIVOR COUNT rather than a promise. A mask whose residue
was not counted is an untested claim that the model saw nothing.

What masking cannot do, stated here so nobody reads the survivor count as a leak rate:
removing every date, ticker and issuer name does not remove the SUBJECT MATTER. A filing
about a pandemic-related facility closure dates itself to anyone who has read a newspaper.
That residual is what the period-recovery attack measures, and it is why the attack exists
alongside the mask instead of the mask being trusted on its own.
"""

from __future__ import annotations

import re

MONTHS = ("january february march april may june july august september october november "
          "december jan feb mar apr jun jul aug sep sept oct nov dec").split()

#: Patterns removed outright. Ordered longest-first where they overlap so a four-digit year
#: inside a full date is not left behind as a fragment.
PATTERNS: tuple[tuple[str, str], ...] = (
    ("date_long", r"\b(?:" + "|".join(MONTHS) + r")\.?\s+\d{1,2},?\s+\d{4}\b"),
    ("date_numeric", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ("date_iso", r"\b\d{4}-\d{2}-\d{2}\b"),
    ("month_year", r"\b(?:" + "|".join(MONTHS) + r")\.?\s+\d{4}\b"),
    ("year", r"\b(?:19|20)\d{2}\b"),
    ("quarter", r"\b(?:first|second|third|fourth|1st|2nd|3rd|4th|Q[1-4])\s+quarter\b"),
    ("fiscal", r"\bfiscal\s+(?:year\s+)?(?:19|20)?\d{2}\b"),
    ("month_name", r"\b(?:" + "|".join(MONTHS) + r")\b"),
)

REDACTED = "[REDACTED]"

#: Exhibit filenames, and they were the single largest leak channel MEASURED rather than
#: guessed. EDGAR exhibit names embed the date in compressed form -- ex991taxseasonupdatepr3518
#: is a press release of 3/5/18, financialsx2018 names its year, ex991lb-20174qearningsrele
#: gives quarter and year. None of the ordinary date patterns match, because the digits sit
#: inside a long alphanumeric token with no word boundary around them. This is metadata, not
#: substance, so removing it costs the comparison nothing.
FILENAME_RE = re.compile(r"\b\S*(?:\.htm|\.html|\.txt|\.pdf)\b|\b(?=\S*\d)(?=\S*[A-Za-z])"
                         r"[A-Za-z0-9_\-]{12,}\b")

#: Compressed dates inside otherwise ordinary tokens: 3518, 20174q, 2018q1, fy18.
COMPACT_DATE_RE = re.compile(
    r"\b(?:fy|cy)\s?\d{2,4}\b|\b\d{4}q[1-4]\b|\bq[1-4]\s?\d{2,4}\b|\b\d{6,8}\b",
    re.IGNORECASE)

#: Corporate-name suffixes. Redacting a capitalised run that ENDS in one of these catches
#: counterparties, subsidiaries and merger partners without touching ordinary prose. The
#: first version of this module masked only the FILER's own ticker, and the period-recovery
#: attack read the period straight off the names that survived -- "Wynn Resorts Finance,
#: LLC", "Overstock/tZERO", "Meta Financial/Crestmark". Other people's names date a document
#: as precisely as your own.
#: "Company" and "Bank" are deliberately ABSENT. Both are ordinary filing vocabulary -- "the
#: Company" is how an issuer refers to itself in every 8-K ever written -- and including them
#: redacted plain prose. A distinctive name inside such a phrase is caught by the name-token
#: pass instead, which is the pass that should be doing that work.
SUFFIXES = ("Inc", "Inc.", "LLC", "L.L.C.", "Corp", "Corp.", "Corporation",
            "Ltd", "Ltd.", "Limited", "PLC", "plc", "N.A.", "L.P.", "LP",
            "Holdings", "Holding", "Bancorp", "Capital", "Partners", "Group", "Trust",
            "Technologies", "Systems", "Industries", "Enterprises")

#: A run of capitalised words ending in a suffix, e.g. "Prime Security Services Borrower, LLC".
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][\w&.\-]*[ ,]+){1,6}(?:" + "|".join(re.escape(s) for s in SUFFIXES) + r")\b")

#: Internal-capital tokens: tZERO, eBay, iRobot. Not caught by a suffix rule and highly
#: identifying.
CAMEL_RE = re.compile(r"\b[a-z]{1,3}[A-Z][A-Za-z]{2,}\b")

#: Person names beside a role or appointment verb. An officer appointment dates a filing very
#: precisely, and the attack used exactly this: it named two executives and reasoned from when
#: one of them joined. Anchored on role words rather than applied to every capitalised pair,
#: because an unanchored person-name rule eats ordinary prose.
ROLE_RE = re.compile(
    r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|appointed|named|elected|succeed(?:s|ed)?|resignation of|"
    r"departure of|promoted)\s+((?:[A-Z][a-z]+\.?\s+){1,3}[A-Z][a-z]+)")
NAME_ROLE_RE = re.compile(
    r"\b((?:[A-Z][a-z]+\.?\s+){1,3}[A-Z][a-z]+)(?=,?\s+(?:as\s+)?"
    r"(?:the\s+)?(?:Chief|President|Vice President|Executive|Senior|Principal|Controller|"
    r"Treasurer|Secretary|Chairman|Director|Officer|CEO|CFO|COO|CAO))")

#: Common words that appear inside registered issuer names and must NOT be redacted globally.
#: MEASURED, not guessed. The first version of this list held only corporate-form words, and
#: the shipped batch came back with quotes reading "common [REDACTED]" and "First Lien
#: [REDACTED] Agreement" -- because "Stock" and "Credit" occur inside registered issuer names
#: and were therefore redacted out of ordinary prose. Over-redaction is not a safe direction:
#: it silently removes the substance a comparison exists to convey, and it does so in the
#: quotes a reader is meant to check. It biases a leak rate DOWNWARD, which is the one mercy.
_GENERIC = {"the", "and", "of", "for", "new", "first", "national", "american", "united",
            "general", "global", "international", "financial", "capital", "group", "trust",
            "holdings", "company", "corporation", "systems", "technologies", "industries",
            "energy", "health", "bank", "bancorp", "resources", "partners", "services",
            # ordinary financial vocabulary that also appears inside issuer names
            "stock", "credit", "common", "preferred", "shares", "share", "equity", "asset",
            "assets", "income", "growth", "value", "select", "quality", "premium", "core",
            "advance", "advanced", "alliance", "summit", "pioneer", "heritage", "liberty",
            "security", "insurance", "investment", "investors", "properties", "property",
            "realty", "media", "digital", "network", "solutions", "products", "materials",
            "brands", "foods", "motors", "airlines", "communications", "electric", "water",
            "power", "gas", "petroleum", "mining", "steel", "paper", "chemical", "pharma",
            "pharmaceutical", "biosciences", "sciences", "laboratories", "medical", "care"}


def name_tokens(names: list[str], *, min_len: int = 5) -> list[str]:
    """Distinctive tokens from registered issuer names, for partial-form matching.

    Filings say "Zimmer Biomet" where the registry says "ZIMMER BIOMET HOLDINGS, INC.", so
    exact-string redaction of the registered name misses the form actually used. Splitting
    into tokens and dropping the generic ones catches the short form without redacting the
    word "financial" out of every sentence that contains it.
    """
    out: set[str] = set()
    for n in names:
        for tok in re.split(r"[^A-Za-z]+", n):
            if len(tok) >= min_len and tok.lower() not in _GENERIC:
                out.add(tok)
    return sorted(out, key=len, reverse=True)


def mask(text: str, *, entities: list[str] | None = None,
         tokens: list[str] | None = None) -> tuple[str, dict]:
    """Strip dates, periods and named entities; return the text and what survived.

    `entities` is the caller's list of issuer names and tickers for the documents in hand.
    It is passed in rather than inferred, because the index already knows exactly which
    issuer filed each document and guessing at names in prose would both miss and overreach.
    """
    out = text
    removed: dict[str, int] = {}
    # Filenames first: they are long tokens that later patterns would only partly chew.
    out, n_fn = FILENAME_RE.subn(REDACTED, out)
    out, n_cd = COMPACT_DATE_RE.subn(REDACTED, out)
    removed["filename"] = n_fn
    removed["compact_date"] = n_cd
    for name, pat in PATTERNS:
        out, n = re.subn(pat, REDACTED, out, flags=re.IGNORECASE)
        removed[name] = n

    tok_n = 0
    for t in (tokens or []):
        out, n = re.subn(rf"\b{re.escape(t)}\b", REDACTED, out, flags=re.IGNORECASE)
        tok_n += n
    removed["entity_token"] = tok_n

    ent = 0
    for e in sorted(entities or [], key=len, reverse=True):
        e = e.strip()
        if len(e) < 3:
            continue
        out, n = re.subn(rf"\b{re.escape(e)}\b", REDACTED, out, flags=re.IGNORECASE)
        ent += n
    removed["entity_known"] = ent

    # Entities the caller did not name: counterparties, subsidiaries, acquirers.
    out, n_suf = ENTITY_RE.subn(REDACTED, out)
    out, n_cam = CAMEL_RE.subn(REDACTED, out)
    removed["entity_suffix"] = n_suf
    removed["entity_camel"] = n_cam

    # People, anchored on roles and appointment verbs.
    out, n_r1 = ROLE_RE.subn(lambda m: m.group(0).replace(m.group(1), REDACTED), out)
    out, n_r2 = NAME_ROLE_RE.subn(REDACTED, out)
    removed["person"] = n_r1 + n_r2

    # The survivor count. Anything here is residue the mask did not catch, and it is reported
    # with every batch rather than assumed to be zero.
    survivors = {
        "year": len(re.findall(r"\b(?:19|20)\d{2}\b", out)),
        "month_name": len(re.findall(r"\b(?:" + "|".join(MONTHS) + r")\b", out, re.I)),
        "date_numeric": len(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", out)),
        # Entity residue is counted too, because the attack read the period off names the
        # first version of this mask never looked at.
        "corporate_entity": len(ENTITY_RE.findall(out)) + len(CAMEL_RE.findall(out)),
        "filename": len(FILENAME_RE.findall(out)),
        "compact_date": len(COMPACT_DATE_RE.findall(out)),
    }
    return out, {"removed": removed, "survivors": survivors,
                 "survivor_total": sum(survivors.values())}
