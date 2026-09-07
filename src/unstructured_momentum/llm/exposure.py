"""Per-name interest-rate exposure, extracted from what a company states in its own filing.

This is the atom the project's architecture rests on: a model reading ONE document and making a
call about it. That is the shape this project has repeatedly measured the model to be good at --
is this article about this firm, is this text templated, do these two filings describe related
businesses -- and it is the opposite of the shape it has repeatedly measured the model to be bad
at, which is producing an aggregate number. Aggregation here is arithmetic and belongs to the
statistical layer.

Rebuilt from its outputs
------------------------
The original module was never committed. What survives is 293 cached extractions across five
strata, and they are used here as a REGRESSION TARGET rather than as documentation: a rebuild
that does not reproduce them has not recovered the instrument, whatever else it does. Both
placebo attempts survive too, which makes the target two-sided -- the sentence-boundary redaction
that leaked 11 of 20, and the word-window redaction that gave 20 of 20.

What the extraction is NOT
--------------------------
It measures DISCLOSED FUNDING EXPOSURE: what the company says about its own borrowing costs and
instrument fair values. That is a different object from equity duration exposure, and the
project already refuted the claim that one tracks the other. The refutation was a contemporaneous
comparison against a returns-derived quantity, which is the benchmark this project has since
ruled out for text; the atom itself passed a verified placebo and a per-stratum positive control.
So the atom is sound and its previous assignment was wrong.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

CACHE = Path("data/interim/exposure_v2")
LEGACY = Path("data/interim/exposure_cache.json")
DEFAULT_MODEL = "claude-opus-4-8"

#: Two term lists, because the two uses pull in opposite directions. REDACT_TERMS is deliberately
#: broad: a placebo's job is to remove the topic entirely, and a survivor is a failed placebo. It
#: includes bare "interest", which caught the last leak -- a filing quoted "market value changes
#: caused by interest fluctuations" after every "interest rate" had gone.
#: RATE_TERMS is narrower and gates EVIDENCE: a quote must state a rate exposure, and "interest"
#: alone in a sentence about interest expense is a weaker claim than a benchmark or a rate move.
REDACT_TERMS = (
    "interest", "rate", "rates", "LIBOR", "SOFR", "federal funds", "basis point",
    "basis points", "yield curve", "swap", "swaps", "prime", "benchmark", "floating",
    "variable-rate", "fixed-rate", "hedge", "hedges", "hedging",
)

RATE_TERMS = (
    "interest rate", "interest rates", "LIBOR", "SOFR", "federal funds", "basis point",
    "basis points", "variable rate", "variable-rate", "floating rate", "floating-rate",
    "fixed rate", "fixed-rate", "yield curve", "swap", "swaps", "discount rate",
    "prime rate", "benchmark rate", "rate risk",
)

Exposure = Literal["none", "hurt_by_rising", "benefits_from_rising", "hedged_neutral"]


class ExposureRecord(BaseModel):
    """Matches the 293 cached records field for field, so regression is a direct comparison."""

    exposure: Exposure = Field(
        description="Direction of the company's stated sensitivity to RISING interest rates. "
                    "'none' if the filing states no material exposure.")
    magnitude: int = Field(
        ge=0, le=8,
        description="0 to 8. 0 for none. Scale on the stated dollar or percentage impact "
                    "relative to the company's size, not on how many words are spent on it.")
    channel: str = Field(
        max_length=120,
        description="The mechanism in under twelve words, in the filing's own terms, e.g. "
                    "'variable-rate debt raises interest expense'. 'none' if no exposure.")
    evidence: str = Field(
        max_length=400,
        description="A sentence QUOTED VERBATIM from the filing that supports the call. Must "
                    "appear in the source text. 'none' if no exposure.")


SYSTEM = """\
You read one company's market-risk disclosure and report what IT says about its own sensitivity
to interest rates. You are not estimating the company's stock sensitivity to rates, and you are
not using anything you know about the company from elsewhere. Only the text given.

Rules:
- Report the direction of exposure to RISING rates, from the company's own statements.
- If the filing states no material interest-rate exposure, say so. That is a common and correct
  answer, not a failure to find something.
- Quote your evidence VERBATIM from the text. Do not paraphrase, do not reconstruct, do not
  combine fragments from different sentences. If you cannot quote a supporting sentence, the
  answer is 'none'.
- The quoted sentence must itself mention interest rates, or a rate benchmark, or rate-linked
  instruments. A sentence that mentions only debt, borrowings or a credit facility does NOT
  establish a stated rate exposure. Do not infer from the existence of debt that the company
  has told you about its rate sensitivity -- it may simply not have said.
- Magnitude scales the stated impact against the company's size. A disclosed 100 basis point
  sensitivity of $400 million is large; one of $0.9 million is small. Do not scale by how much
  prose the filing devotes to the topic.
"""

USER = """\
{ticker} -- market-risk disclosure.

{text}

Report the stated interest-rate exposure."""


# ------------------------------------------------------------------ placebo redaction

def redact_rate_language(text: str, *, window: int = 12) -> tuple[str, int]:
    """Remove rate language by WORD WINDOW, and report how many terms survive.

    The first version of this redacted whole sentences on full-stop boundaries and leaked 11 of
    20 placebo filings, because filings are dense with decimals and abbreviations and a
    sentence splitter cuts in the wrong places. Removing a window of words around each hit does
    not depend on punctuation at all.

    The surviving-term count is returned rather than assumed. A placebo whose redaction was not
    verified is not a placebo; it is an untested claim that the model saw nothing.
    """
    words = text.split()
    lowered = [w.lower().strip(".,;:()[]$%") for w in words]
    joined = " ".join(lowered)
    if not any(t in joined for t in REDACT_TERMS):
        return text, 0
    drop = set()
    singles = {t for t in REDACT_TERMS if " " not in t}
    multis = [t.split() for t in REDACT_TERMS if " " in t]
    for i, w in enumerate(lowered):
        hit = w in singles or any(
            lowered[i:i + len(m)] == m for m in multis if i + len(m) <= len(lowered))
        if hit:
            drop.update(range(max(0, i - window), min(len(words), i + window + 1)))
    kept = [w for i, w in enumerate(words) if i not in drop]
    out = " ".join(kept)
    survivors = sum(len(re.findall(rf"\b{re.escape(t)}\b", out.lower())) for t in REDACT_TERMS)
    return out, survivors


# ------------------------------------------------------------------ the quote gate

def quote_is_grounded(evidence: str, source: str, *, threshold: float = 0.85) -> bool:
    """Is the quoted evidence actually in the document?

    Machine-checkable for free, and it is the failure mode the field exists to prevent. Exact
    matching is too brittle -- filings carry non-breaking spaces, soft hyphens and inconsistent
    whitespace -- so this normalises and then allows a fuzzy match over a sliding window.
    """
    if not evidence or evidence.strip().lower() == "none":
        return True
    norm = lambda s: re.sub(r"\s+", " ", s.lower().replace(" ", " ")).strip()  # noqa: E731
    e, src = norm(evidence), norm(source)
    if e in src:
        return True
    # Ellipsis is allowed in the cached records; check each fragment separately.
    frags = [f.strip() for f in e.split("...") if len(f.strip()) > 24]
    if frags and all(f in src for f in frags):
        return True
    n = len(e)
    for i in range(0, max(1, len(src) - n), max(1, n // 2)):
        if difflib.SequenceMatcher(None, e, src[i:i + n]).ratio() >= threshold:
            return True
    return False


class UngroundedQuote(ValueError):
    """Raised when extracted evidence does not appear in the document it claims to quote."""


def evidence_states_rate_exposure(evidence: str) -> bool:
    """Does the quoted sentence actually mention rates, or only debt?

    The placebo caught this and it is the atom's real failure mode. Handed a filing with every
    rate term removed, the model still reported exposure for eleven of twenty companies, quoting
    sentences about credit facilities and borrowings and INFERRING that rates must matter. The
    inference is usually correct about the world and is not what this instrument measures: the
    claim is that the company STATED an exposure, and a quote with no rate language does not
    support it.

    Mechanical, like the grounding check, and for the same reason -- a discipline the prompt
    asks for is a discipline the model may skip, while one the code enforces it cannot.
    """
    if not evidence:
        return False
    low = evidence.lower()
    return any(t in low for t in RATE_TERMS)


# ------------------------------------------------------------------ extraction

@dataclass
class ExposureExtractor:
    model: str = DEFAULT_MODEL
    max_tokens: int = 700
    enforce_quote: bool = True
    #: An exposure claim whose quote contains no rate language is downgraded to 'none'. The
    #: placebo exists to catch exactly this and did: the model inferred rate sensitivity from
    #: sentences about borrowings after every rate term had been removed.
    require_rate_language: bool = True

    def _key(self, ticker: str, text: str, stratum: str) -> str:
        return hashlib.sha256(
            f"{ticker}|{stratum}|{self.model}|{text[:4000]}".encode()).hexdigest()[:20]

    #: The section this instrument was validated on. Its prompt says "market-risk
    #: disclosure"; its 96.6% precision, its placebo and its per-stratum positive control
    #: were all established on Item 7A. A run that fed it Item 1A risk factors instead
    #: produced a 10% schema-rejection rate and 72% apparent disagreement with the legacy
    #: records -- both artifacts of the mismatch, neither a property of the model. The
    #: caller now has to name the section and is refused if it is not the validated one.
    #: Same species as the certificate-universe mismatch: an instrument pointed at a
    #: population it was not certified on, where everything downstream measures the
    #: mismatch rather than the phenomenon.
    section: str = "item_7a"

    def _assert_section(self, section: str | None) -> None:
        if section is None:
            raise ValueError(
                "extract_one requires an explicit `section=`. This extractor is validated "
                f"on {self.section!r} only; passing text without naming its section is how "
                "the Item 1A mismatch happened.")
        if section != self.section:
            raise ValueError(
                f"section {section!r} is not the validated input for this extractor "
                f"({self.section!r}). Re-validate before repointing it, or construct "
                f"ExposureExtractor(section={section!r}) deliberately and record that the "
                "precision, placebo and positive-control figures no longer apply.")

    def extract_one(self, ticker: str, text: str, *, stratum: str = "real",
                    section: str | None = None,
                    use_cache: bool = True) -> ExposureRecord:
        self._assert_section(section)
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"{self._key(ticker, text, stratum)}.json"
        if use_cache and f.exists():
            return ExposureRecord(**json.loads(f.read_text()))
        import anthropic
        resp = anthropic.Anthropic().messages.parse(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": USER.format(ticker=ticker, text=text[:60000])}],
            output_format=ExposureRecord)
        rec = resp.parsed_output
        if self.enforce_quote and rec.exposure != "none":
            if not quote_is_grounded(rec.evidence, text):
                raise UngroundedQuote(
                    f"{ticker}: evidence not found in the source filing -- {rec.evidence[:120]!r}")
            if self.require_rate_language and not evidence_states_rate_exposure(rec.evidence):
                # Downgraded rather than raised: an unsupported claim is a 'none', not an error.
                rec = ExposureRecord(exposure="none", magnitude=0, channel="none",
                                     evidence="none")
        f.write_text(rec.model_dump_json(indent=2))
        return rec


# ------------------------------------------------------------------ regression against the 293

def legacy_records(stratum: str) -> dict[str, dict]:
    if not LEGACY.exists():
        return {}
    d = json.loads(LEGACY.read_text())
    return {k.split("|", 1)[1]: v for k, v in d.items() if k.split("|")[0] == stratum}


def agreement(new: dict[str, ExposureRecord], stratum: str) -> dict:
    """Direction agreement against the surviving cache. The gate is 80 percent."""
    old = legacy_records(stratum)
    both = [t for t in new if t in old]
    if not both:
        return {"n": 0}
    same = sum(1 for t in both if new[t].exposure == old[t]["exposure"])
    return {"n": len(both), "agree": same, "rate": same / len(both),
            "passes_gate": same / len(both) >= 0.80}
