"""Translation: from a measured tilt to something a portfolio manager can act on and score.

The statistical channels produce sentences like "the book just inherited a 1.46 beta spread
with a defensive tilt at the 20th percentile of its history". That is a measurement, not an
instruction, and nothing numeric turns it into one. Naming the driver, the transmission
channel, the instrument that hedges it in macro space and the scheduled events that would move
it is the one job in this system that a model does and a regression cannot.

The falsifier is what keeps that from being decoration
------------------------------------------------------
Every generated brief carries a condition that would show it wrong, and the condition is
STRUCTURED rather than prose: a series in a fixed registry, a direction, a threshold in the
series' own units, and a horizon in trading days. Prose falsifiers cannot be scored, and a
falsifier that cannot be scored is a rhetorical gesture. Because the registry is fixed and the
threshold is numeric, scoring is a lookup — no model is involved in marking the model's work.

The condition is logged AT GENERATION TIME with the vintage of every input, so a later score
reads what was actually claimed rather than what it is convenient to have claimed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..data.eightk import MATERIAL_ITEMS

DEFAULT_MODEL = "claude-opus-4-8"
LOG = Path("data/processed/falsifiers.jsonl")

#: Series a falsifier may reference. Fixed, because a falsifier naming a series the system
#: cannot fetch is unscoreable, which is the failure this whole module exists to avoid.
REGISTRY: dict[str, tuple[str, str, str]] = {
    "DGS10": ("data/raw/fred/DGS10.csv", "DGS10", "10-year Treasury yield, percent"),
    "T10Y2Y": ("data/raw/fred/T10Y2Y.csv", "T10Y2Y", "10-year minus 2-year spread, percent"),
    "T10Y3M": ("data/raw/fred/T10Y3M.csv", "T10Y3M", "10-year minus 3-month spread, percent"),
    "BAA10Y": ("data/raw/fred/BAA10Y.csv", "BAA10Y", "Baa credit spread over 10-year, percent"),
    "VIXCLS": ("data/raw/fred/VIXCLS.csv", "VIXCLS", "VIX index level"),
    "NFCI": ("data/raw/fred/NFCI_NFCICREDIT.csv", "NFCI", "Chicago Fed financial conditions"),
}


def series(name: str) -> pd.Series:
    path, col, _ = REGISTRY[name]
    d = pd.read_csv(path)
    idx = pd.to_datetime(d[d.columns[0]])
    return pd.to_numeric(d[col], errors="coerce").set_axis(idx).dropna()


# ------------------------------------------------------------------ output contract

class Falsifier(BaseModel):
    variable: str = Field(description=f"One of: {', '.join(REGISTRY)}")
    direction: str = Field(description="'rise' or 'fall' -- the move that would refute the read")
    threshold: float = Field(description="Magnitude of the move, in the series' own units")
    horizon_days: int = Field(description="Trading days from the as-of date, 5 to 63")
    refutes: str = Field(description="One sentence: what this move would show was wrong")


#: Human-readable meanings for the closed 8-K item vocabulary, used only to render prompts.
#: The vocabulary itself is ``eightk.MATERIAL_ITEMS`` -- an event falsifier naming an item the
#: corpus does not carry would be unscoreable, the same failure the series REGISTRY exists to
#: avoid.
ITEM_MEANINGS: dict[str, str] = {
    "1.01": "entry into a material definitive agreement",
    "1.02": "termination of a material definitive agreement",
    "1.03": "bankruptcy or receivership",
    "2.03": "creation of a direct financial obligation",
    "2.04": "triggering events that accelerate a financial obligation",
    "2.05": "costs associated with exit or disposal activities",
    "2.06": "material impairments",
    "4.02": "non-reliance on previously issued financial statements",
    "5.02": "departure of directors or principal officers",
    "7.01": "Regulation FD disclosure -- where guidance moves",
    "8.01": "other events",
}


class EventFalsifier(BaseModel):
    """An issuer-event falsifier: the condition under which the COMPANIES carrying the bet
    would themselves say the driver is wrong, in their 8-Ks.

    These fields are the model's CLAIM, declared once at generation time. The two integers
    (min_distinct_filers, horizon_days) parameterise the claim exactly as a threshold
    parameterises a macro falsifier -- they are never measurements, and nothing the model
    emits is ever counted or scored by a model. All watching, matching, counting and the
    trip decision are deterministic (see text/tripwire.py).
    """

    item_codes: list[str] = Field(description=(
        "8-K item codes the refuting evidence would arrive under. Closed vocabulary: "
        + ", ".join(MATERIAL_ITEMS)))
    anchor_terms: list[str] = Field(description=(
        "3 to 8 lowercase phrases. A filing is only read if one of them appears verbatim "
        "(case-insensitive substring) in its body, so choose phrases the refuting filing "
        "would actually contain, not abstractions."))
    min_distinct_filers: int = Field(description=(
        "How many DISTINCT theme members must file quote-gated refuting evidence inside the "
        "horizon before the tripwire trips. One filer is an idiosyncratic story; the number "
        "here is what makes it a claim about the shared bet."))
    horizon_days: int = Field(description="Trading days from the as-of date, 5 to 63")
    refutes: str = Field(description="One sentence: what such filings would show was wrong")


class Translation(BaseModel):
    driver: str = Field(description="The macro driver the tilt is a bet on, in five words or fewer")
    transmission: str = Field(description="One sentence on how the driver reaches these holdings")
    hedge_instrument: str = Field(description="A liquid macro instrument that offsets the tilt")
    hedge_rationale: str = Field(description="One sentence on why that instrument offsets it")
    catalysts: list[str] = Field(description="Scheduled events in the horizon that move the driver")
    falsifier: Falsifier
    event_falsifier: EventFalsifier | None = Field(
        default=None,
        description=("Issuer-event falsifier for the same driver, or null to DECLINE. "
                     "Declining is a correct answer when no 8-K item family could carry "
                     "evidence for or against the driver."))
    event_decline_reason: str = Field(
        default="",
        description=("Required when event_falsifier is null: one sentence on why no 8-K "
                     "item family could carry evidence on this driver."))
    confidence: int = Field(description="1-10 that the driver is correctly identified")


SYSTEM = """\
You are translating a measured factor tilt into something a portfolio manager can act on.

You are NOT being asked to find the exposure. It has already been measured statistically, and
the measurement is given to you. Your job is to say what it means operationally and to state
what would show you wrong.

The tilt was CONFERRED BY THE SORT, not chosen by anyone. A momentum ranking applied to past
returns hands the book a factor loading as a by-product of selection. So the driver you name
should be a macro variable the loading is exposed to, not a story about why these particular
companies are good or bad.

Rules:
- The falsifier must name a variable from the given registry, a direction, a numeric threshold
  in that variable's own units, and a horizon between 5 and 63 trading days. It will be scored
  mechanically against data. Choose a threshold that is genuinely discriminating: one the
  variable clears perhaps one time in four over that horizon, not one it clears every week or
  one it never clears.
- Catalysts must be scheduled events with dates in the horizon, not vague conditions.
- If the measured tilt is weak or mixed, say so in the driver and set confidence low. A tilt
  near zero has no operational content and pretending otherwise is the failure mode here.
- The event falsifier is OPTIONAL and refers to what the companies carrying the bet would
  file in their 8-Ks if the driver were wrong. Its item codes must come from the given 8-K
  item vocabulary. Its anchor terms are a recall net -- short topic vocabulary any filing
  on the subject would contain, since only anchor-matched filings get read at all; the
  refutation judgment is made by a reader afterwards, never by the anchors.
  Some drivers have no issuer-event signature -- a rates/duration bet is refuted by market
  series, not by anything a company files. If no 8-K item family could carry evidence for or
  against this driver, return DECLINE for the event falsifier (null, with a reason) and keep
  the macro falsifier only. A declined event falsifier is a correct answer.
"""

USER = """\
As of {asof}.

MEASURED TILT, estimated on the formation window that closes at this date:
{tilt}

NAMED COMPOSITION, from a partition of the holdings that beats a sector- and size-matched
control (this is what the book is made of, not what it is exposed to):
{composition}

CHANNEL STATE:
{channels}

FALSIFIER REGISTRY -- you must choose one:
{registry}

8-K ITEM VOCABULARY for the optional event falsifier (or DECLINE):
{items}

Translate.
"""


@dataclass
class Translator:
    model: str = DEFAULT_MODEL

    def generate(self, asof: str, tilt: dict, composition: list[str],
                 channels: dict, *, use_cache: bool = True) -> Translation:
        key = hashlib.sha256(
            json.dumps([asof, tilt, composition, channels, self.model],
                       sort_keys=True, default=str).encode()).hexdigest()[:20]
        cache = Path("data/interim/translations")
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / f"{key}.json"
        if use_cache and f.exists():
            return Translation(**json.loads(f.read_text()))
        import anthropic
        reg = "\n".join(f"  {k}: {v[2]}" for k, v in REGISTRY.items())
        body = USER.format(
            asof=asof,
            tilt="\n".join(f"  {k}: {v}" for k, v in tilt.items()),
            composition="\n".join(f"  - {c}" for c in composition) or "  (not available)",
            channels="\n".join(f"  {k}: {v}" for k, v in channels.items()) or "  (not available)",
            registry=reg,
            items="\n".join(f"  {k}: {v}" for k, v in ITEM_MEANINGS.items()))
        resp = anthropic.Anthropic().messages.parse(
            model=self.model, max_tokens=2000, system=SYSTEM,
            messages=[{"role": "user", "content": body}],
            output_format=Translation)
        out = resp.parsed_output
        f.write_text(out.model_dump_json(indent=2))
        return out


def content_key(rec: dict) -> str:
    """Idempotency key over the CLAIM, not the run. Nightly re-runs regenerate the same
    translation and used to re-append it verbatim; the daily X-ray then had to dedupe on
    read. The claim is identified by what would be scored -- as-of, driver, series,
    direction, threshold, horizon -- so re-logging the same claim is a no-op while a changed
    threshold or horizon is a NEW claim and appends."""
    fields = [str(rec.get("as_of")), str(rec.get("driver")), str(rec.get("variable")),
              str(rec.get("direction")), f"{float(rec.get('threshold', 0.0)):.10g}",
              str(int(rec.get("horizon_days", 0)))]
    return hashlib.sha256("|".join(fields).encode()).hexdigest()[:16]


def log_falsifier(asof: str, t: Translation, vintages: dict) -> dict:
    """Write the condition at generation time, with input vintages, so scoring reads the claim.

    Append-once: the write is keyed on the claim's content (see ``content_key``), and keys
    are recomputed from row fields for rows logged before keys existed, so historical
    duplicates also stop re-appending."""
    rec = {"as_of": asof, "driver": t.driver, "confidence": t.confidence,
           "variable": t.falsifier.variable, "direction": t.falsifier.direction,
           "threshold": float(t.falsifier.threshold),
           "horizon_days": int(t.falsifier.horizon_days),
           "refutes": t.falsifier.refutes, "vintages": vintages}
    if t.event_falsifier is not None:
        ef = t.event_falsifier
        rec["event_falsifier"] = {
            "item_codes": list(ef.item_codes), "anchor_terms": list(ef.anchor_terms),
            "min_distinct_filers": int(ef.min_distinct_filers),
            "horizon_days": int(ef.horizon_days), "refutes": ef.refutes}
    elif t.event_decline_reason:
        rec["event_falsifier"] = None
        rec["event_decline_reason"] = t.event_decline_reason
    rec["key"] = content_key(rec)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        have = {json.loads(l).get("key") or content_key(json.loads(l))
                for l in LOG.open() if l.strip()}
        if rec["key"] in have:
            return rec
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


#: Comparison tolerance for the threshold test. FRED quotes these series to two decimals, so
#: a realised move is only ever meaningful to that precision -- but 1.9 - 1.5 evaluates to
#: 0.3999999999999999 in binary floating point, which is BELOW a 0.4 threshold by 1.1e-16.
#: The first real falsifier this module generated landed exactly on its threshold and was
#: scored as standing when it had in fact been refuted. A track record that flips on binary
#: representation is worse than no track record, so the comparison is made at a precision the
#: data actually carries.
THRESHOLD_EPS = 1e-9


def score(rec: dict, asof: pd.Timestamp | None = None) -> dict:
    """Mechanically check one logged falsifier. No model involved in marking the model."""
    name = rec["variable"]
    if name not in REGISTRY:
        return {**rec, "status": "unscoreable", "reason": f"{name} not in registry"}
    s = series(name)
    a = pd.Timestamp(asof or rec["as_of"])
    fwd = s[s.index > a]
    if fwd.empty:
        return {**rec, "status": "pending", "reason": "no data after the as-of date"}
    window = fwd.iloc[: int(rec["horizon_days"])]
    if len(window) < int(rec["horizon_days"]):
        return {**rec, "status": "pending",
                "reason": f"horizon incomplete: {len(window)} of {rec['horizon_days']} days"}
    base = float(s[s.index <= a].iloc[-1])
    move = float(window.max() - base) if rec["direction"] == "rise" else float(base - window.min())
    hit = move >= float(rec["threshold"]) - THRESHOLD_EPS
    return {**rec, "status": "refuted" if hit else "stood",
            "base": base, "realised_move": move, "extreme": float(
                window.max() if rec["direction"] == "rise" else window.min())}
