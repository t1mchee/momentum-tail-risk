"""Typed evidence records — the contract between the language layer and everything else.

The central design rule of this system: **the LLM never produces a number that enters a
model.** It reads text, decides whether that text is genuinely about the mechanism, and
emits a structured record with a citation. Aggregation into features, and every
probability or severity, is computed by deterministic code.

That boundary is enforced here rather than by prompt discipline. `EvidenceRecord` has no
field for a probability, a risk score, or a forecast, so an extractor physically cannot
return one. If a future version needs the model to quantify something, that has to be an
explicit, reviewed schema change — not a sentence that drifted into a prompt.

Why a schema at all, rather than free text: the measured failure of keyword counting
(82% of pre-event "crowding" matches were one content farm's templated filler) means the
useful work is *relevance judgment*, and a judgment is only auditable if it comes with the
claim, the mechanism it bears on, the direction, and the source it came from.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Mechanism(str, Enum):
    """Which channel of the system a piece of evidence bears on.

    Deliberately mirrors the mechanisms found in the return data rather than a generic
    sentiment taxonomy: these are the things that actually move the momentum factor.
    """

    CROWDING = "crowding"                 # positioning concentration in the trade itself
    DELEVERAGING = "deleveraging"         # forced degrossing across the factor complex
    ROTATION = "rotation"                 # leadership change between factors/styles
    SHORT_SQUEEZE = "short_squeeze"       # constraint on the short leg
    CATALYST = "catalyst"                 # a scheduled or pending binary event
    NONE = "none"                         # not about the mechanism at all


class Direction(str, Enum):
    """Which way the evidence points for momentum reversal risk."""

    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"
    NEUTRAL = "neutral"


class Leg(str, Enum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"
    NOT_APPLICABLE = "not_applicable"


class Timing(str, Enum):
    """Whether the piece describes something that has happened or something pending.

    This is the field the Sept 2019 finding demands. Coverage of that unwind was
    overwhelmingly *retrospective* -- published after the move, describing it. A system
    that counts those articles as risk signal is reading the newspaper's account of
    yesterday and calling it a forecast. Separating retrospective from prospective is what
    makes the distinction measurable rather than assumed.
    """

    RETROSPECTIVE = "retrospective"   # describes a move that already happened
    CONTEMPORANEOUS = "contemporaneous"  # describes conditions now
    PROSPECTIVE = "prospective"       # warns about or anticipates something ahead


class EvidenceRecord(BaseModel):
    """One judged piece of text evidence.

    Note the absence of any probability, score, or forecast field. That is deliberate and
    load-bearing -- see the module docstring.
    """

    is_relevant: bool = Field(
        description=(
            "True only if this text substantively discusses equity factor positioning, "
            "crowding, deleveraging, or a leadership rotation. False for templated or "
            "auto-generated content, generic market wrap-ups, single-company news with no "
            "factor angle, and pieces that merely use a keyword in passing."
        )
    )
    mechanism: Mechanism = Field(description="Which mechanism this bears on; 'none' if not relevant.")
    direction: Direction = Field(
        description="Whether this points to higher or lower momentum reversal risk."
    )
    leg: Leg = Field(description="Which leg of the momentum trade it concerns, if identifiable.")
    timing: Timing = Field(
        description=(
            "Whether the piece describes a move that already happened (retrospective), "
            "current conditions (contemporaneous), or something anticipated (prospective)."
        )
    )
    is_templated: bool = Field(
        description=(
            "True if this looks auto-generated or templated -- e.g. formulaic "
            "shareholder-composition pieces, screener output, or syndicated boilerplate."
        )
    )
    claim: str = Field(
        max_length=300,
        description="One sentence stating what this text actually asserts. Quote or closely paraphrase; do not editorialise.",
    )
    reasoning: str = Field(
        max_length=300,
        description="Brief justification for the relevance judgment.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the RELEVANCE JUDGMENT ONLY -- not a probability of any market "
            "outcome. This never enters a forecast."
        ),
    )


class ExtractionResult(BaseModel):
    """An evidence record joined back to its source document and timestamps."""

    record: EvidenceRecord
    url: str = ""
    domain: str = ""
    title: str = ""
    seendate: str = ""
    extractor: str = ""
    model: str = ""
