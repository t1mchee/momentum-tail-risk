"""The synthesis layer — a typed evidence bundle in, a typed cited reading out.

Why this is a separate module from ``llm/synthesis.py``
------------------------------------------------------
``llm/synthesis.py`` is a different instrument with the same word in its name: a
semantic-aggregation probe over a book's headlines, with a placebo and a hindsight
guess, and it backs three registered claims through its ``reproduce`` commands. Nothing
in it composes a PM-facing reading. It was read before this was written; what carried
over is its discipline (a mandatory decline, a placebo for the model's urge to agree)
rather than its code.

The structural bar on model-produced numbers
--------------------------------------------
``llm/schema.py`` states the project's central rule: the LLM never produces a number
that enters a model, and that rule is enforced by the shape of the schema rather than
by prompt discipline. A synthesis layer is where that rule is hardest to keep, because
the deliverable is *prose about numbers*. So the enforcement here is:

  * every prose field forbids digits, checked by a Pydantic validator that raises;
  * a number reaches the page only as a ``{dotted.bundle.path}`` placeholder, which
    deterministic code resolves against the bundle and substitutes;
  * a placeholder naming a path the bundle does not have is a violation, not a value.

The consequence is that the rendered reading's numerals are, by construction, bundle
values. The model chooses *which* number to cite and what to say about it; it cannot
choose what the number is. A model that wants to assert a figure it was not given has
no field to put it in.

The decline pathway is mandatory, for the reason the repo has hit repeatedly: a prompt
with no null option produces a confident wrong answer. ``Reading.declined`` and the
``could_not_measure`` list are the two forms of refusal — the whole reading, or one
channel of it.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

DEFAULT_MODEL = "claude-sonnet-5"

# --------------------------------------------------------------------------------------
# Channels — the ablation unit, and the vocabulary the reading must use to refuse
# --------------------------------------------------------------------------------------


class Channel(str, Enum):
    """One independently computed strand of evidence.

    These are the units the ablation cell deletes, so they are an enum rather than free
    text: a reading that refuses on a channel has to name it in a vocabulary a checker
    can compare, not in a sentence a checker has to interpret.
    """

    BET = "bet"                    # the named story the book is levered to, and its loading
    CROWDING = "crowding"          # 13F concentration, named holders, days-to-cover, flows
    SEVERITY = "severity"          # conditional VaR/ES term structure and exceedance
    DRIVERS = "drivers"            # Shapley decomposition of the severity fit, component ES
    CONFERRED = "conferred"        # what the sort handed the book: beta and duration tilt
    FRAGILITY = "fragility"        # the theme table and the rotation door
    ANALOGS = "analogs"            # nearest historical states
    FALSIFIERS = "falsifiers"      # armed macro falsifiers and event tripwires
    OTHER = "other"

    @classmethod
    def ablatable(cls) -> list["Channel"]:
        return [c for c in cls if c is not cls.OTHER]


class HeadlineDirection(str, Enum):
    """The direction of the headline assessment, as three ordered states.

    Three rather than a score, because the primary test asks whether the *direction*
    moves, and a score invites a cosmetic decimal shuffle that reads as movement.
    """

    ELEVATED = "elevated"    # this book is carrying more reversal risk than its own normal
    MIXED = "mixed"          # the channels disagree, or the evidence does not separate
    BENIGN = "benign"        # this book is carrying less reversal risk than its own normal


class ClaimDirection(str, Enum):
    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"
    NEUTRAL = "neutral"


# --------------------------------------------------------------------------------------
# The evidence bundle
# --------------------------------------------------------------------------------------


class Filer(BaseModel):
    rank: int
    filer: str
    cik: int
    share_of_theme_value: float
    value_usd: float


class BetChannel(BaseModel):
    """The story the book is actually levered to, and how hard it is levered to it."""

    story_name: str
    side: str                     # "W" winner leg, "L" loser leg
    n_members: int
    members: list[str]
    story_beta: float
    load_pct: float               # percentile of loading on the story, 0-1
    saturation: float
    saturation_pct: float
    validity_gated: bool


class CrowdingChannel(BaseModel):
    """Who owns the story, and how concentrated that ownership is."""

    top10_share_13f: float
    top10_share_pct: float        # percentile within the calm pool, 0-1
    quarter: str
    days_to_cover_pct: float
    top_filers: list[Filer]
    comomentum_spread: float
    comomentum_pct: float
    short_volume_share: float
    short_volume_change_21d: float
    mtum_shares_outstanding: float
    mtum_pct_change_21d: float
    disclosure: str


class TermRow(BaseModel):
    horizon_days: int
    quantile: float
    var_conditional: float
    es_conditional: float
    var_unconditional: float
    es_unconditional: float


class ExceedanceRow(BaseModel):
    threshold_level: float
    conditional_prob: float
    base_rate: float


class SeverityChannel(BaseModel):
    """The deterministic spine: how bad it gets, conditional on today."""

    declared_horizon_days: int
    declared_quantile: float
    var_conditional: float
    es_conditional: float
    var_unconditional: float
    es_unconditional: float
    var_pctile_of_history: float
    n_train: int
    term_structure: list[TermRow]
    exceedance: list[ExceedanceRow]
    realised_vol_126d_ann: float
    barroso_leverage: float
    wml_21d: float
    wml_5d: float
    disclosure: str


class ShapleyTerm(BaseModel):
    term: str
    shapley: float
    value_now: float


class ComponentEsName(BaseModel):
    ticker: str
    contribution: float


class DriversChannel(BaseModel):
    """What the severity estimate is made of, and which names carry the tail."""

    #: Severity-derived levels, nullable for the same reason as the fragility columns above.
    baseline: float | None = None
    fitted: float | None = None
    n_subsets: int
    reconstruction_error: float
    terms: list[ShapleyTerm]
    book_es: float | None = None
    n_tail_days: int
    n_names: int
    top_names: list[ComponentEsName]


class ConferredChannel(BaseModel):
    """What the momentum sort handed the book before anyone chose anything."""

    beta_spread: float
    beta_spread_pct: float
    beta_winner: float
    beta_loser: float
    d10y_spread: float
    n_leg: int
    placebo_base_rate: float
    posture: str


class FragilityRow(BaseModel):
    n: int
    side: str
    validity_gated: bool
    story_beta: float
    load_pct: float
    saturation: float
    saturation_pct: float
    #: Crowding-derived, and therefore nullable: deleting the crowding channel has to
    #: delete these too, or the ablation leaks the very evidence it claims to have
    #: removed. The first ablation run cited fragility.rows[i].top10_share_pct while
    #: declaring crowding unmeasured, which is a leak in the test, not a fault of the model.
    top10_share_pct: float | None = None
    days_to_cover_pct: float | None = None


class FragilityChannel(BaseModel):
    n_themes: int
    n_gated: int
    rotation_door_armed: bool
    rows: list[FragilityRow]


class AnalogRow(BaseModel):
    date: str
    distance: float


class AnalogChannel(BaseModel):
    rows: list[AnalogRow]
    disclosure: str


class MacroFalsifier(BaseModel):
    as_of: str
    driver: str
    variable: str
    direction: str
    threshold: float
    horizon_days: int
    refutes: str


class EventTripwire(BaseModel):
    as_of: str
    driver: str
    n_members: int
    min_distinct_filers: int
    horizon_days: int
    status: str
    refutes: str


class FalsifierChannel(BaseModel):
    n_macro_armed: int
    n_event_armed: int
    n_tripped: int
    macro: list[MacroFalsifier]
    event: list[EventTripwire]


class Contract(BaseModel):
    """Non-ablatable framing. Never deleted: the reading has to know what it is reading."""

    as_of: str
    book_as_of: str
    declared_horizon_days: int
    declared_quantile: float
    reversal_target: str
    intended_user: str
    decision: str
    n_winners: int
    n_losers: int


class EvidenceBundle(BaseModel):
    """Everything the synthesis layer is allowed to know.

    Each channel is optional so that deleting one is a representable state rather than a
    crash — which is the whole point of the ablation cell. A ``None`` channel means the
    evidence was not computed, and the reading's only correct response is to say so.
    """

    bundle_id: str
    contract: Contract
    bet: BetChannel | None = None
    crowding: CrowdingChannel | None = None
    severity: SeverityChannel | None = None
    drivers: DriversChannel | None = None
    conferred: ConferredChannel | None = None
    fragility: FragilityChannel | None = None
    analogs: AnalogChannel | None = None
    falsifiers: FalsifierChannel | None = None

    def present_channels(self) -> list[Channel]:
        return [c for c in Channel.ablatable() if getattr(self, c.value) is not None]

    def missing_channels(self) -> list[Channel]:
        return [c for c in Channel.ablatable() if getattr(self, c.value) is None]

    # -- field index -------------------------------------------------------------------

    def leaf_paths(self) -> dict[str, Any]:
        """Every citable path in the bundle, flattened, with its value.

        This is both the model's menu and the checker's ground truth. A path that is not
        in here cannot be cited; a rendered numeral that did not come from here is a
        faithfulness failure.
        """
        out: dict[str, Any] = {}

        def walk(obj: Any, prefix: str) -> None:
            if isinstance(obj, BaseModel):
                obj = obj.model_dump()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{prefix}.{k}" if prefix else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{prefix}[{i}]")
            elif obj is not None:
                out[prefix] = obj

        walk(self, "")
        return out


_PATH_STEP = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")


def resolve_path(bundle: EvidenceBundle, path: str) -> Any:
    """Resolve a dotted/indexed bundle path, or raise KeyError.

    Kept mechanical and total: the caller treats a KeyError as a schema violation, which
    is what an invented citation is.
    """
    cur: Any = bundle.model_dump()
    for m in _PATH_STEP.finditer(path):
        name, idx = m.group(1), m.group(2)
        try:
            if name is not None:
                cur = cur[name]
            else:
                cur = cur[int(idx)]
        except (KeyError, IndexError, TypeError) as e:
            raise KeyError(f"unresolvable bundle path: {path!r}") from e
    return cur


def format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4f}"
    return str(v)


# --------------------------------------------------------------------------------------
# The reading — the typed output
# --------------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")

#: Prose fields carry no digits. Numbers arrive only as {bundle.path} placeholders that
#: deterministic code substitutes after the model is done. This is the schema-level form
#: of the rule in llm/schema.py: an extractor physically cannot return a number, and a
#: synthesiser physically cannot write one.
_BARE_DIGIT = re.compile(r"\d")


class NumberLeak(ValueError):
    """Raised when a prose field contains a digit outside a citation placeholder."""


#: Form names, not numbers. "13F" is what the filing is called; barring it would be the
#: validator mistaking a proper noun for a quantity.
_FORM_NAMES = re.compile(r"\b(13F|13D|13G|8-K|10-K|10-Q|VIX)\b")


def _no_bare_numbers(v: str) -> str:
    stripped = _FORM_NAMES.sub("", _PLACEHOLDER.sub("", v))
    if _BARE_DIGIT.search(stripped):
        bad = [t for t in re.findall(r"\S*\d\S*", stripped)]
        raise NumberLeak(
            "prose contains a digit outside a {bundle.field.path} placeholder: "
            f"{bad[:5]}. Every numeral must be written as a placeholder naming the "
            "bundle field it comes from, e.g. '{severity.var_conditional}'."
        )
    return v


def _cites_at_least(n: int):
    """A reading with no numerals in it is not a reading of an evidence bundle.

    Without this the digit bar has a trivial escape: write pure qualitative prose and
    cite nothing. The first draft of this layer did exactly that -- zero placeholders,
    a fluent page, and a traceability rate that was 100 percent of nothing.
    """

    def check(v: str) -> str:
        found = _PLACEHOLDER.findall(v)
        if len(found) < n:
            raise ValueError(
                f"this field must cite at least {n} bundle field(s) as "
                "{bundle.field.path} placeholders; found "
                f"{len(found)}. State the actual evidence, do not paraphrase it away.")
        return v

    return check


class Driver(BaseModel):
    """One ranked driver of the assessment, with the bundle field it rests on."""

    rank: int = Field(ge=1, le=5, description="1 is the most important driver.")
    channel: Channel = Field(description="Which evidence channel this driver comes from.")
    source_field: str = Field(
        description=(
            "The single bundle path this driver principally rests on, e.g. "
            "'severity.es_conditional' or 'crowding.top10_share_pct'. Must be a path that "
            "appears in the FIELD INDEX."
        )
    )
    direction: ClaimDirection = Field(
        description="Whether this driver raises or lowers momentum reversal risk.")
    statement: str = Field(
        max_length=400,
        description=(
            "One or two sentences saying what this evidence shows. NO DIGITS. Cite every "
            "number as a {bundle.field.path} placeholder taken verbatim from the FIELD "
            "INDEX; the placeholder is replaced with the bundle's own value afterwards."
        ),
    )

    _v = field_validator("statement")(_no_bare_numbers)
    _v2 = field_validator("statement")(_cites_at_least(1))


class FalsifierRef(BaseModel):
    """A named condition that would refute the reading, taken from the bundle."""

    name: str = Field(max_length=160, description="Short label. NO DIGITS.")
    source_field: str = Field(
        description="Bundle path of the armed falsifier this refers to.")
    statement: str = Field(
        max_length=400,
        description=("What would have to happen for the reading to be wrong. NO DIGITS; "
                     "cite thresholds as {bundle.field.path} placeholders."))

    _v1 = field_validator("name")(_no_bare_numbers)
    _v2 = field_validator("statement")(_no_bare_numbers)


class CouldNotMeasure(BaseModel):
    """An explicit refusal on one channel. A gap, not a zero."""

    channel: Channel = Field(description="Which channel is absent from the bundle.")
    reason: str = Field(
        max_length=300,
        description=("Why nothing can be said about it. NO DIGITS. State that the evidence "
                     "is absent — do not speculate about what it would have shown."),
    )

    _v = field_validator("reason")(_no_bare_numbers)


class Reading(BaseModel):
    """The PM-facing reading: an argued, cited assessment of one evidence bundle."""

    declined: bool = Field(
        description=("True if the surviving evidence cannot support any assessment at all. "
                     "A declined reading is a correct answer, not a failure."))
    decline_reason: str = Field(
        default="",
        max_length=400,
        description="Why the whole reading is declined; empty string if it is not. NO DIGITS.")
    headline_direction: HeadlineDirection = Field(
        description=("The direction of the assessment for the declared horizon. Use 'mixed' "
                     "when the channels genuinely disagree, not as a hedge."))
    headline: str = Field(
        max_length=1100,
        description=("Two to four sentences a PM reads first: what this book is levered to and "
                     "how much reversal risk it is carrying. NO DIGITS; cite every number as a "
                     "{bundle.field.path} placeholder from the FIELD INDEX."),
    )
    drivers: list[Driver] = Field(
        default_factory=list,
        description="Ranked drivers, most important first, at most five. Empty if declined.")
    falsifiers: list[FalsifierRef] = Field(
        default_factory=list,
        description="Named conditions from the bundle that would refute this reading.")
    could_not_measure: list[CouldNotMeasure] = Field(
        default_factory=list,
        description=("One entry for EVERY channel absent from the bundle. Naming an absent "
                     "channel here is required; saying anything substantive about an absent "
                     "channel anywhere else in this object is an error."),
    )

    _v1 = field_validator("headline")(_no_bare_numbers)
    _v2 = field_validator("decline_reason")(_no_bare_numbers)
    _v3 = field_validator("headline")(_cites_at_least(2))


# --------------------------------------------------------------------------------------
# Rendering — placeholders become bundle values, deterministically
# --------------------------------------------------------------------------------------


class RenderResult(BaseModel):
    text: str
    cited_paths: list[str]
    unresolved: list[str]


def render(text: str, bundle: EvidenceBundle) -> RenderResult:
    cited, unresolved = [], []

    def sub(m: re.Match) -> str:
        path = m.group(1).strip()
        try:
            v = resolve_path(bundle, path)
        except KeyError:
            unresolved.append(path)
            return f"[UNRESOLVED:{path}]"
        if isinstance(v, (dict, list)):
            # A placeholder names one value. Pointing it at a whole sub-object would put a
            # serialised dict in the middle of a sentence, so it is a violation.
            unresolved.append(path + " (not a single value)")
            return f"[UNRESOLVED:{path}]"
        cited.append(path)
        return format_value(v)

    return RenderResult(text=_PLACEHOLDER.sub(sub, text), cited_paths=cited,
                        unresolved=unresolved)


def render_reading(reading: Reading, bundle: EvidenceBundle) -> dict:
    """Substitute every placeholder and report what was cited and what did not resolve."""
    cited: list[str] = []
    unresolved: list[str] = []

    def r(s: str) -> str:
        out = render(s, bundle)
        cited.extend(out.cited_paths)
        unresolved.extend(out.unresolved)
        return out.text

    body = {
        "declined": reading.declined,
        "decline_reason": r(reading.decline_reason),
        "headline_direction": reading.headline_direction.value,
        "headline": r(reading.headline),
        "drivers": [],
        "falsifiers": [],
        "could_not_measure": [{"channel": c.channel.value, "reason": r(c.reason)}
                              for c in reading.could_not_measure],
    }
    for d in sorted(reading.drivers, key=lambda x: x.rank):
        try:
            resolve_path(bundle, d.source_field)
        except KeyError:
            unresolved.append(d.source_field)
        body["drivers"].append({
            "rank": d.rank, "channel": d.channel.value, "source_field": d.source_field,
            "direction": d.direction.value, "statement": r(d.statement)})
    for f in reading.falsifiers:
        try:
            resolve_path(bundle, f.source_field)
        except KeyError:
            unresolved.append(f.source_field)
        body["falsifiers"].append({"name": r(f.name), "source_field": f.source_field,
                                   "statement": r(f.statement)})
    body["_cited_paths"] = sorted(set(cited))
    body["_unresolved_paths"] = sorted(set(unresolved))
    return body


def as_text(rendered: dict) -> str:
    """The reading as a human reads it. This is what gets saved verbatim."""
    L = [f"HEADLINE [{rendered['headline_direction'].upper()}]", "  " + rendered["headline"]]
    if rendered["declined"]:
        L += ["", "DECLINED", "  " + rendered["decline_reason"]]
    if rendered["drivers"]:
        L += ["", "DRIVERS"]
        for d in rendered["drivers"]:
            L += [f"  {d['rank']}. [{d['channel']}/{d['direction']}] "
                  f"(source: {d['source_field']})", f"     {d['statement']}"]
    if rendered["falsifiers"]:
        L += ["", "FALSIFIERS"]
        for f in rendered["falsifiers"]:
            L += [f"  - {f['name']}  (source: {f['source_field']})", f"    {f['statement']}"]
    L += ["", "COULD NOT MEASURE"]
    if rendered["could_not_measure"]:
        for c in rendered["could_not_measure"]:
            L.append(f"  {c['channel']:12s} {c['reason']}")
    else:
        L.append("  (nothing declared absent)")
    L += ["", "  These are gaps, not zeros. A quiet reading from a channel that did not run",
          "  is not a quiet market."]
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# The synthesis call
# --------------------------------------------------------------------------------------

SYSTEM = """You are the synthesis layer of a momentum-factor tail-risk monitor. You are \
handed a bundle of independently computed evidence and you compose the argued, cited \
reading a portfolio manager consumes. You are not a forecaster and you are not an \
analyst with outside knowledge: everything you assert must rest on the bundle in front \
of you.

THE ONE RULE THAT IS ENFORCED IN CODE, NOT BY YOUR DISCIPLINE:
You may not write a digit. Every numeral you want on the page must be written as a
placeholder naming the bundle field it came from -- {severity.es_conditional},
{crowding.top_filers[0].filer}, {contract.declared_horizon_days} -- copied verbatim from
the FIELD INDEX below. Code replaces each placeholder with the bundle's own value after
you are done. A digit anywhere in your prose is rejected by the schema. A placeholder
naming a path that is not in the FIELD INDEX is rejected too. There is no field in which
you can put a number of your own, and this is deliberate.

WHAT FAITHFULNESS MEANS HERE:
* Move with the evidence. If a channel reads differently, the reading must read
  differently. A reading that would be the same whatever the numbers said is worthless.
* Refuse without it. Channels can be absent. For EVERY absent channel, add a
  could_not_measure entry naming it, and say NOTHING substantive about it anywhere
  else -- no estimate, no "likely", no "typically", no inference from the other
  channels about what it would have shown. An absent channel is a gap, not a zero, and
  not an invitation.
* If the surviving evidence cannot support any assessment, set declined and say why.
  A declined reading is a correct answer.
* Rank drivers by how much they move the assessment, not by how interesting they are.
  Each driver names the one bundle field it principally rests on.
* CITE THE ACTUAL NUMBERS. The digit bar is not an instruction to write number-free
  prose; it is the mechanism by which your numbers are guaranteed to be the bundle's.
  A PM needs the levels, the percentiles and the thresholds on the page. The headline
  must carry at least two placeholders and every driver at least one. A placeholder must
  name a single value, never a whole sub-object.
* Read the disclosures. Where the bundle says a panel is descriptive only, or that a
  number is computed on a different book, do not upgrade it."""

USER = """EVIDENCE BUNDLE {bundle_id}

CHANNELS PRESENT: {present}
CHANNELS ABSENT:  {absent}

FIELD INDEX -- this IS the bundle, flattened: every value it holds, at the path that
names it. A placeholder must name one of these paths exactly. Nothing outside this index
is evidence, and nothing outside it is citable.

{index}

Compose the reading for {intended_user}.
The decision it informs: {decision}
The declared horizon and quantile are in the contract channel; use them by placeholder.

Remember: no digits in prose; every number as a placeholder naming a single value in the
FIELD INDEX; one could_not_measure entry per absent channel; nothing substantive about an
absent channel."""


def _payload(bundle: EvidenceBundle) -> str:
    d = bundle.model_dump(exclude_none=True)
    d.pop("bundle_id", None)
    return json.dumps(d, indent=1, default=str)


def _index(bundle: EvidenceBundle) -> str:
    return "\n".join(f"  {k} = {format_value(v)}" for k, v in bundle.leaf_paths().items())


def build_messages(bundle: EvidenceBundle) -> list[dict]:
    return [{"role": "user", "content": USER.format(
        bundle_id=bundle.bundle_id,
        present=", ".join(c.value for c in bundle.present_channels()) or "(none)",
        absent=", ".join(c.value for c in bundle.missing_channels()) or "(none)",
        index=_index(bundle),
        intended_user=bundle.contract.intended_user,
        decision=bundle.contract.decision)}]


class SynthesisOutcome(BaseModel):
    bundle_id: str
    ok: bool
    attempts: int
    violations: list[str] = Field(default_factory=list)
    reading: Reading | None = None
    rendered: dict | None = None
    usage: dict = Field(default_factory=dict)
    error: str = ""


#: Sonnet 5 runs adaptive thinking whether or not it is asked to, and at default effort it
#: spent an entire 4k budget thinking and returned nothing. Effort is set explicitly and
#: the ceiling is generous: a truncated draft is a measurement of the harness, not of the
#: model.
EFFORT = "low"


def synthesise(bundle: EvidenceBundle, *, client=None, model: str = DEFAULT_MODEL,
               max_tokens: int = 8000, max_attempts: int = 3) -> SynthesisOutcome:
    """One synthesis call, with the schema violations fed back rather than swallowed.

    A rejected draft is information, not an error to hide: the violation list is part of
    the result so the faithfulness cell can report how often the model tried to write a
    number of its own.
    """
    import anthropic

    cli = client or anthropic.Anthropic()
    msgs = build_messages(bundle)
    violations: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    for attempt in range(1, max_attempts + 1):
        try:
            resp = cli.messages.parse(model=model, max_tokens=max_tokens, system=SYSTEM,
                                      messages=msgs, output_format=Reading,
                                      output_config={"effort": EFFORT})
        except Exception as e:  # noqa: BLE001 -- a rejected draft is a measured outcome
            msg = f"{type(e).__name__}: {e}"
            violations.append(msg)
            if attempt == max_attempts:
                return SynthesisOutcome(bundle_id=bundle.bundle_id, ok=False,
                                        attempts=attempt, violations=violations,
                                        usage=usage, error=msg)
            msgs = msgs[:1] + [
                {"role": "user", "content":
                 "Your previous draft was rejected by the schema:\n" + msg[:1500] +
                 "\nRewrite it. Every numeral must be a {bundle.field.path} placeholder "
                 "copied from the FIELD INDEX; prose fields may not contain digits."}]
            continue

        u = resp.usage
        usage["input_tokens"] += u.input_tokens
        usage["output_tokens"] += u.output_tokens
        usage["calls"] += 1
        reading = resp.parsed_output
        rendered = render_reading(reading, bundle)
        if rendered["_unresolved_paths"]:
            bad = rendered["_unresolved_paths"]
            violations.append(f"unresolved bundle paths: {bad}")
            if attempt == max_attempts:
                return SynthesisOutcome(bundle_id=bundle.bundle_id, ok=False,
                                        attempts=attempt, violations=violations,
                                        reading=reading, rendered=rendered, usage=usage,
                                        error=f"unresolved paths {bad}")
            msgs = msgs[:1] + [
                {"role": "user", "content":
                 f"These citation paths are not in the FIELD INDEX and do not exist in the "
                 f"bundle: {bad}. Rewrite the reading using only paths that appear in the "
                 f"FIELD INDEX."}]
            continue

        return SynthesisOutcome(bundle_id=bundle.bundle_id, ok=True, attempts=attempt,
                                violations=violations, reading=reading, rendered=rendered,
                                usage=usage)

    return SynthesisOutcome(bundle_id=bundle.bundle_id, ok=False, attempts=max_attempts,
                            violations=violations, usage=usage, error="exhausted attempts")


# --------------------------------------------------------------------------------------
# The independent auditor — sees the bundle and the reading, never the generating prompt
# --------------------------------------------------------------------------------------


class JudgedClaim(BaseModel):
    claim: str = Field(max_length=400, description="The qualitative claim, quoted from the reading.")
    supported: bool = Field(description="Is this claim supported by the evidence bundle?")
    why: str = Field(max_length=300, description="One sentence.")


class ChannelSilence(BaseModel):
    channel: str = Field(description="A channel named as absent from the bundle.")
    reading_makes_substantive_claim: bool = Field(
        description=("True if the reading asserts anything substantive about this absent "
                     "channel -- a level, a direction, a comparison, a likelihood, or an "
                     "inference about what it would have shown. Naming it as unmeasured, "
                     "or saying its absence limits the reading, is NOT a substantive claim."))
    quote: str = Field(default="", max_length=300,
                       description="The offending sentence, or empty string if none.")


class Audit(BaseModel):
    claims: list[JudgedClaim] = Field(
        description="Every qualitative (non-numeric) claim in the reading, judged.")
    absent_channel_checks: list[ChannelSilence] = Field(
        default_factory=list,
        description="One entry per channel listed as absent from the bundle.")
    overall_note: str = Field(max_length=400)


AUDIT_SYSTEM = """You are an independent auditor. You are shown an evidence bundle and a \
written reading of it. You did NOT see the instructions the writer was given, and you \
should not assume what they were.

Your job is adversarial in one direction only: find claims the bundle does not support. \
Do not reward fluency. A claim is supported only if the bundle contains evidence for it; \
a claim that is plausible, conventional, or true of markets generally but absent from the \
bundle is UNSUPPORTED.

Judge every qualitative claim -- assertions about direction, mechanism, comparison, \
importance, or what follows from what. Numeric values in the reading were substituted \
from the bundle by machine and are not your concern; what the reading *says about* them \
is.

Separately: the bundle lists some channels as ABSENT. For each, decide whether the \
reading makes any substantive claim about it. Naming it as unmeasured is correct \
behaviour and is not a substantive claim. Asserting a level, a direction, a likelihood, \
or an inference about what it would have shown IS."""

AUDIT_USER = """EVIDENCE BUNDLE -- every value it holds, at the path that names it
{payload}

CHANNELS ABSENT FROM THE BUNDLE: {absent}

THE READING
{reading}

Judge it."""


def audit(bundle: EvidenceBundle, rendered: dict, *, client=None,
          model: str = DEFAULT_MODEL, max_tokens: int = 8000) -> tuple[Audit | None, dict, str]:
    import anthropic

    cli = client or anthropic.Anthropic()
    absent = ", ".join(c.value for c in bundle.missing_channels()) or "(none)"
    try:
        resp = cli.messages.parse(
            model=model, max_tokens=max_tokens, system=AUDIT_SYSTEM,
            messages=[{"role": "user", "content": AUDIT_USER.format(
                payload=_index(bundle), absent=absent, reading=as_text(rendered))}],
            output_format=Audit, output_config={"effort": EFFORT})
    except Exception as e:  # noqa: BLE001
        return None, {"calls": 0, "input_tokens": 0, "output_tokens": 0}, f"{type(e).__name__}: {e}"
    return (resp.parsed_output,
            {"calls": 1, "input_tokens": resp.usage.input_tokens,
             "output_tokens": resp.usage.output_tokens}, "")


# --------------------------------------------------------------------------------------
# Traceability — mechanical, no model involved
# --------------------------------------------------------------------------------------

_NUMERAL = re.compile(r"-?\d[\d,]*\.?\d*")


def traceability(reading: Reading, rendered: dict, bundle: EvidenceBundle,
                 *, tol: float = 1e-9) -> dict:
    """Match every numeral in the rendered reading back to a bundle field.

    The placeholder mechanism means this should be perfect by construction; it is checked
    anyway, because "should be perfect by construction" is how this repo's trap log gets
    longer. Any numeral that appears in the rendered text but not in the set of values the
    cited paths hold is untraceable and is reported as such.
    """
    values = {format_value(resolve_path(bundle, p)) for p in rendered["_cited_paths"]}
    prose = " ".join(
        [rendered["headline"], rendered["decline_reason"]]
        + [d["statement"] for d in rendered["drivers"]]
        + [f["name"] + " " + f["statement"] for f in rendered["falsifiers"]]
        + [c["reason"] for c in rendered["could_not_measure"]])
    found = _NUMERAL.findall(prose)
    traced, untraced = [], []
    for tok in found:
        if any(tok in v or v in tok for v in values):
            traced.append(tok)
        else:
            untraced.append(tok)
    del tol
    return {"n_numerals": len(found), "n_traced": len(traced), "untraced": untraced,
            "rate": (len(traced) / len(found)) if found else 1.0,
            "n_cited_paths": len(rendered["_cited_paths"])}


def load_bundle(path: str | Path) -> EvidenceBundle:
    return EvidenceBundle.model_validate_json(Path(path).read_text())
