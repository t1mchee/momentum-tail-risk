"""The dual harvest: what a filing says in its own words, and how it reads against eight anchors.

Two harvests from one read, kept apart on purpose.

OPEN harvest first: every material stated exposure the company names, in the company's own
language, unprompted by any list. This is the half that can find what a fixed taxonomy cannot,
and it is asked FIRST so the anchors cannot prime it.

ANCHORED harvest second: the eight registered themes, each assessed explicitly, so every record
is comparable across firms and quarters whatever the open harvest returned.

Every claim in both halves carries a verbatim quote, and the schema version is stamped into the
record because a record whose shape is unknown is a record that cannot be pooled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .. import config as _config  # noqa: F401  -- puts the key in this process
from .exposure import quote_is_grounded

SCHEMA_VERSION = "harvest-v3"
CACHE = Path("data/interim/harvest")

THEMES = ("rates_duration", "inflation_pricing", "fx_dollar", "energy_commodities",
          "china_trade", "credit_refinancing", "supply_chain", "ai_compute")

#: Six levels, not four. The v1 vocabulary forced a directional guess onto disclosure that states
#: an exposure and declines to state a direction, and it collapsed an explicit denial into silence.
#: Both defects were measured before the corpus was bought: on sixty passages read by two tiers,
#: 28 of 73 disagreements cited the IDENTICAL quote, and on fx_dollar -- where disclosure is almost
#: always two-sided -- nine of twelve did. That is the schema guessing, not the reader.
#:
#:   hurt_by_rising / benefits_from_rising  the passage states a SIGNED exposure
#:   exposed_direction_unstated             "a 10% appreciation OR depreciation would result in"
#:   hedged_neutral                         exposure stated AND offset by a described hedge
#:   stated_immaterial                      "we bear no significant foreign exchange risk"
#:   not_discussed                          the passage does not address the theme
#:
#: The last two were one label in v1. They are different facts -- a denial is information and
#: silence is its absence -- and the firm-by-quarter panel is specified to carry three-state
#: missingness, which four levels cannot express.
Direction = Literal["hurt_by_rising", "benefits_from_rising", "exposed_direction_unstated",
                    "hedged_neutral", "stated_immaterial", "not_discussed"]


class StatedExposure(BaseModel):
    """One exposure the company names itself, in its own words."""

    exposure: str = Field(description="the exposure in the company's own phrasing, under 12 words")
    direction: Direction
    quote: str = Field(description="one sentence quoted verbatim from the passage")


class ThemeRead(BaseModel):
    theme: str
    direction: Direction
    magnitude: int = Field(ge=0, le=3, description="0 none, 1 mentioned, 2 material, 3 primary")
    quote: str = Field(description="verbatim sentence, or empty when direction is not_discussed")


class Harvest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    stated: list[StatedExposure] = Field(default_factory=list)
    themes: list[ThemeRead] = Field(default_factory=list)


SYSTEM = """You read one passage from a US company's regulatory filing and report what it says
about the company's exposures. You report only what the passage states. You never infer an
exposure from the industry, the company name, or general knowledge.

Do TWO things, in this order.

FIRST, the open harvest. List every material exposure the passage states, in the company's own
phrasing. Do not consult any list. If the passage states no material exposure, return an empty
list. Each entry carries one sentence quoted VERBATIM from the passage.

SECOND, the anchored harvest. For each of these eight themes, say what the passage states:
rates_duration, inflation_pricing, fx_dollar, energy_commodities, china_trade,
credit_refinancing, supply_chain, ai_compute.
Direction is one of six. Choose by what the passage STATES, never by what is likely:

  hurt_by_rising              the passage says the company is hurt when the DRIVER RISES
  benefits_from_rising        the passage says the company is hurt when the DRIVER FALLS, or
                              gains when it rises
  exposed_direction_unstated  the passage states an exposure WITHOUT saying which way it cuts.
                              Use this for two-sided language -- "a 10% appreciation or
                              depreciation would result in", "we are exposed to changes in
                              interest rates", "fluctuations in foreign exchange rates". This is
                              the correct answer far more often than it looks; do not resolve a
                              direction the filing did not give you.
  hedged_neutral              an exposure is stated AND the passage describes a hedge or offset
                              that neutralises it. Requires the hedge to be described, not assumed.
  stated_immaterial           the passage explicitly says there is no material exposure -- "we
                              bear no significant foreign exchange risk", "would not have a
                              material impact". A denial is a statement and is recorded as one.
  not_discussed               the passage does not address this theme at all.

DIRECTION IS ABOUT WHICH WAY THE DRIVER MOVES, NOT ABOUT WHETHER THE PASSAGE SOUNDS BAD. This is
the most common way to get a direction wrong, because filings describe risk far more often as harm
from a FALL than harm from a rise:

  "a substantial or extended decline in metals prices would have a material adverse effect on us"
      -> benefits_from_rising. The company does BETTER when metals rise. It is not hurt_by_rising.
  "oil and gas prices have declined significantly, resulting in lower expenditures by the industry"
      -> benefits_from_rising, for a company selling into that industry.
  "the imposition of tariffs may negatively impact our costs"
      -> hurt_by_rising. Here the driver itself is going up.

Read the sentence for which way the DRIVER moves, then ask what the passage says happens to the
company. If the passage only says the theme is bad for the company without saying which way the
driver moves, that is exposed_direction_unstated.

The distinction between stated_immaterial and not_discussed is load-bearing and is not a matter of
degree: the first is something the company said, the second is something it did not say.

Magnitude is 0 when the theme is absent, 1 when mentioned in passing, 2 when material, 3 when a
primary stated exposure. When direction is not_discussed, the quote is an empty string; every
other direction, stated_immaterial included, requires a verbatim quote.

Every non-empty quote must appear word for word in the passage. A quote you cannot find in the
passage is a failure, not an approximation."""

USER = "Passage from {ticker}, section {section}:\n\n{text}"


def prompt_hash() -> str:
    return hashlib.sha256((SYSTEM + USER + SCHEMA_VERSION).encode()).hexdigest()[:16]


class Harvester:
    """One passage, one read, both harvests. Cached by content, prompt and model together."""

    def __init__(self, model: str, max_tokens: int = 12_000, enforce_quotes: bool = True):
        # 4,000 was the first ceiling and it BOUND on the cheaper tier: sonnet's clean responses
        # topped out at 3,933 tokens with a p90 of 3,677, and 22 of 60 passages truncated mid-JSON.
        # The failures surfaced as `NoneType has no attribute stated` and `EOF while parsing`,
        # neither of which names a ceiling, so the cause had to be inferred from the token counts.
        # Reported without that check it would have read as a 37 percent failure rate for the
        # model -- an instrument bug attributed to a capability difference.
        self.model = model
        self.max_tokens = max_tokens
        self.enforce_quotes = enforce_quotes

    def _key(self, ticker: str, section: str, text: str) -> str:
        return hashlib.sha256(
            f"{ticker}|{section}|{self.model}|{prompt_hash()}|{text[:4000]}".encode()
        ).hexdigest()[:20]

    def read(self, ticker: str, section: str, text: str, *, use_cache: bool = True) -> dict:
        # `max_tokens` is deliberately NOT in the cache key, which looks like the trp-59 bug
        # (an embedding cache that omitted `max_length`) and is not. There the parameter changed
        # the vector for EVERY input; here the ceiling can only ever truncate, and a truncated
        # response raises before the write below, so nothing truncated is ever cached. Every
        # cached result is therefore untruncated and identical under any higher ceiling.
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"{self._key(ticker, section, text)}.json"
        if use_cache and f.exists():
            d = json.loads(f.read_text())
            d["from_cache"] = True
            return d
        import anthropic
        import time

        t0 = time.time()
        resp = anthropic.Anthropic().messages.parse(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            messages=[{"role": "user",
                       "content": USER.format(ticker=ticker, section=section, text=text)}],
            output_format=Harvest,
        )
        # Truncation must announce itself. Without this the ceiling surfaces as a confusing parse
        # error several frames away from its cause, which is exactly how it cost a pilot run.
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise RuntimeError(
                f"output truncated at max_tokens={self.max_tokens} for {ticker}/{section}; "
                f"raise the ceiling rather than treating this as a model failure")
        h = resp.parsed_output
        # The quote gate is mechanical and runs on every claim, because hallucinated evidence is
        # the failure mode this layer exists to prevent and it is checkable for free.
        checked = []
        for s in h.stated:
            checked.append(bool(s.quote) and quote_is_grounded(s.quote, text))
        tchecked = []
        for th in h.themes:
            tchecked.append(True if th.direction == "not_discussed" and not th.quote
                            else bool(th.quote) and quote_is_grounded(th.quote, text))
        out = {"schema_version": SCHEMA_VERSION, "prompt_hash": prompt_hash(),
               "model": self.model, "ticker": ticker, "section": section,
               "harvest": json.loads(h.model_dump_json()),
               "n_stated": len(h.stated), "n_themes": len(h.themes),
               "stated_quotes_grounded": int(sum(checked)),
               "theme_quotes_grounded": int(sum(tchecked)),
               "schema_valid": len(h.themes) == len(THEMES),
               "latency_s": round(time.time() - t0, 2),
               "input_tokens": getattr(getattr(resp, "usage", None), "input_tokens", None),
               "output_tokens": getattr(getattr(resp, "usage", None), "output_tokens", None),
               "from_cache": False}
        f.write_text(json.dumps(out, indent=2))
        return out
