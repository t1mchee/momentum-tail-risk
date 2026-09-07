"""Ask a model to name what a book jointly bets on, and which holdings carry each bet.

Why this and not another similarity measure
-------------------------------------------
Four text nulls in this project reduce to one. News carries no factor exposure, filing
similarity recovers industry, disclosed rate exposure is a funding cost rather than an equity
duration, and theme concentration turned out to be a coverage count. Each asked ISSUERS about
THEMSELVES and aggregated upward. But no company's annual report says it is a rates proxy,
because from inside the company it is not one -- that label is assigned by the market, and it
lives in return covariance rather than in any disclosure.

A model is the one instrument here that might supply the label an issuer cannot, because the
label is exactly the kind of thing world knowledge encodes. This is classification, which is
the role the project has measured the model succeeding in -- 97 percent precision on
aboutness, factual labels invariant to date-swapping -- and not forecasting, which is the
role it has measured the model failing in every time.

What the model is and is not shown
----------------------------------
It sees ticker, company name, portfolio weight and sector. It does not see returns, dates,
episode labels, or any outcome. Date blinding is known not to work in this project: a model
placed every historical window correctly with dates removed. So blinding is not claimed as a
defence. What bounds contamination instead is the target: the test scores forward co-movement
among the named groups, which is not a fact any document records and not an event the model
could be recalling.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-opus-4-8"
CACHE = Path("data/interim/partitions")

SYSTEM = """\
You are a risk analyst examining the constituents of an equity portfolio.

Your task is to identify what the holdings JOINTLY bet on -- the shared exposures that would
cause many of these names to fall together, beyond the fact that some share an industry.

A useful answer names exposures that CUT ACROSS sectors. "Technology companies" is not an
exposure, it is a sector, and the reader already has sector labels. "Companies whose
valuations depend on long-duration cash flows and therefore on the discount rate" is an
exposure, and it may include names from technology, real estate, utilities and staples alike.

Rules:
- Name between 2 and 8 exposures. Fewer is better than padding.
- Assign each ticker to AT MOST ONE exposure. Leave a ticker unassigned if it does not
  clearly carry any of the exposures you named.
- Do not invent tickers. Use only those given.
- If the portfolio genuinely has no shared exposure beyond its sector composition, say so by
  returning a single group named "no shared exposure beyond sector" and assigning nothing.
  That is a valid and useful answer, not a failure.
"""

USER = """\
Portfolio constituents ({n} names). Columns are ticker, company, weight percent, sector.

{table}

Name the shared exposures and assign the tickers.
"""


class ExposureGroup(BaseModel):
    name: str = Field(description="Short name for the shared exposure, cutting across sectors")
    rationale: str = Field(description="One sentence on the economic channel")
    tickers: list[str] = Field(description="Tickers from the input carrying this exposure")


class Partition(BaseModel):
    groups: list[ExposureGroup]


@dataclass
class BookPartitioner:
    model: str = DEFAULT_MODEL
    max_tokens: int = 8000

    def _key(self, holdings: pd.DataFrame, tag: str) -> str:
        body = "|".join(sorted(holdings["ticker"].astype(str))) + tag + self.model
        return hashlib.sha256(body.encode()).hexdigest()[:20]

    def partition(self, holdings: pd.DataFrame, *, tag: str = "", use_cache: bool = True) -> Partition:
        """One partition of one book. Cached by holdings content, not by date.

        Keying the cache on the holdings themselves rather than on a date means the random
        basket and the real book cannot collide, and a re-run with the same book costs
        nothing.
        """
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"{self._key(holdings, tag)}.json"
        if use_cache and f.exists():
            return Partition(**json.loads(f.read_text()))

        import anthropic

        table = "\n".join(
            f"{r.ticker}\t{str(r.name_)[:38]}\t{r.weight:.3f}\t{r.sector}"
            for r in holdings.itertuples()
        )
        client = anthropic.Anthropic()
        resp = client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": USER.format(n=len(holdings), table=table)}],
            output_format=Partition,
        )
        out = resp.parsed_output
        f.write_text(out.model_dump_json(indent=2))
        return out


def to_labels(p: Partition, universe: list[str]) -> pd.Series:
    """Group label per ticker, NaN where the model assigned nothing.

    Tickers the model invented are dropped rather than kept: an assignment to a name not in
    the book is a hallucination and counting it would inflate group sizes with holdings the
    portfolio does not have.
    """
    known = set(universe)
    out = {}
    for i, g in enumerate(p.groups):
        for t in g.tickers:
            t = str(t).strip().upper()
            if t in known and t not in out:
                out[t] = i
    return pd.Series(out, name="group").reindex(universe)


def hallucination_rate(p: Partition, universe: list[str]) -> float:
    """Share of assigned tickers that are not in the book at all."""
    known = {u.upper() for u in universe}
    assigned = [str(t).strip().upper() for g in p.groups for t in g.tickers]
    if not assigned:
        return 0.0
    return sum(1 for t in assigned if t not in known) / len(assigned)
