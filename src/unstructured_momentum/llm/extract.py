"""Evidence extraction: a keyword baseline and a Claude extractor, built to be compared.

Why both
--------
The brief asks whether AI *materially improves* the research process. That is only
answerable against a baseline, so both are implemented and evaluated on the same corpus.

The baseline is not a strawman -- it is exactly what a keyword-count feature does, and it
is what this project would have shipped had the contamination not been measured. On the
Sept-2019 pre-event window it marks 246 articles as crowding evidence, of which 202 are
one content farm's templated shareholder-composition filler and 4 are Reuters. If the
Claude extractor cannot beat that, the honest finding is that the LLM adds nothing here,
and that is a reportable result rather than a failure.

What the LLM is for, and what it is not for
-------------------------------------------
It judges **relevance** and extracts a **typed, cited claim**. It never emits a
probability, a score, or a forecast -- `EvidenceRecord` has no field for one (see
`schema.py`). Aggregation and every number come from deterministic code.

Cost
----
Corpus classification is not latency-sensitive, so `extract_batch` uses the Message
Batches API at 50% of standard price. Prompt caching is deliberately *not* used: the
shared system prompt is well under Opus 4.8's 4096-token minimum cacheable prefix, so a
`cache_control` marker would silently do nothing while implying otherwise.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .schema import Direction, EvidenceRecord, ExtractionResult, Leg, Mechanism, Timing

#: Default per the project's model policy. Corpus classification runs through the Batch
#: API at half price rather than being downgraded to a smaller model -- picking a cheaper
#: tier is a quality decision that belongs to the user, not a silent default here.
DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You judge whether a news headline is genuine evidence about US equity FACTOR positioning \
-- specifically the momentum factor -- or whether it merely contains matching keywords.

Context you need:
- "Momentum" as a factor means a long/short portfolio of past winners vs past losers. It \
is not a single company's share price rising.
- Crowding means many investors hold the same factor exposure, making an unwind \
self-reinforcing.
- Deleveraging means levered market-neutral books cutting gross exposure, which hits every \
factor at once.
- Rotation means market leadership shifting between factors or styles (e.g. momentum to \
value).

Mark is_relevant=false for:
- Templated or auto-generated pieces (shareholder-composition boilerplate, screener \
output, "What Kind Of Investor Owns..." formats, syndicated filler).
- Single-company news with no factor or positioning angle.
- Generic market wrap-ups that mention a keyword in passing.
- Pieces about price momentum of one stock rather than the momentum factor.

Judge timing carefully and honestly. Financial media overwhelmingly reports factor moves \
AFTER they happen. A piece describing an unwind that already occurred is RETROSPECTIVE, \
however dramatic its language. Only mark PROSPECTIVE when the text genuinely warns about \
or anticipates something ahead. Do not inflate retrospective coverage into a warning.

You are judging text only. Never infer, state, or imply any probability of a market \
outcome. Your confidence field refers solely to your relevance judgment.\
"""

USER_TEMPLATE = """\
Source: {domain}
Published: {seendate}
Headline: {title}

Judge this headline against the criteria."""


# --------------------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------------------


@dataclass
class KeywordExtractor:
    """The naive baseline: a keyword match is treated as evidence.

    This is what a chatter-intensity feature does. It has no notion of whether the text is
    about the momentum *factor* or about one company's share price, and no notion of
    whether the piece precedes or follows the move it describes.
    """

    name: str = "keyword"

    def extract(self, articles: pd.DataFrame, mechanism: str = "crowding") -> list[ExtractionResult]:
        out: list[ExtractionResult] = []
        for _, a in articles.iterrows():
            out.append(
                ExtractionResult(
                    record=EvidenceRecord(
                        # The baseline's defining property: matching the query IS the
                        # judgment. It cannot tell filler from analysis.
                        is_relevant=True,
                        mechanism=Mechanism(mechanism)
                        if mechanism in {m.value for m in Mechanism}
                        else Mechanism.NONE,
                        direction=Direction.INCREASES_RISK,
                        leg=Leg.NOT_APPLICABLE,
                        timing=Timing.CONTEMPORANEOUS,  # baseline cannot distinguish
                        is_templated=False,             # baseline cannot detect
                        claim=str(a.get("title", ""))[:300],
                        reasoning="matched keyword query",
                        confidence=1.0,
                    ),
                    url=str(a.get("url", "")),
                    domain=str(a.get("domain", "")),
                    title=str(a.get("title", "")),
                    seendate=str(a.get("seendate", "")),
                    extractor=self.name,
                    model="none",
                )
            )
        return out


# --------------------------------------------------------------------------------------
# Claude
# --------------------------------------------------------------------------------------


def _api_schema() -> dict[str, Any]:
    """JSON Schema for the API's structured-output constraint.

    Structured outputs reject numeric and length constraints, so `ge`/`le`/`max_length`
    from the Pydantic model are stripped before sending and re-validated client-side when
    the response is parsed back into `EvidenceRecord`.
    """
    schema = EvidenceRecord.model_json_schema()

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: strip(v)
                for k, v in node.items()
                if k not in {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                             "minLength", "maxLength", "multipleOf", "default"}
            }
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    schema = strip(schema)
    schema["additionalProperties"] = False
    for defn in schema.get("$defs", {}).values():
        if isinstance(defn, dict) and defn.get("type") == "object":
            defn["additionalProperties"] = False
    return schema


class MissingCredentials(RuntimeError):
    """No Anthropic credentials are available."""


@dataclass
class ClaudeExtractor:
    """Schema-constrained relevance judgment and evidence extraction."""

    model: str = DEFAULT_MODEL
    max_tokens: int = 1024
    name: str = "claude"

    def _client(self):
        import anthropic

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # The SDK also resolves an `ant auth login` profile, so this is a warning
            # path rather than a hard precondition -- let the SDK try, and translate its
            # auth failure into something actionable.
            pass
        try:
            return anthropic.Anthropic()
        except Exception as exc:  # noqa: BLE001
            raise MissingCredentials(
                "Could not construct an Anthropic client. Set ANTHROPIC_API_KEY, or run "
                "`ant auth login` to store a profile the SDK reads automatically."
            ) from exc

    def _messages(self, article: pd.Series) -> list[dict]:
        return [
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    domain=article.get("domain", ""),
                    seendate=article.get("seendate", ""),
                    title=article.get("title", ""),
                ),
            }
        ]

    # -- synchronous, one article at a time (small samples, interactive use) ------------

    def extract(self, articles: pd.DataFrame) -> list[ExtractionResult]:
        """Judge each article with a separate request. Use `extract_batch` for corpora."""
        import anthropic

        client = self._client()
        out: list[ExtractionResult] = []

        for _, a in articles.iterrows():
            try:
                resp = client.messages.parse(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=self._messages(a),
                    output_format=EvidenceRecord,
                )
                record = resp.parsed_output
            except anthropic.APIStatusError:
                continue
            out.append(
                ExtractionResult(
                    record=record,
                    url=str(a.get("url", "")),
                    domain=str(a.get("domain", "")),
                    title=str(a.get("title", "")),
                    seendate=str(a.get("seendate", "")),
                    extractor=self.name,
                    model=self.model,
                )
            )
        return out

    # -- batch (50% cost, the right mode for a corpus) ---------------------------------

    def submit_batch(self, articles: pd.DataFrame) -> str:
        """Submit the corpus as a Message Batch; returns the batch id."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        client = self._client()
        schema = _api_schema()

        requests = [
            Request(
                custom_id=f"art-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=self._messages(a),
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                ),
            )
            for i, (_, a) in enumerate(articles.iterrows())
        ]
        return client.messages.batches.create(requests=requests).id

    def collect_batch(
        self, batch_id: str, articles: pd.DataFrame, *, poll_seconds: int = 30
    ) -> list[ExtractionResult]:
        """Poll a batch to completion and join results back to their articles.

        Results arrive in arbitrary order, so they are keyed by ``custom_id`` and never by
        position.
        """
        client = self._client()

        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            time.sleep(poll_seconds)

        rows = {i: a for i, (_, a) in enumerate(articles.iterrows())}
        out: list[ExtractionResult] = []

        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                continue
            idx = int(result.custom_id.split("-")[1])
            a = rows.get(idx)
            if a is None:
                continue
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"), None
            )
            if not text:
                continue
            try:
                record = EvidenceRecord.model_validate(json.loads(text))
            except Exception:  # noqa: BLE001
                continue
            out.append(
                ExtractionResult(
                    record=record,
                    url=str(a.get("url", "")),
                    domain=str(a.get("domain", "")),
                    title=str(a.get("title", "")),
                    seendate=str(a.get("seendate", "")),
                    extractor=f"{self.name}-batch",
                    model=self.model,
                )
            )
        return out


# --------------------------------------------------------------------------------------
# Aggregation — deterministic, no LLM
# --------------------------------------------------------------------------------------


def to_frame(results: list[ExtractionResult]) -> pd.DataFrame:
    """Flatten extraction results into a table."""
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        row = r.record.model_dump()
        row.update(
            {
                "url": r.url,
                "domain": r.domain,
                "title": r.title,
                "seendate": r.seendate,
                "extractor": r.extractor,
                "model": r.model,
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    return df


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate judged evidence into daily features. Pure code -- no model involved.

    Counts *prospective relevant* evidence separately from the total. Given that Sept 2019
    coverage was overwhelmingly retrospective, a feature that does not make that split is
    measuring news volume about the past, not risk about the future.
    """
    if not len(df):
        return pd.DataFrame()

    d = df.copy()
    d["date"] = d["seendate"].dt.tz_convert("America/New_York").dt.normalize()
    rel = d[d["is_relevant"] & ~d["is_templated"]]

    out = pd.DataFrame(
        {
            "n_articles": d.groupby("date").size(),
            "n_relevant": rel.groupby("date").size(),
            "n_prospective": rel[rel["timing"] == "prospective"].groupby("date").size(),
            "n_retrospective": rel[rel["timing"] == "retrospective"].groupby("date").size(),
            "n_risk_increasing": rel[rel["direction"] == "increases_risk"]
            .groupby("date")
            .size(),
        }
    ).fillna(0)
    out["relevant_share"] = out["n_relevant"] / out["n_articles"].replace(0, pd.NA)
    return out.sort_index()
