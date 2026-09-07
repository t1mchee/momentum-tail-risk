"""Retrieval as a measurement procedure, not as citation garnish.

The pipeline this serves is: a candidate driver is the QUERY, retrieval pulls each holding's
relevant passages, extraction turns them into structured records, and arithmetic aggregates them
to a book-level exposure profile. Every component of that except retrieval already existed, and
retrieval had never been built at all -- `config.CORPUS`'s "retrieval index" comment describes an
empty directory.

Design choices and why
----------------------
**Lexical first.** BM25 needs no GPU, no vintage weights, and no embedding cache that a
single added document invalidates. It is also trivially point-in-time, which matters more here
than semantic recall: a dense arm is a later rung, added only if recall is MEASURED insufficient
rather than assumed to be.

**`as_of` is a hard filter, not a re-rank.** A passage from a filing accepted after the
measurement date is not eligible, full stop. Down-weighting it would leave a leak whose size
depends on the weighting.

**Recall is measured before anything downstream is believed.** Coverage becomes a function of
retrieval recall, so an unmeasured recall silently caps every result built on it. The 293
surviving extraction records make this free: each carries a sentence QUOTED from a known filing,
so each is a known-relevant passage with a known home, and recall@k is computable with no
hand-marking at all.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

_WORD = re.compile(r"[a-z][a-z0-9'\-]+")
#: Ordinary filing furniture. Dropping it costs nothing and stops every query matching the
#: boilerplate every document shares.
_STOP = frozenset("""the a an and or of to in for on at by with from as is are was were be been
being that this these those it its our we us their they he she his her which who whom what when
where how all any both each few more most other some such no nor not only own same so than too
very can will just should now may might must would could have has had do does did""".split())


def tokenise(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


@dataclass
class Chunk:
    doc_id: str
    ticker: str
    section: str
    ordinal: int
    text: str
    available_at: pd.Timestamp
    tokens: list[str] = field(default_factory=list)


def chunk(text: str, *, target_tokens: int = 512, overlap: int = 64) -> list[str]:
    """Split on sentence boundaries, packing to a target size with overlap.

    Sentence-aware because a passage cut mid-sentence cannot be quoted as evidence, and the
    extraction atom's quote gate would then reject its own retrieval. Overlap exists so a
    sensitivity statement spanning a boundary survives in at least one chunk whole.
    """
    parts = re.split(r"(?<=[.;])\s+", re.sub(r"\s+", " ", text).strip())
    out, buf, n = [], [], 0
    for p in parts:
        w = len(p.split())
        if n + w > target_tokens and buf:
            out.append(" ".join(buf))
            keep, kn = [], 0
            for s in reversed(buf):
                kn += len(s.split())
                keep.insert(0, s)
                if kn >= overlap:
                    break
            buf, n = keep, kn
        buf.append(p)
        n += w
    if buf:
        out.append(" ".join(buf))
    return [c for c in out if len(c.split()) > 20]


@dataclass
class Index:
    """A BM25 index over chunks, with an availability stamp on every one."""

    chunks: list[Chunk] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    avgdl: float = 0.0
    k1: float = 1.5
    b: float = 0.75

    def add(self, c: Chunk) -> None:
        c.tokens = tokenise(c.text)
        self.chunks.append(c)
        for t in set(c.tokens):
            self.df[t] += 1

    def finalise(self) -> "Index":
        if self.chunks:
            self.avgdl = sum(len(c.tokens) for c in self.chunks) / len(self.chunks)
        return self

    def _score(self, q: list[str], c: Chunk) -> float:
        N, tf = len(self.chunks), Counter(c.tokens)
        dl = len(c.tokens) or 1
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def query(self, q: str, *, k: int = 8, as_of: pd.Timestamp | str | None = None,
              ticker: str | None = None) -> list[tuple[float, Chunk]]:
        """Top-k chunks. `as_of` excludes anything not yet filed -- a filter, never a re-rank."""
        toks = tokenise(q)
        pool = self.chunks
        if as_of is not None:
            cut = pd.Timestamp(as_of)
            pool = [c for c in pool if c.available_at <= cut]
        if ticker is not None:
            pool = [c for c in pool if c.ticker == ticker]
        scored = [(self._score(toks, c), c) for c in pool]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def build_index(records, *, target_tokens: int = 512) -> Index:
    """Records are dicts with ticker, accession, section, text and an availability timestamp."""
    idx = Index()
    for r in records:
        for i, body in enumerate(chunk(r["text"], target_tokens=target_tokens)):
            idx.add(Chunk(doc_id=str(r.get("accession", r["ticker"])), ticker=r["ticker"],
                          section=r.get("section", "item_7a"), ordinal=i, text=body,
                          available_at=pd.Timestamp(r["available_at"])))
    return idx.finalise()


def recall_at_k(idx: Index, cases: list[tuple[str, str]], query: str, *, k: int = 8) -> dict:
    """Fraction of known-relevant passages retrieved in the top k, per ticker.

    `cases` are (ticker, quoted_sentence) pairs. The 293 surviving extraction records supply
    them for free: each carries a sentence quoted verbatim from a known filing, so relevance is
    established without anyone hand-marking a thing.
    """
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()  # noqa: E731
    hits, tried, missing = 0, 0, []
    for ticker, quote in cases:
        q = norm(quote)
        if not q or q == "none" or len(q) < 30:
            continue
        tried += 1
        got = idx.query(query, k=k, ticker=ticker)
        # Cached quotes elide with "...", e.g. "hypothetical 10% increase ... would change our
        # annualized interest expense". Matching a fixed prefix straight through the ellipsis
        # can never succeed, and produced a hard recall plateau at 88 percent that looked like a
        # retrieval ceiling and was a measurement bug. Each fragment is checked separately, the
        # same way the extraction atom's quote gate handles it.
        frags = [f.strip() for f in q.split("...") if len(f.strip()) >= 24] or [q[:80]]
        if any(all(f[:80] in norm(c.text) for f in frags) for _, c in got):
            hits += 1
        else:
            missing.append(ticker)
    return {"n": tried, "hits": hits, "recall": hits / tried if tried else float("nan"),
            "k": k, "missed": missing[:10]}
