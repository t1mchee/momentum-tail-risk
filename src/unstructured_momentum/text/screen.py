"""screen-v2: the operative funnel, versioned so no record can ambiguate about which one ran.

A screen is not a taxonomy. The taxonomy says which terms belong to a driver; the screen says how
many of them a document must carry before it is worth reading. Those change independently and a
record needs both to be reproducible, so the screen carries its own identifier alongside the
taxonomy hash in every extraction record.

screen-v1 was implicit and required ONE term. Measured, it selected 94 percent of the corpus with
4.84 of 8 drivers firing on the average filing, unique shares near zero, and a projected 700M
input tokens. That is a funnel that does no funnelling.

screen-v2 requires a FRACTION of the driver's own vocabulary rather than an absolute count,
because vocabularies run 55 to 125 terms and an absolute threshold silently demands twice as much
of a small vocabulary as of a large one. At ten percent the screen selects 59 percent of the
corpus at 1.53 drivers per document and 221M tokens -- a 3.2-fold reduction that costs no reseed
and re-hashes nothing.

The threshold is chosen from the measured table, which is licensed hindsight of the same kind as
choosing a horizon: it is an operational cost parameter, not a verdict-bearing statistic, it is
disclosed here rather than buried, and what the tightening MISSES is measured by the
false-negative design rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VERSION = "screen-v2"
#: Fraction of a driver's own vocabulary a document must carry. From the D2 sweep: 0.05 leaves
#: 2.79 drivers per document, 0.10 gives 1.53, 0.15 gives 0.86 and starts losing coverage.
VOCAB_FRACTION = 0.10
#: Never demand fewer than this, or a small vocabulary would be selected by a single term.
MIN_TERMS = 2


@dataclass(frozen=True)
class Screen:
    version: str = VERSION
    vocab_fraction: float = VOCAB_FRACTION
    min_terms: int = MIN_TERMS

    def threshold_for(self, vocab_size: int) -> int:
        return max(self.min_terms, int(round(self.vocab_fraction * vocab_size)))

    def stamp(self) -> dict:
        """What an extraction record carries so its funnel is recoverable."""
        return {"screen_version": self.version, "vocab_fraction": self.vocab_fraction,
                "min_terms": self.min_terms}


def compile_driver(terms: list[str]) -> re.Pattern:
    """One alternation per driver, longest term first so a phrase wins over its own first word."""
    ts = sorted({t.lower() for t in terms}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in ts) + r")\b")


def distinct_hits(pattern: re.Pattern, text: str) -> set[str]:
    return set(pattern.findall(text.lower()))


def select(text: str, vocab: dict[str, list[str]], screen: Screen | None = None) -> dict[str, int]:
    """Distinct-term hit count per driver, and which drivers clear their own threshold.

    Returns counts rather than booleans on purpose: the count is what lets a later threshold be
    evaluated without re-reading the corpus, and re-reading the corpus is what made choosing this
    threshold expensive the first time.
    """
    sc = screen or Screen()
    out = {}
    for d, terms in vocab.items():
        n = len(distinct_hits(compile_driver(terms), text))
        out[d] = n
    return out


def clears(counts: dict[str, int], vocab: dict[str, list[str]],
           screen: Screen | None = None) -> dict[str, bool]:
    sc = screen or Screen()
    return {d: counts.get(d, 0) >= sc.threshold_for(len(vocab[d])) for d in vocab}
