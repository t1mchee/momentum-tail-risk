"""Tests for retrieval-as-measurement.

The point-in-time filter is the one that would fail silently and matter most, so it is tested
directly rather than through a recall number.
"""

from __future__ import annotations

import pandas as pd

from unstructured_momentum.text import retrieve as R

DOCS = [
    {"ticker": "AAA", "accession": "a1", "section": "item_7a", "available_at": "2019-02-01",
     "text": ("Our variable-rate debt of $500 million exposes us to interest rate risk. "
              "A 100 basis point increase in interest rates would raise annual interest "
              "expense by $5.0 million. " * 6)},
    {"ticker": "BBB", "accession": "b1", "section": "item_7a", "available_at": "2020-02-01",
     "text": ("We lease our headquarters and operate three segments. Employee training "
              "programs are reviewed annually. " * 8)},
]


def _idx():
    return R.build_index(DOCS, target_tokens=64)


def test_chunks_end_on_sentence_boundaries():
    """A passage cut mid-sentence cannot be quoted, and the atom's quote gate would reject it."""
    for c in R.chunk(DOCS[0]["text"], target_tokens=40, overlap=10):
        assert c.rstrip().endswith((".", ";")), f"chunk does not end on a boundary: {c[-40:]!r}"


def test_as_of_excludes_filings_not_yet_accepted():
    """A hard filter, never a re-rank: a later filing must be invisible, not down-weighted."""
    idx = _idx()
    early = idx.query("interest rate risk", k=10, as_of="2019-06-01")
    assert early, "expected the 2019 filing to be retrievable"
    assert all(c.ticker != "BBB" for _, c in early), "a 2020 filing leaked into a 2019 query"
    later = idx.query("interest rate risk", k=10, as_of="2021-01-01")
    assert len(later) >= len(early)


def test_as_of_boundary_is_inclusive_of_the_acceptance_date():
    idx = _idx()
    got = idx.query("interest rate risk", k=10, as_of=pd.Timestamp("2019-02-01"))
    assert any(c.ticker == "AAA" for _, c in got)


def test_query_ranks_the_relevant_document_first():
    idx = _idx()
    got = idx.query("interest rate basis point increase", k=3)
    assert got and got[0][1].ticker == "AAA"


def test_recall_handles_elided_quotes():
    """Cached quotes elide with '...'. Matching straight through the ellipsis can never succeed
    and produced a hard recall plateau that looked like a retrieval ceiling."""
    idx = _idx()
    quote = "Our variable-rate debt of $500 million ... would raise annual interest expense"
    r = R.recall_at_k(idx, [("AAA", quote)], "interest rate risk", k=5)
    assert r["recall"] == 1.0, f"elided quote not matched: {r}"


def test_recall_reports_n_so_a_thin_measurement_is_visible():
    idx = _idx()
    r = R.recall_at_k(idx, [("AAA", "x" * 10)], "interest", k=3)
    assert r["n"] == 0, "a too-short quote should be skipped, not counted as a miss"
