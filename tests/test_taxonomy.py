"""Self-tests for the driver taxonomy, on a planted mini-corpus.

The expansion is a scoring rule and a scoring rule can be tested against documents whose answer
is known. These plant a driver's vocabulary into a subset of synthetic filings and check the
expansion finds it -- which is the same planted-effect discipline the power certificates use,
applied one layer down.

They also cover the two failures this module has already had: a storage layer that sorted terms
alphabetically and destroyed the lift ranking, and a candidate filter whose ceiling sat below its
floor so every driver returned its seeds unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from unstructured_momentum.text import taxonomy as tx

FILLER = ("the company operates facilities and employs staff across several regions "
          "and reports results quarterly to shareholders under applicable regulations ")


def _corpus(n: int = 400, planted: int = 80, seed: int = 7) -> tuple[list[str], set[str]]:
    """Documents where a known vocabulary appears only in the planted subset."""
    rng = np.random.default_rng(seed)
    marker = {"libor", "maturities", "swaps", "amortization"}
    extra = {"debenture", "covenantlite", "tranching"}
    docs = []
    for i in range(n):
        body = FILLER * 3 + " ".join(rng.choice(
            ["revenue", "customers", "products", "markets", "segments"], 40))
        if i < planted:
            body += " " + " ".join(sorted(marker | extra)) * 4
        docs.append(body)
    return docs, extra


def test_expansion_recovers_a_planted_vocabulary():
    docs, extra = _corpus()
    out = tx.expand(docs, seeds={"rates": ("libor", "maturities")}, min_df_frac=0.01,
                    min_df_floor=5)
    found = set(out["rates"])
    assert extra <= found, f"planted terms not recovered: {sorted(extra - found)}"


def test_expansion_ranks_by_lift_not_alphabetically():
    """The failure that made a good expansion look like noise on inspection."""
    docs, extra = _corpus()
    out = tx.expand(docs, seeds={"rates": ("libor", "maturities")}, min_df_frac=0.01,
                    min_df_floor=5)
    terms = out["rates"]
    beyond_seeds = [t for t in terms if t not in {"libor", "maturities"}]
    assert beyond_seeds, "expansion added nothing"
    assert beyond_seeds != sorted(beyond_seeds), (
        "terms are in alphabetical order; the lift ranking has been discarded")


def test_an_empty_candidate_filter_raises_rather_than_returning_seeds():
    """The v2 failure: a ceiling below the floor emptied the filter and seven snapshots were
    written containing nothing but their seeds."""
    docs, _ = _corpus()
    with pytest.raises(ValueError):
        tx.expand(docs, seeds={"rates": ("libor",)}, min_df_frac=0.99, min_df_floor=395)


def test_a_snapshot_is_immutable_once_written(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "STORE", tmp_path)
    a = tx.Snapshot(version="t", vintage="2019", drivers={"d": ["x", "y"]}, n_documents=10)
    a.write()
    b = tx.Snapshot(version="t", vintage="2019", drivers={"d": ["x", "z"]}, n_documents=10)
    with pytest.raises(ValueError, match="immutable"):
        b.write()


def test_content_hash_ignores_order_but_not_content():
    a = tx.Snapshot(version="t", vintage="2019", drivers={"d": ["x", "y"]}, n_documents=10)
    b = tx.Snapshot(version="t", vintage="2019", drivers={"d": ["y", "x"]}, n_documents=10)
    c = tx.Snapshot(version="t", vintage="2019", drivers={"d": ["x", "z"]}, n_documents=10)
    assert a.content_hash() == b.content_hash(), "reordering must not change identity"
    assert a.content_hash() != c.content_hash(), "a content change must change identity"


def test_seed_audit_flags_a_seed_that_matches_most_of_the_corpus():
    """Selectivity: a seed firing on three quarters of filings cannot discriminate."""
    docs = ["alpha " + FILLER for _ in range(80)] + ["beta " + FILLER for _ in range(20)]
    a = tx.seed_coherence(docs, {"d": ("alpha", "beta")}, bigrams=False)
    row = a[a.seed == "alpha"].iloc[0]
    assert row.doc_frac > 0.5 and bool(row.too_common) and bool(row.suspect)
    assert not bool(a[a.seed == "beta"].iloc[0].too_common)


def test_seed_audit_flags_a_seed_that_keeps_no_company():
    """Coherence: a seed whose documents contain none of its siblings is selecting elsewhere."""
    docs = ([f"gamma delta {FILLER}" for _ in range(40)]
            + [f"epsilon {FILLER}" for _ in range(40)])
    a = tx.seed_coherence(docs, {"d": ("gamma", "delta", "epsilon")}, bigrams=False)
    lone = a[a.seed == "epsilon"].iloc[0]
    assert lone.coherence == 0.0, "a seed with no siblings present should score zero coherence"
    assert bool(lone.incoherent_for_its_driver)


# ---------------------------------------------------------------- screen-v2

def test_screen_threshold_scales_with_vocabulary_size():
    """An absolute count demands twice as much of a 55-term driver as of a 125-term one, which
    is why the measured screen used a fraction instead."""
    from unstructured_momentum.text import screen as sc
    s = sc.Screen()
    assert s.threshold_for(55) < s.threshold_for(125)
    assert s.threshold_for(125) == round(0.10 * 125)


def test_screen_never_selects_on_a_single_term():
    """A small vocabulary under a pure fraction would be selected by one hit, which is the
    behaviour screen-v1 had and screen-v2 exists to remove."""
    from unstructured_momentum.text import screen as sc
    s = sc.Screen()
    assert s.threshold_for(5) >= 2
    assert s.threshold_for(1) >= 2


def test_screen_stamp_identifies_the_funnel():
    """A record carries the screen alongside the taxonomy hash, because the two change
    independently and either alone leaves the funnel ambiguous."""
    from unstructured_momentum.text import screen as sc
    st = sc.Screen().stamp()
    assert st["screen_version"] == "screen-v2"
    assert set(st) == {"screen_version", "vocab_fraction", "min_terms"}


def test_screen_counts_distinct_terms_not_occurrences():
    """Ten mentions of one term is one term. Counting occurrences would let a single repeated
    phrase clear a threshold meant to require breadth."""
    from unstructured_momentum.text import screen as sc
    vocab = {"d": ["alpha beta", "gamma", "delta"]}
    counts = sc.select("alpha beta " * 10, vocab)
    assert counts["d"] == 1
    counts2 = sc.select("alpha beta gamma delta", vocab)
    assert counts2["d"] == 3
