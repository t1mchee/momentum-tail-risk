"""Tests for the rebuilt exposure atom.

The gates that matter are ground truth by construction -- a placebo whose redaction is verified,
and a positive control whose ordering is known before the numbers are computed. Agreement with
the 293 surviving cached records is deliberately NOT a gate: they are one model's output from a
prompt nobody has, and tuning until a new prompt agrees with them would recover agreement rather
than the instrument.
"""

from __future__ import annotations

from unstructured_momentum.llm import exposure as X


FILLER = ("The company operates three segments across North America and Europe. " * 30)


def test_redaction_removes_every_term_and_counts_survivors():
    t = (FILLER + "Our variable-rate debt of $500 million means a 100 basis point increase in "
         "interest rates would raise annual interest expense by $5.0 million. " + FILLER)
    out, survivors = X.redact_rate_language(t)
    assert survivors == 0, f"{survivors} rate terms survived redaction"
    assert "three segments" in out, "redaction removed unrelated content"


def test_redaction_leaves_enough_document_to_read():
    """A placebo on an emptied document tests nothing.

    The word-window removes a band around each hit, so on a short dense passage it can remove
    everything -- which is correct behaviour and a useless test. On the real filings it retains
    a median of 51 percent (24 to 81), so the model has plenty of text and simply no rate
    language. That is the claim the placebo is entitled to make.
    """
    t = FILLER + "A 100 basis point increase in interest rates raises expense. " + FILLER
    out, _ = X.redact_rate_language(t)
    assert len(out) / len(t) > 0.5, "redaction left too little of the document to be a fair test"


def test_bare_interest_is_redacted():
    """A filing quoted 'market value changes caused by interest fluctuations' after every
    'interest rate' had gone, and that leak defeated the placebo for one company."""
    out, survivors = X.redact_rate_language("Changes caused by interest fluctuations.")
    assert survivors == 0


def test_quote_gate_rejects_fabricated_evidence():
    src = "We have a $500 million revolving credit facility maturing in 2024."
    assert X.quote_is_grounded("We have a $500 million revolving credit facility", src)
    assert not X.quote_is_grounded("Our exposure to commodity prices is material.", src)


def test_quote_gate_allows_none():
    assert X.quote_is_grounded("none", "anything")


def test_evidence_must_state_a_rate_exposure_not_merely_debt():
    """The atom's real failure mode, caught by the placebo: the model inferred rate sensitivity
    from sentences about borrowings after every rate term had been removed."""
    assert not X.evidence_states_rate_exposure("We have a $500 million revolving credit facility.")
    assert X.evidence_states_rate_exposure(
        "A 100 basis point increase would raise interest expense by $5 million.")


def test_record_schema_matches_the_surviving_cache():
    old = X.legacy_records("real")
    assert len(old) == 120
    sample = next(iter(old.values()))
    assert set(sample) == {"exposure", "magnitude", "channel", "evidence"}
    assert set(X.ExposureRecord.model_fields) == set(sample)


def test_both_placebo_strata_survive_as_a_two_sided_target():
    """The failed sentence-boundary attempt is as useful as the one that worked."""
    first, second = X.legacy_records("placebo"), X.legacy_records("placebo2")
    assert sum(1 for v in first.values() if v["exposure"] == "none") == 9
    assert sum(1 for v in second.values() if v["exposure"] == "none") == 20
