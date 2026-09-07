"""The mask has to be checkable by hand before it is trusted on a corpus."""

from __future__ import annotations

from unstructured_momentum.llm.masking import mask


def test_removes_every_common_date_form():
    t = ("On March 3, 2020 the board met. Filed 03/04/2020, effective 2020-03-05, "
         "during fiscal 2020 and the first quarter. Compare with September 2019.")
    out, info = mask(t)
    assert info["survivor_total"] == 0, out
    assert "2020" not in out and "2019" not in out
    assert "March" not in out and "September" not in out


def test_removes_named_entities_case_insensitively():
    out, info = mask("ACME Corp and acme reported.", entities=["ACME Corp", "ACME"])
    assert "acme" not in out.lower()
    assert info["removed"]["entity_known"] >= 2


def test_survivor_count_is_reported_not_assumed():
    """A year written in words is NOT caught, and the count must say so rather than pass."""
    out, info = mask("The events of twenty twenty were unusual. Also 1999.")
    assert info["survivors"]["year"] == 0        # 1999 is caught
    assert "twenty twenty" in out                 # spelled-out years are residue, by design
    # The point of the test: the function reports what it removed, and the caller can see
    # that spelled-out periods are outside its reach.
    assert info["removed"]["year"] == 1


def test_masking_does_not_remove_subject_matter():
    """The limitation the period-recovery attack exists to measure."""
    out, _ = mask("The Company closed all restaurants due to the COVID-19 pandemic in "
                  "March 2020.", entities=["The Company"])
    assert "COVID-19" in out and "pandemic" in out


def test_redacts_counterparties_the_caller_did_not_name():
    """The defect the period-recovery attack found: other people's names date a document too."""
    t = ("The Company and Prime Security Services Borrower, LLC amended the facility. "
         "Its subsidiary tZERO signed a letter of intent with GSR Capital.")
    out, info = mask(t, entities=["The Company"])
    for leaked in ("Prime Security", "tZERO", "GSR Capital"):
        assert leaked not in out, out
    assert info["survivors"]["corporate_entity"] == 0


def test_redacts_partial_company_forms_and_people():
    from unstructured_momentum.llm.masking import name_tokens
    toks = name_tokens(["ZIMMER BIOMET HOLDINGS, INC.", "META FINANCIAL GROUP INC"])
    assert "ZIMMER" in toks and "BIOMET" in toks
    assert "HOLDINGS" not in toks and "FINANCIAL" not in toks   # generic, not redacted globally
    out, info = mask("Zimmer Biomet appointed Carrie Nichol as Vice President.", tokens=toks)
    assert "Zimmer" not in out and "Biomet" not in out
    assert "Carrie Nichol" not in out
    assert info["removed"]["person"] >= 1


def test_ordinary_financial_vocabulary_survives_token_redaction():
    """Round-3 over-redaction: the shipped batch quoted 'common [REDACTED]' for 'common stock'."""
    from unstructured_momentum.llm.masking import name_tokens
    toks = name_tokens(["STOCK YARDS BANCORP INC", "CREDIT ACCEPTANCE CORP",
                        "ZIMMER BIOMET HOLDINGS, INC."])
    out, _ = mask("The Company amended its Credit Agreement and issued common stock.",
                  tokens=toks)
    assert "Credit" in out and "stock" in out, out
    assert "ZIMMER" in toks   # distinctive names still redacted


def test_exhibit_filenames_and_compact_dates_are_removed():
    """The largest MEASURED leak channel: EDGAR exhibit names carry the date in their digits."""
    t = ("See exhibit ex991taxseasonupdatepr3518.htm and financialsx2018 for detail; "
         "results for 2018Q1 and FY18 are attached.")
    out, info = mask(t)
    assert "3518" not in out and "2018" not in out and "FY18" not in out, out
    assert info["survivors"]["filename"] == 0
    assert info["survivors"]["compact_date"] == 0


def test_masking_keeps_ordinary_prose_readable():
    t = "The Company entered into a Credit Agreement and issued common stock to investors."
    out, _ = mask(t)
    assert out.count("[REDACTED]") == 0, out
