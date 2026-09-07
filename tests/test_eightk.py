"""Tests for the 8-K body loader.

The failure this guards against is silent: an 8-K's primary document is registrant boilerplate
wrapped in inline XBRL, and the material event is in a 99-series exhibit. Fetching only the
primary would have built a corpus of cover sheets that parse cleanly and say nothing.
"""

from __future__ import annotations

from unstructured_momentum.data import eightk


def test_earnings_only_filings_are_excluded_by_default():
    """Item 2.02 is already consumed for its timestamp by the earnings calendar."""
    assert eightk.EARNINGS_ITEM == "2.02"
    assert "2.02" not in eightk.MATERIAL_ITEMS


def test_material_items_cover_the_events_worth_reading():
    for code in ("2.05", "4.02", "5.02", "7.01", "8.01"):
        assert code in eightk.MATERIAL_ITEMS, f"{code} missing from the material set"


def test_exhibit_pattern_matches_the_99_series_only():
    m = eightk._EXHIBIT
    for good in ("d62084dex991.htm", "ex-99.1.htm", "EX99_2.HTM", "a8kex991.htm"):
        assert m.search(good), f"did not match {good}"
    for bad in ("d62084d8k.htm", "ex-101.ins.xml", "FilingSummary.xml"):
        assert not m.search(bad), f"wrongly matched {bad}"


def test_prose_start_skips_the_xbrl_preamble():
    """Stripped inline XBRL leaves a run of identifiers and repeated dates before the prose."""
    raw = ("8-K MICROSOFT CORP 2020-09-21 false 0000789019 0000789019 2020-09-21 "
           "FORM 8-K CURRENT REPORT Pursuant to Section 13")
    m = eightk._PROSE_START.search(raw)
    assert m and raw[m.start():].startswith("FORM 8-K")


def test_rate_limit_matches_the_filing_loader():
    """Both hit the same host; together they must stay under the SEC's ten per second."""
    from unstructured_momentum.data import filings
    assert eightk.RATE_LIMIT <= filings.RATE_LIMIT
