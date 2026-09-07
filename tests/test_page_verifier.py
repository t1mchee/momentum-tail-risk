"""The verifier must REFUSE. A verifier never seen to refuse is an assumption (trp-70)."""

from __future__ import annotations

import copy

import pytest

from unstructured_momentum.pipeline.page import Composer, render
from unstructured_momentum.pipeline.verify_page import verify


@pytest.fixture(scope="module")
def good():
    return Composer().compose("2019-09-30").as_dict()


def test_a_real_page_is_accepted(good):
    v = verify(good)
    assert v.accepted, v.reasons
    assert v.checked["numeric_fields"] > 10
    assert v.checked["could_not_measure_entries"] >= 4
    assert v.checked["falsifier_checkable"]


def test_refuses_a_number_whose_provenance_was_stripped(good):
    bad = copy.deepcopy(good)
    bad["sections"]["severity"]["horizons"]["1m"]["q05"]["provenance"] = {}
    v = verify(bad)
    assert not v.accepted
    assert any("provenance" in r for r in v.reasons)


def test_refuses_a_falsifier_missing_its_threshold(good):
    bad = copy.deepcopy(good)
    bad["falsifier"]["threshold"] = None
    v = verify(bad)
    assert not v.accepted
    assert any("machine-checkable" in r for r in v.reasons)


def test_refuses_a_page_with_no_could_not_measure_section(good):
    bad = copy.deepcopy(good)
    bad["could_not_measure"] = []
    v = verify(bad)
    assert not v.accepted


def test_refuses_an_altered_quote(good):
    """Built as a fixture rather than borrowed from a real page.

    Most retrieved analogues predate the 2018 corpus and carry no comparison, so a test that
    depended on finding one would skip silently on most dates -- and a check that skips is a
    check nobody is running.
    """
    bad = copy.deepcopy(good)
    quote = "entered into a material definitive agreement with respect to the facility"
    bad["sections"]["analogues"]["matches"][0]["narrative_comparison"] = {
        "shared_conditions": [], "what_followed_then": [],
        "disanalogies": [{"block": "QUERY", "claim": "a financing arrangement",
                          "quote": quote}],
        "hindsight_disclosure": "x"}
    d = bad["sections"]["analogues"]["matches"][0]["date"]
    src = {f"{bad['as_of']}|{d}|QUERY": f"On that date the issuer {quote} and filed it."}
    assert verify(bad, block_texts=src).accepted

    bad["sections"]["analogues"]["matches"][0]["narrative_comparison"]["disanalogies"][0][
        "quote"] = "categorically fabricated text appearing in no filing anywhere"
    v = verify(bad, block_texts=src)
    assert not v.accepted
    assert any("did not verify" in r for r in v.reasons)


def test_render_is_deterministic(good):
    p = Composer().compose("2019-09-30")
    assert render(p) == render(p)
