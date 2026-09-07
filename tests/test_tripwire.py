"""Tests for the event-tripwire loop's deterministic layer.

Everything the trip decision rests on — normalisation, the quote gate, the distinct-filer
count, the append-once ledgers — is plain code, so it is testable offline. The judge's
schema is exercised only structurally (no network in tests).
"""

from __future__ import annotations

import json

import pandas as pd

from unstructured_momentum.llm import translate
from unstructured_momentum.text import tripwire as tw


def test_normalise_drops_unknown_items_and_reports_it():
    rec, problems = tw.normalise_recipe({
        "item_codes": ["7.01", "9.99", "2.06", "7.01"],
        "anchor_terms": ["Withdraw GUIDANCE", "impairment", "impairment"],
        "min_distinct_filers": 0, "horizon_days": 90, "refutes": "x"})
    assert rec["item_codes"] == ["2.06", "7.01"]
    assert rec["anchor_terms"] == ["withdraw guidance", "impairment"]
    assert rec["min_distinct_filers"] == 1 and rec["horizon_days"] == 63
    assert any("9.99" in p for p in problems)


def test_quote_gate_is_exact_substring():
    body = "The Company recorded a non-cash impairment charge of $1.2 billion."
    assert tw.quote_gate("impairment charge of $1.2 billion", body)
    assert not tw.quote_gate("impairment charge of $1.2bn", body)   # paraphrase
    assert not tw.quote_gate("impairment  charge", body)            # whitespace drift
    assert not tw.quote_gate("", body)                              # empty never passes


def test_excerpts_merge_overlapping_windows_and_keep_verbatim_text():
    body = "A" * 100 + " withdraw guidance " + "B" * 100
    ex = tw._excerpts(body, ["withdraw guidance"], width=50)
    assert "withdraw guidance" in ex
    assert ex in body  # excerpt windows are verbatim slices


def test_append_once_is_idempotent(tmp_path):
    p = tmp_path / "ledger.jsonl"
    assert tw._append_once(p, {"a": 1}, "k1")
    assert not tw._append_once(p, {"a": 1}, "k1")
    assert tw._append_once(p, {"a": 2}, "k2")
    assert len(p.read_text().strip().splitlines()) == 2


def _rec(**kw):
    base = {"as_of": "2020-10-31", "driver": "d", "tripwire_id": "t1",
            "members": ["AAA", "BBB", "CCC"], "item_codes": ["7.01"],
            "anchor_terms": ["guidance"], "min_distinct_filers": 2,
            "horizon_days": 21, "refutes": "r", "status_at_arming": "armed"}
    return {**base, **kw}


def _hit(ticker, accepted, **kw):
    base = {"tripwire_id": "t1", "ticker": ticker, "accession": f"x-{ticker}-{accepted}",
            "accepted_at": accepted, "is_evidence": True, "bears_on": "refutes",
            "quote_gate": "passed"}
    return {**base, **kw}


def _idx(rows):
    return pd.DataFrame(rows, columns=["ticker", "accession", "filing_date",
                                       "accepted_at", "items"])


def test_status_trips_only_on_distinct_gated_refuting_filers():
    idx = _idx([("AAA", "a1", "2020-11-05", "2020-11-05T12:00:00+00:00", "7.01"),
                ("AAA", "a2", "2020-11-06", "2020-11-06T12:00:00+00:00", "7.01"),
                ("BBB", "b1", "2020-11-09", "2020-11-09T12:00:00+00:00", "7.01")])
    # two filings, ONE distinct filer: not tripped
    hits = [_hit("AAA", "2020-11-05T12:00:00+00:00"),
            _hit("AAA", "2020-11-06T12:00:00+00:00")]
    st = tw.status(_rec(), hits=hits, idx=idx, asof="2021-01-15")
    assert st["status"] == "stood" and st["n_refuting_filers"] == 1
    # a second DISTINCT filer trips it
    hits.append(_hit("BBB", "2020-11-09T12:00:00+00:00"))
    st = tw.status(_rec(), hits=hits, idx=idx, asof="2021-01-15")
    assert st["status"] == "refuted" and st["refuting_filers"] == ["AAA", "BBB"]


def test_status_ignores_gate_failures_and_consistent_evidence():
    idx = _idx([("AAA", "a1", "2020-11-05", "2020-11-05T12:00:00+00:00", "7.01")])
    hits = [_hit("AAA", "2020-11-05T12:00:00+00:00", quote_gate="failed"),
            _hit("BBB", "2020-11-05T12:00:00+00:00", bears_on="consistent"),
            _hit("CCC", "2020-11-05T12:00:00+00:00", is_evidence=False)]
    st = tw.status(_rec(min_distinct_filers=1), hits=hits, idx=idx, asof="2021-01-15")
    assert st["status"] == "stood" and st["n_refuting_filers"] == 0


def test_status_pending_inside_horizon_and_unscoreable_without_coverage():
    idx = _idx([("AAA", "a1", "2020-11-05", "2020-11-05T12:00:00+00:00", "7.01")])
    st = tw.status(_rec(), hits=[], idx=idx, asof="2020-11-10")
    assert st["status"] == "pending"
    empty = _idx([])
    st = tw.status(_rec(), hits=[], idx=empty, asof="2021-01-15")
    assert st["status"] == "unscoreable"


def test_status_verdicts_carry_a_power_statement():
    """Repo rule: a verdict without coverage counts is rejected."""
    idx = _idx([("AAA", "a1", "2020-11-05", "2020-11-05T12:00:00+00:00", "7.01,9.01"),
                ("BBB", "b1", "2020-11-06", "2020-11-06T12:00:00+00:00", "5.02")])
    st = tw.status(_rec(), hits=[], idx=idx, asof="2021-01-15")
    cov = st["coverage"]
    assert cov["members_with_any_filing"] == 2
    assert cov["filings_matching_items"] == 1  # only a1 carries 7.01


def test_calm_windows_avoid_registered_episodes():
    for w in tw.calm_windows(42):
        end = w + pd.tseries.offsets.BDay(42)
        for e in tw.EPISODES:
            ep = pd.Timestamp(e)
            assert not (w - pd.Timedelta(days=21) <= ep <= end + pd.Timedelta(days=21))


def test_log_falsifier_appends_once_per_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(translate, "LOG", tmp_path / "falsifiers.jsonl")
    t = translate.Translation(
        driver="d", transmission="t", hedge_instrument="h", hedge_rationale="hr",
        catalysts=[], confidence=5,
        falsifier=translate.Falsifier(variable="DGS10", direction="rise",
                                      threshold=0.4, horizon_days=42, refutes="r"))
    translate.log_falsifier("2020-01-31", t, {"run_at": "2026-08-27"})
    translate.log_falsifier("2020-01-31", t, {"run_at": "2026-08-28"})  # nightly re-run
    lines = (tmp_path / "falsifiers.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    # a changed threshold is a NEW claim
    t2 = t.model_copy(deep=True)
    t2.falsifier.threshold = 0.5
    translate.log_falsifier("2020-01-31", t2, {})
    assert len((tmp_path / "falsifiers.jsonl").read_text().strip().splitlines()) == 2
    # legacy rows without a key still suppress duplicates
    rec = json.loads(lines[0])
    del rec["key"]
    (tmp_path / "falsifiers.jsonl").write_text(json.dumps(rec) + "\n")
    translate.log_falsifier("2020-01-31", t, {"run_at": "2026-08-29"})
    assert len((tmp_path / "falsifiers.jsonl").read_text().strip().splitlines()) == 1


def test_event_falsifier_fields_survive_logging(tmp_path, monkeypatch):
    monkeypatch.setattr(translate, "LOG", tmp_path / "falsifiers.jsonl")
    t = translate.Translation(
        driver="d", transmission="t", hedge_instrument="h", hedge_rationale="hr",
        catalysts=[], confidence=5,
        falsifier=translate.Falsifier(variable="DGS10", direction="rise",
                                      threshold=0.4, horizon_days=42, refutes="r"),
        event_falsifier=translate.EventFalsifier(
            item_codes=["7.01"], anchor_terms=["withdraw guidance"],
            min_distinct_filers=2, horizon_days=42, refutes="er"))
    rec = translate.log_falsifier("2020-01-31", t, {})
    assert rec["event_falsifier"]["min_distinct_filers"] == 2
    # and an explicit DECLINE is logged as such, not silently dropped
    td = t.model_copy(deep=True)
    td.event_falsifier = None
    td.event_decline_reason = "no issuer-event signature"
    td.driver = "rates"
    rec = translate.log_falsifier("2020-01-31", td, {})
    assert rec["event_falsifier"] is None
    assert rec["event_decline_reason"] == "no issuer-event signature"
