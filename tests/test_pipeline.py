"""Seam tests for the end-to-end pipeline.

These do not check that any number is right — the experiments do that. They check the
properties the brief's honesty rests on: that a stage which cannot run says why, that a gap is
never silently rendered as a zero, that a partition computed for one book cannot be quietly
reused for a book eight months later, and that nothing reaches forward in time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from unstructured_momentum.pipeline import run as R
from unstructured_momentum.pipeline.brief import render
from unstructured_momentum.pipeline.dates import GOLDEN, resolve

#: The tier protocol seals 2023 onward. A test that RENDERS a sealed date reads sealed data
#: whatever its intent, so the split is structural rather than a convention anyone must
#: remember: golden-date tests run on pre-2023 dates, the live date gets its own test, and
#: nothing in this file renders 2023 or later.
PRE_2023 = [g for g in GOLDEN if g.as_of != "TODAY" and g.as_of < "2023-01-01"]
LIVE = [g for g in GOLDEN if g.as_of == "TODAY"]
SEALED = [g for g in GOLDEN if g.as_of != "TODAY" and g.as_of >= "2023-01-01"]


def test_golden_set_is_documented():
    """Each regression date must say what its failure would mean, or it is not diagnostic."""
    assert len(GOLDEN) == 6
    for g in GOLDEN:
        assert g.why.strip() and g.breaks_if.strip(), f"{g.as_of} lacks a rationale"


@pytest.mark.slow
@pytest.mark.parametrize("g", PRE_2023, ids=[g.label for g in PRE_2023])
def test_every_pre_2023_golden_date_renders(g):
    res = R.run(resolve(g), translate=False)
    text = render(res)
    assert text.strip() and "MOMENTUM REVERSAL RISK" in text
    # A run that measured nothing at all is a failure, not a quiet market.
    assert res.values, f"{g.as_of}: no stage produced a value"


@pytest.mark.slow
def test_gaps_always_carry_a_reason():
    res = R.run("2020-05-22", translate=False)
    assert res.could_not_measure, "expected unbuilt stages to be reported"
    for name, why in res.could_not_measure.items():
        assert why and why.strip(), f"{name} reported an empty reason"
        assert len(why) > 12, f"{name} reason is too thin to act on: {why!r}"


@pytest.mark.slow
def test_gaps_are_rendered_not_hidden():
    """The reader must be able to tell 'measured and quiet' from 'never measured'."""
    res = R.run("2020-05-22", translate=False)
    text = render(res)
    assert "COULD NOT MEASURE" in text
    for name in res.could_not_measure:
        assert name in text, f"{name} was omitted from the brief rather than reported"


@pytest.mark.slow
def test_stale_partition_is_refused_not_reused():
    """A partition describes one book. Reusing it months later would describe other holdings.

    Re-dated from 2023-01-19 to a pre-2023 date. The property under test is staleness refusal,
    which has nothing to do with which date is used, so the sealed date bought nothing and cost
    a tier violation on every run.
    """
    res = R.run("2022-06-30", translate=False)
    why = res.could_not_measure.get("composition", "")
    assert "stale" in why, f"expected a staleness refusal, got {why!r}"


@pytest.mark.slow
def test_conferred_stage_does_not_reach_forward():
    asof = pd.Timestamp("2019-09-06")
    res = R.run(asof, translate=False)
    c = res.values.get("conferred")
    assert c, "conferred stage did not run on a date the panel supports"
    assert pd.Timestamp(c["available_at"]) <= asof, (
        "conferred loading claims availability after the date it describes")


@pytest.mark.slow
def test_translation_refuses_without_a_measurement():
    """The model must not write an operational read from nothing."""
    with pytest.raises(RuntimeError, match="no measured tilt"):
        R.stage_translation(pd.Timestamp("2019-09-06"), {})


@pytest.mark.slow
@pytest.mark.live
@pytest.mark.parametrize("g", LIVE, ids=[g.label for g in LIVE])
def test_the_live_date_renders(g):
    """Today's brief, tagged separately because it is the only one whose date moves.

    Kept apart from the golden set so a failure here reads as "the live path broke" rather than
    as a regression against a fixed case.
    """
    res = R.run(resolve(g), translate=False)
    text = render(res)
    assert text.strip() and "MOMENTUM REVERSAL RISK" in text
    assert res.values, "no stage produced a value on the live date"


def test_no_test_in_this_file_renders_a_sealed_date():
    """The structural guarantee, asserted rather than trusted.

    The golden set legitimately CONTAINS a sealed date -- it is the out-of-design check -- and
    nothing here proposes removing it. What this asserts is that the test suite does not render
    it, so the tier protocol cannot be violated by a parametrisation nobody re-reads.
    """
    src = Path(__file__).read_text()
    import re
    rendered = set(re.findall(r'R\.run\(\s*"(\d{4}-\d{2}-\d{2})"', src))
    sealed = {d for d in rendered if d >= "2023-01-01"}
    assert not sealed, f"tests render sealed dates: {sorted(sealed)}"
    assert not any(g.as_of >= "2023-01-01" for g in PRE_2023 if g.as_of != "TODAY")
    assert SEALED, "the golden set should still contain its sealed out-of-design date"


# ------------------------------------------------------------------ the shared contract
#
# Element 1 of the brief is answered by ONE block printed at the head of both PM-facing
# products. These check the properties that make it worth having: that it is the same text in
# both, that the horizon it declares is one the severity model actually fits, and that it names
# every book in use rather than quietly picking a favourite.

from unstructured_momentum.model import severity as _sev  # noqa: E402
from unstructured_momentum.pipeline import contract  # noqa: E402


def _definition_only(lines, prefix):
    """The block minus its per-run measurements, which are allowed to differ."""
    out, skip = [], False
    for ln in lines:
        ln = ln[len(prefix):] if ln.startswith(prefix) else ln
        if "this run" in ln:
            skip = True
            continue
        if skip and ln.startswith(" " * 22):
            continue
        skip = False
        out.append(ln.rstrip())
    return out


def test_the_problem_definition_block_is_identical_in_both_products():
    """One block, two products. If they can drift apart, element 1 has two answers again."""
    a = _definition_only(contract.header_lines("french_wml",
                                               resolved={"french_wml": "brief-side detail"}),
                         "  ")
    b = _definition_only(contract.header_lines("xray_month_end",
                                               resolved={"xray_month_end": "xray-side detail"},
                                               comment=""),
                         "")
    assert a == b, "the canonical block differs between the two products"


def test_declared_horizon_is_one_the_model_actually_fits():
    """A declared horizon the severity model does not fit would be a promise, not a spec."""
    assert contract.DECLARED_HORIZON_DAYS in _sev.HORIZONS
    assert contract.DECLARED_QUANTILE in _sev.QUANTILES
    assert contract.DECLARED_LEVEL == _sev.CRASH_THRESHOLD_10D[contract.DECLARED_QUANTILE]


def test_every_live_book_is_named_and_the_divergence_is_flagged():
    """The repo's own finding is that construction changes which crashes exist. Say so."""
    text = "\n".join(contract.header_lines("french_wml"))
    for b in contract.BOOKS:
        assert f"[{b.tag}]" in text and b.label in text, f"{b.key} is not in the block"
    assert "NOT THE SAME PORTFOLIO" in text
    assert "clm-construction" in text


def test_an_unknown_book_cannot_print_the_block():
    with pytest.raises(KeyError):
        contract.header_lines("some_book_nobody_declared")


# ------------------------------------------------------------------ the named-bet seam


def test_named_bet_refuses_a_stale_xray_rather_than_serving_it(tmp_path, monkeypatch):
    """The X-ray's themes come from a trailing 21-day window. Older is a different news state."""
    import json as _json
    d = tmp_path / "xray"
    d.mkdir()
    (d / "2020-01-02.json").write_text(_json.dumps(
        {"top_theme": {"members": ["A"], "n": 1}, "bet_name": "x"}))
    monkeypatch.setattr(R, "XRAY_DIR", d)
    with pytest.raises(RuntimeError, match="days before this brief"):
        R.stage_named_bet(pd.Timestamp("2020-06-01"))


def test_named_bet_never_reaches_forward_for_an_xray(tmp_path, monkeypatch):
    import json as _json
    d = tmp_path / "xray"
    d.mkdir()
    (d / "2020-06-30.json").write_text(_json.dumps(
        {"top_theme": {"members": ["A"], "n": 1}, "bet_name": "x"}))
    monkeypatch.setattr(R, "XRAY_DIR", d)
    with pytest.raises(RuntimeError, match="no X-ray artifact at or before"):
        R.stage_named_bet(pd.Timestamp("2020-06-01"))


def test_named_bet_reports_a_reason_when_no_xray_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "XRAY_DIR", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="never run"):
        R.stage_named_bet(pd.Timestamp("2020-06-01"))


# ------------------------------------------------------------------ the odds panel


def test_conditioner_recipes_are_the_ones_the_ratios_were_measured_under():
    """The state printed beside a ratio must come from the code that produced the ratio."""
    from unstructured_momentum.features import conditioners as cond
    assert (cond.TOP_Q, cond.HORIZON, cond.CRASH_Q) == (0.80, 21, 0.01)
    for name in ("comomentum", "momentum_gap", "bear_vol"):
        assert name in cond.BUILDERS and name in cond.SOURCES
    # The failed control is carried deliberately: it is the scale the three are read against.
    assert "ours_top_share_descriptive" in cond.BUILDERS


def test_the_odds_panel_carries_its_no_timing_disclosure():
    for phrase in ("NOT A FORECAST", "NOT A TIMING SIGNAL", "HISTORICAL CRASH FREQUENCY RATIO",
                   "FAILED"):
        assert phrase in R.ODDS_DISCLOSURE, f"{phrase!r} missing from the odds disclosure"


# ------------------------------------------------------------------ the nightly seam


def test_a_broken_xray_cannot_take_the_brief_down(monkeypatch):
    """The brief is the older, load-bearing product. An X-ray failure costs it one line."""
    from unstructured_momentum.pipeline import nightly
    monkeypatch.setattr(nightly, "XRAY_SCRIPT", Path("/nonexistent/daily_xray.py"))
    row = nightly.run_xray(pd.Timestamp("2020-06-01"))
    assert row["status"] == "failed" and row["error"], "a failure must be recorded, not raised"
