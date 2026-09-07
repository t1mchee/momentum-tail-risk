"""Contract tests for the conferred-tilt stage.

These guard the two properties a downstream consumer relies on and cannot check for itself:
that the in-formation block is genuinely available at the date it describes, and that the
after-formation block is not. The 2020-05-31 book shows why it matters — the conferred spread
is -0.043 and the realised one -0.480, an order of magnitude apart at the same date — so a
consumer reaching the wrong column gets a plausible number that could not have been known.
"""

from __future__ import annotations

import pandas as pd
import pytest

from unstructured_momentum.features import conferred


def test_units_guard_rejects_percent_factors(monkeypatch):
    """A factor series in percent must raise, not silently inflate every loading."""
    real = conferred.french.market

    def as_percent():
        return real() * 100.0

    monkeypatch.setattr(conferred.french, "market", as_percent)
    with pytest.raises(ValueError, match="not decimal returns"):
        conferred._factors()


def test_coverage_counts_dates_not_months():
    """The panel is monthly before 2013; a year of monthly snapshots is not usable."""
    c = conferred.coverage()
    assert {"year", "dates", "months", "cadence", "usable"} <= set(c.columns)
    monthly = c[c.cadence == "monthly"]
    assert not monthly["usable"].any(), "a monthly year cannot support a daily-return loading"
    assert c[c.year == 2013]["usable"].item() is True


def test_first_usable_date_is_derived():
    d = conferred.first_usable_date()
    assert d is not None and d.year == 2013


@pytest.mark.slow
def test_availability_stamps_separate_the_two_estimands():
    conf = conferred.build("2020-03-31", "2020-06-30")
    real = conferred.build("2020-03-31", "2020-06-30", realised=True)
    assert len(conf) and len(real)

    # In-formation loadings are known at the date they describe.
    assert (conf["available_at"] <= conf["as_of"]).all()
    # After-formation loadings are NOT, and must be stamped forward.
    assert (real["available_at"] > real["as_of"]).all()

    assert set(conf["estimand"]) == {"conferred_in_formation"}
    assert set(real["estimand"]) == {"realised_after_formation"}

    # The two blocks must not share a value column, so a consumer cannot reach the
    # unavailable one by asking for a familiar name.
    shared = (set(conf.columns) & set(real.columns)) - {
        "as_of", "observed_at", "available_at", "estimand", "n_leg"}
    assert not shared, f"value columns collide across estimands: {shared}"
