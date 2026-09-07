"""Gate 0 data-integrity guards, verified in code rather than asserted in comments.

Each test corresponds to an item in the Gate 0 audit. They exist because every one of these
faults produces SILENCE rather than an error: a terminated series reads as a calm market, a
carried-forward book reads as a re-struck one, and a retrospectively built index reads as a
contemporaneous observation.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from conftest import needs
import pytest
import yaml


# ---- item 2: a terminated series must propagate as missing, never as a low value ------
def test_discontinued_cboe_series_are_nan_after_termination() -> None:
    from unstructured_momentum.data import cboe

    disc = getattr(cboe, "DISCONTINUED", None)
    assert disc, "cboe module must declare which series are discontinued"
    for name in list(disc)[:3]:
        try:
            s = cboe.close(name)
        except Exception as exc:  # pragma: no cover - a missing local file is not a fault here
            pytest.skip(f"{name} not cached locally: {type(exc).__name__}")
        assert s.notna().all(), (
            f"{name} carries NaN inside its own span; a loader must drop rather than keep them")
        # The series must simply END. Reindexing past the end must yield NaN, not a stale or
        # zero value, because downstream a zero reads as calm and a stale value reads as live.
        after = s.reindex(pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=5))
        assert after.isna().all(), f"{name} does not propagate as missing after termination"


def test_market_returns_are_not_zero_filled_in_the_state_features() -> None:
    """A missing market return must not become a zero return.

    fillna(0) on a return series turns an absent observation into an observed flat day, which
    is the same failure mode as a terminated index reading as calm.
    """
    src = Path("src/unstructured_momentum/pipeline/features.py").read_text()
    offending = [ln.strip() for ln in src.splitlines()
                 if "fillna(0" in ln and "Mkt-RF" in ln]
    assert not offending, (
        "market returns are zero-filled in pipeline/features.py, which converts a missing "
        f"observation into an observed flat day: {offending}")


# ---- item 3: leg membership must not be carried across a holdings gap -----------------
def _registered_gaps() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    needs("project/sources.yaml", "the source register")
    data = yaml.safe_load(Path("project/sources.yaml").read_text())
    entries = data if isinstance(data, list) else data.get("sources", data)
    for s in entries:
        if s.get("id") == "src-iwv":
            return [(pd.Timestamp(g["from"]), pd.Timestamp(g["to"]))
                    for g in (s.get("gaps") or [])]
    return []


def test_leg_membership_is_not_silently_carried_across_a_holdings_gap() -> None:
    """Byte-identical consecutive legs inside a known gap mean a frozen book.

    Month-to-month loser-leg Jaccard is normally 0.5-0.6. A run of 1.000 inside a period with
    no underlying holdings files is the 2016-12 sort being presented as if re-struck monthly.
    Green since the Gate 0 rebuild omitted those seven month-ends at the source.
    """
    path = Path("data/processed/leg_members.pkl")
    if not path.exists():
        pytest.skip("leg panel not built in this checkout")
    legs = pickle.load(path.open("rb"))
    keys = sorted(legs)
    frozen: list[str] = []
    for lo, hi in _registered_gaps():
        inside = [k for k in keys if lo <= k <= hi]
        for a, b in zip(inside, inside[1:]):
            if tuple(legs[a]["losers"]) == tuple(legs[b]["losers"]):
                frozen.append(f"{a.date()}->{b.date()}")
    assert not frozen, (
        "leg membership is carried forward byte-identically inside a registered holdings gap, "
        f"so these month-ends are a frozen book presented as re-struck: {frozen}")


# ---- item 4: retrospectively constructed inputs are restricted ------------------------
def test_uncertainty_indices_are_restricted_to_the_contemporaneous_era() -> None:
    needs("scripts/build_state_vector.py", "the state-vector builder")
    src = Path("scripts/build_state_vector.py").read_text()
    assert "TEXTMACRO_START" in src, "the uncertainty block must declare a start cutoff"
    assert "t[t.index >= TEXTMACRO_START]" in src, (
        "the uncertainty block must apply its cutoff; the pre-1985 portion had its term list "
        "chosen by people who knew what followed")


def test_no_consecutive_month_is_a_near_duplicate_of_its_predecessor() -> None:
    """The general form of the frozen-book fault, not tied to a registered gap.

    The gap-scoped test above only fires where the registry already knows there is a hole, and
    the registry was wrong about one of its two holes. This one needs no registry at all: a
    12-1 decile sort re-struck a month later turns over roughly 40-50 percent of each leg, so
    any consecutive pair above 0.95 is a book that was carried rather than rebuilt, wherever it
    happens and whether or not anyone has recorded a gap there.
    """
    path = Path("data/processed/leg_members.pkl")
    if not path.exists():
        pytest.skip("leg panel not built in this checkout")
    legs = pickle.load(path.open("rb"))
    keys = sorted(legs)
    offenders = []
    for a, b in zip(keys, keys[1:]):
        if (b - a).days > 45:          # a genuine omission, not a carried book
            continue
        for side in ("winners", "losers"):
            x, y = set(legs[a][side]), set(legs[b][side])
            if not x or not y:
                continue
            j = len(x & y) / len(x | y)
            if j > 0.95:
                offenders.append(f"{a.date()}->{b.date()} {side} J={j:.3f}")
    assert not offenders, (
        "consecutive month-end books are near-duplicates, which means membership was carried "
        f"forward rather than re-sorted: {offenders}")


# ---- standing guard: no unclassified zero-fill may enter the package ------------------
def test_every_zero_fill_in_the_package_is_classified() -> None:
    """A missing value silently becoming zero is this codebase's characteristic fault.

    Gate 0 found one that turned an absent market return into an observed flat day. The set of
    sites is small enough to enumerate, so every one is classified in
    project/zero_fill_registry.yaml and this test fails on any that is not. The registry is
    the argument; the test only enforces that the argument was made.
    """
    import re

    needs("project/zero_fill_registry.yaml", "the zero-fill classification registry")
    reg = yaml.safe_load(Path("project/zero_fill_registry.yaml").read_text())
    known = {(s["where"], s["code"]) for s in reg["sites"]}
    pattern = re.compile(r"fillna\(\s*0|nan_to_num|fill_value\s*=\s*0")
    found = set()
    for f in Path("src/unstructured_momentum").rglob("*.py"):
        for line in f.read_text().splitlines():
            if pattern.search(line):
                found.add((str(f.relative_to("src/unstructured_momentum")), line.strip()))
    unclassified = found - known
    assert not unclassified, (
        "these zero-fills are not classified in project/zero_fill_registry.yaml; a missing "
        f"value becoming a zero must be argued for, not defaulted to: {sorted(unclassified)}")
    stale = known - found
    assert not stale, (
        f"the registry lists zero-fills that no longer exist; remove them: {sorted(stale)}")


# ---- standing guard: module constants must actually be bound -------------------------
def test_declared_constants_are_bound_at_module_level() -> None:
    """A constant written into a docstring parses cleanly and is never defined.

    This happened during the Gate 0 fixes: TEXTMACRO_START was inserted inside the module
    docstring of build_state_vector.py. The file compiled, because it was text in a string, and
    the reference below it would have raised at runtime. Parsing is not enough; the name has to
    be bound.
    """
    import ast

    checks = [("scripts/build_state_vector.py", "TEXTMACRO_START")]
    for _p, _ in checks:
        needs(_p, "the module whose constant binding is under test")
    for path, name in checks:
        tree = ast.parse(Path(path).read_text())
        bound = {
            t.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        } | {
            node.target.id
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert name in bound, (
            f"{name} is referenced in {path} but is not bound at module level -- check it did "
            f"not land inside the docstring, which parses cleanly and defines nothing")


# ---- standing guard: the registry must match the store it describes -------------------
def test_registry_reconciles_with_the_store() -> None:
    """Three of five Gate 0 faults lived in the registry and none was visible by reading it.

    This measures instead: first date, last date, distinct dates and month coverage per source,
    compared against the registered claims. It found a third IWV holdings gap nobody had
    recorded, the same 2017 outage in MTUM, and a two-week error in a source entry written the
    previous day.
    """
    needs("scripts/reconcile_sources.py", "the source-coverage reconciler")
    sys.path.insert(0, "scripts")
    from reconcile_sources import reconcile

    failures = []
    for r in reconcile():
        if r["status"] == "NO RESOLVER":
            failures.append(f"{r['id']}: no resolver, and not declared unverifiable or debt")
        elif r["status"] == "ok":
            if r["starts_before_claim"]:
                failures.append(
                    f"{r['id']}: store starts {r['measured_first']} before the claimed "
                    f"{r['claimed_first']}")
            if r["ends_after_claim"]:
                failures.append(
                    f"{r['id']}: store ends {r['measured_last']} after the claimed "
                    f"{r['claimed_last']}")
            if r["undeclared_missing"]:
                failures.append(
                    f"{r['id']}: {len(r['undeclared_missing'])} undeclared missing months "
                    f"{r['undeclared_missing'][:6]}")
    assert not failures, "registry disagrees with the store:\n  " + "\n  ".join(failures)


# ---- standing guard: no "last N files" window without a calendar check ----------------
def test_snapshot_windows_are_calendar_checked() -> None:
    """Counting files is not counting days on a panel that has outages.

    The MTUM flow feature computed a "21-session" change as the 22nd-most-recent snapshot,
    which across the 2017 holdings hole spanned 215 calendar days. Sweeping for the pattern
    found the same fault in the short-volume feature. Any site that indexes a sorted list of
    per-date files by position must bound the resulting window in calendar time.
    """
    import re

    src = Path("src/unstructured_momentum/pipeline/run.py").read_text()
    hits = [i for i, ln in enumerate(src.splitlines(), 1)
            if re.search(r"usable\[-\d+\]\[", ln)]
    for line_no in hits:
        window = "\n".join(src.splitlines()[max(0, line_no - 12):line_no + 2])
        assert "span_days" in window, (
            f"run.py:{line_no} indexes a snapshot list by position with no calendar bound; "
            f"an upstream outage would silently widen the window")
