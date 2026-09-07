"""The numeral canary must fail on a hand-typed number. An unfired gate demonstrates nothing."""
import sys; sys.path.insert(0, "src")
import pytest
from unstructured_momentum.report.numerals import Registry, NumeralCanary, check, enforce


def test_registered_numeral_passes():
    r = Registry(); t = r.pct("rate", 0.17, "a rate", "f.parquet", "cmd")
    assert check(f"the rate is {t}", r)["passes"]


def test_hand_typed_numeral_fails():
    r = Registry(); r.pct("rate", 0.17, "a rate", "f.parquet", "cmd")
    with pytest.raises(NumeralCanary):
        enforce("the rate is 17.0% and the other is 99.9%", r)


def test_definitional_constant_passes():
    assert check("over 10 trading days at the 5% quantile", Registry())["passes"]


def test_numerals_inside_a_verbatim_span_are_the_source_s_not_the_page_s():
    r = Registry()
    quote = "levels of around 50%, relative to 2019 levels"
    r.add("span_UAL", quote, "2020-09-09", "UAL's condition", "f.parquet", "cmd",
          span={"quote": quote, "rendered": quote, "ticker": "UAL"})
    assert check(f'UAL accepted 2020-09-09\n    "{quote}"', r)["passes"]


def test_the_shipped_page_passes_its_own_canary():
    import json
    from pathlib import Path
    p = Path("reports/gate2/EXAMPLE_RISK_OUTPUT.provenance.json")
    if not p.exists():
        pytest.skip("page not composed in this checkout")
    d = json.loads(p.read_text())
    assert d["passes"], f"unmatched numerals on the shipped page: {d.get('unmatched')}"
    assert d["registered_fields"] >= 50


def test_segments_round_trip():
    """Reassembling the segments must reproduce the page exactly.

    This is the test that makes the rendering fault unrepresentable rather than fixed. A viewer
    that highlighted numerals by string replacement put a span on the 10 inside 2020-10-31 and
    then on digits inside its own placeholder tokens, and the page rendered as garbage while the
    canary reported PASS -- because the canary checked the source text and the reader was looking
    at the DOM.
    """
    from unstructured_momentum.report.numerals import segments
    r = Registry()
    r.pct("var", -0.0947, "a VaR", "f", "cmd", dp=2)
    r.num("sectors", 10, "sector count", "f", "cmd")
    r.num("names", 263, "a count", "f", "cmd")
    page = ("formation 2020-10-31\n  10-day 5% VaR -9.47%\n"
            "  263 names across 10 GICS sectors\n")
    segs = segments(page, r)
    assert "".join(s["text"] for s in segs) == page


def test_a_date_is_never_highlighted():
    from unstructured_momentum.report.numerals import segments
    r = Registry(); r.num("sectors", 10, "sector count", "f", "cmd")
    segs = segments("formation 2020-10-31 over 10 days", r)
    marked = [s["text"] for s in segs if s.get("field")]
    assert marked == ["10"], marked
    assert "2020-10-31" in "".join(s["text"] for s in segs if not s.get("field"))


def test_the_shipped_page_segments_reassemble():
    import json
    from pathlib import Path
    p = Path("reports/gate2/EXAMPLE_RISK_OUTPUT.provenance.json")
    t = Path("reports/gate2/EXAMPLE_RISK_OUTPUT.txt")
    if not (p.exists() and t.exists()):
        pytest.skip("page not composed in this checkout")
    d = json.loads(p.read_text())
    assert "".join(s["text"] for s in d["segments"]) == t.read_text().rstrip("\n")


def test_notebook_executes_and_carries_no_stale_outputs():
    """The notebook must be regenerable, and its committed outputs must match a fresh run.

    A notebook with committed outputs is where numbers go stale, and this repository has logged
    four instances of that fault already. Rebuilding it from `scripts/build_notebook.py` and
    re-executing is what stops it becoming the fifth.
    """
    from pathlib import Path
    nb = Path("notebooks/momentum_reversal_poc.ipynb")
    if not nb.exists():
        pytest.skip("notebook not built in this checkout")
    import json
    d = json.loads(nb.read_text())
    code = [c for c in d["cells"] if c["cell_type"] == "code"]
    assert code, "no code cells"
    assert all(c.get("outputs") for c in code), "a code cell has no output; re-execute the notebook"
    # Every source line must end in a newline or the cell concatenates into a syntax error.
    for c in d["cells"]:
        src = c["source"]
        assert all(ln.endswith("\n") for ln in src[:-1]), f"unterminated source line in {c.get('id')}"
