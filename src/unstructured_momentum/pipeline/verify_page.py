"""exp-068 — the composition verifier. The one agent role whose value does not depend on a
language model behaving well.

It checks the finished document rather than trusting the process that made it: every numeric
field names an artifact and a hash, every quote is re-checked against text re-derived from
the corpus, and the falsifier names a variable, a comparator, a threshold and a horizon. If
any check fails the page is REFUSED with reasons.

The refusal path is exercised, not assumed. This repo recorded a trap earlier the same day
about a guard placed where it could never run, so the test suite corrupts a real page three
ways -- provenance stripped, a quote altered by one word, a falsifier missing its threshold --
and requires a refusal for each.
"""

from __future__ import annotations

from dataclasses import dataclass, field

REQUIRED_FALSIFIER = ("variable", "comparator", "threshold", "horizon_months")


@dataclass
class Verdict:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    checked: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"accepted": self.accepted, "reasons": self.reasons, "checked": self.checked}


def _walk_numbers(obj, path="") -> list[tuple[str, dict]]:
    """Every dict that looks like a Num record, with the path that reached it."""
    out = []
    if isinstance(obj, dict):
        if "value" in obj and "provenance" in obj:
            out.append((path, obj))
        for k, v in obj.items():
            out += _walk_numbers(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _walk_numbers(v, f"{path}[{i}]")
    return out


def verify(page: dict, *, block_texts: dict[str, str] | None = None) -> Verdict:
    """`block_texts` maps 'query_month|analogue_month|BLOCK' to the re-derived source text."""
    from ..llm.exposure import quote_is_grounded

    reasons: list[str] = []

    # 1. every number traces to an artifact and a hash
    nums = _walk_numbers(page)
    bad_prov = []
    for path, n in nums:
        p = n.get("provenance") or {}
        if not p.get("source") or not p.get("sha256") or not p.get("field"):
            bad_prov.append(path)
        elif p["sha256"] == "MISSING":
            bad_prov.append(f"{path} (artifact absent)")
    if bad_prov:
        reasons.append(f"{len(bad_prov)} numeric field(s) without usable provenance: "
                       + ", ".join(bad_prov[:5]))

    # 2. every quote re-verifies against text re-derived from the corpus
    n_quotes, n_unverified = 0, 0
    for match in page.get("sections", {}).get("analogues", {}).get("matches", []):
        nc = match.get("narrative_comparison")
        if not nc:
            continue
        for fieldname in ("shared_conditions", "disanalogies", "what_followed_then"):
            for it in nc.get(fieldname, []):
                n_quotes += 1
                if block_texts is None:
                    continue
                key = f"{page['as_of']}|{match['date']}|{it['block'].upper().strip()}"
                src = block_texts.get(key)
                if src is None or not quote_is_grounded(it["quote"], src):
                    n_unverified += 1
    if n_unverified:
        reasons.append(f"{n_unverified} of {n_quotes} quotes did not verify against source")

    # 3. the falsifier is machine-checkable
    f = page.get("falsifier") or {}
    if f.get("available"):
        missing = [k for k in REQUIRED_FALSIFIER if f.get(k) is None]
        if missing:
            reasons.append("falsifier is not machine-checkable, missing: "
                           + ", ".join(missing))
        elif f.get("comparator") not in ("<", ">", "<=", ">="):
            reasons.append(f"falsifier comparator {f.get('comparator')!r} is not evaluable")
    elif f and not f.get("reason"):
        reasons.append("falsifier unavailable without a stated reason")

    # 4. the could-not-measure section must exist and give reasons
    cnm = page.get("could_not_measure", [])
    if not cnm:
        reasons.append("no could-not-measure section; a page claiming full coverage is wrong")
    for c in cnm:
        if not c.get("reason"):
            reasons.append(f"could-not-measure entry {c.get('item')!r} has no reason")

    return Verdict(accepted=not reasons, reasons=reasons,
                   checked={"numeric_fields": len(nums), "quotes": n_quotes,
                            "quotes_unverified": n_unverified,
                            "could_not_measure_entries": len(cnm),
                            "falsifier_checkable": bool(f.get("available")
                                                        and not [k for k in REQUIRED_FALSIFIER
                                                                 if f.get(k) is None])})
