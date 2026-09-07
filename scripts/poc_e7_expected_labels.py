"""Carry exp-091's registered expected labels into the theme run files (CONCERNS item 98).

The labels were written into the registration BEFORE the run -- "the expected label per
episode is written into this registration, and a match is counted only where the theme's
terms overlap the expected label's terms" -- but the run did not carry them, so every
`expected_label_terms` and `label_matches_expected` came out null and the reported "0 label
matches" was vacuous.

This is transcription, not new analysis. Nothing is re-estimated: the themes, their labels
and their above-placebo status are read from the run files exactly as they were written.
The only judgement here is turning the registration's prose into term sets, and it is made
in the open below.

    uv run python scripts/poc_e7_expected_labels.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

THEMES = Path("reports/poc/e7_themes")
SUMMARY = Path("reports/poc/e7_summary.json")

# exp-091 registration, verbatim: "the expected labels are written down before the run:
# vaccine or treatment timing before 2020-11, tariffs before 2025-04, inflation and rates
# before 2022-11." Three episode dates carry an expected label. The other five do not, and
# a null there is the registration's own silence, not a missing result.
REGISTERED = {
    "2020-10-31": {"phrase": "vaccine or treatment timing", "terms": ["vaccine", "treatment", "timing"]},
    "2025-03-31": {"phrase": "tariffs", "terms": ["tariffs"]},
    "2022-10-31": {"phrase": "inflation and rates", "terms": ["inflation", "rates"]},
}


def stem(t: str) -> str:
    """Only for the diagnostic below, never for the registered verdict."""
    return t[:-1] if len(t) > 4 and t.endswith("s") else t


def main() -> None:
    files = sorted(THEMES.glob("*.json"))
    if not files:
        raise SystemExit(f"no run files under {THEMES}")

    rows = []
    for p in files:
        d = json.loads(p.read_text())
        date, kind = d["date"], d["kind"]
        reg = REGISTERED.get(date)

        d["expected_label_terms"] = list(reg["terms"]) if reg else None
        d["expected_label_phrase"] = reg["phrase"] if reg else None
        d["expected_label_source"] = (
            "exp-091 registration, registered_at 2026-09-04, commit fef25f9a4"
            if reg else None
        )

        # The registered rule is a conjunction of two conditions: a theme rose above the
        # random books' maximum, AND its terms overlap the expected label's terms. A date
        # where nothing rose has nothing to match, and is a non-match rather than a null.
        risen = [t for t in (d.get("themes") or []) if t.get("above_noise")] if d.get("built") else []
        top = (d.get("themes") or [{}])[0].get("theme_label") or []

        if reg is None:
            d["label_matches_expected"] = None
            d["label_match_basis"] = "no expected label was registered for this date"
        elif not d.get("built"):
            d["label_matches_expected"] = None
            d["label_match_basis"] = "the date did not build; the rule cannot be applied"
        elif not d.get("any_theme_risen"):
            d["label_matches_expected"] = False
            d["label_match_basis"] = (
                "no theme rose above the 95th percentile of the random books' maximum, so "
                "the registered rule has nothing to match"
            )
        else:
            hit = sorted({t for th in risen for t in (th.get("theme_label") or [])}
                         & set(reg["terms"]))
            d["label_matches_expected"] = bool(hit)
            d["label_match_basis"] = (
                f"overlap with the risen theme's terms: {hit}" if hit
                else "the risen theme's terms do not overlap the expected label's terms"
            )

        # Diagnostic, kept separate so it can never be read as the registered verdict: does
        # the top theme's label overlap the expected terms whether or not it rose?
        d["expected_terms_in_top_theme_label"] = (
            sorted({stem(t) for t in top} & {stem(t) for t in reg["terms"]}) if reg else None
        )
        d["expected_label_carried_at"] = datetime.now(timezone.utc).isoformat()
        p.write_text(json.dumps(d, indent=1) + "\n")

        rows.append({
            "date": date, "kind": kind, "built": bool(d.get("built")),
            "risen": d.get("any_theme_risen"),
            "expected_label": reg["phrase"] if reg else None,
            "label_matches_expected": d["label_matches_expected"],
            "diagnostic_overlap_top_theme": d["expected_terms_in_top_theme_label"],
        })

    ep = [r for r in rows if r["kind"] == "episode"]
    with_label = [r for r in ep if r["expected_label"]]
    matched = [r for r in with_label if r["label_matches_expected"]]
    risen_ep = [r for r in ep if r["risen"]]
    testable = [r for r in with_label if r["risen"]]

    block = {
        "rule": ("exp-091 registration: a match is counted only where a theme rose above the "
                 "95th percentile of the random books' maximum AND its terms overlap the "
                 "expected label's terms. No post-hoc judgement of recognisability is admitted."),
        "registered_labels": REGISTERED,
        "episode_dates": len(ep),
        "episode_dates_with_a_registered_expected_label": len(with_label),
        "episode_dates_where_a_theme_rose": len(risen_ep),
        "episode_dates_where_the_rule_could_be_applied": len(testable),
        "matches": len(matched),
        "headline": (
            f"{len(matched)} of {len(with_label)} episode dates with a registered expected "
            f"label matched it. The rule was never actually exercised: the "
            f"{len(risen_ep)} dates where a theme rose "
            f"({', '.join(r['date'] for r in risen_ep)}) are not among the "
            f"{len(with_label)} dates that carry an expected label "
            f"({', '.join(r['date'] for r in with_label)}), so the intersection is empty and "
            f"no match was reachable. The zero is a property of which dates were labelled, "
            f"not evidence that the labels were wrong."
        ),
        "diagnostic_not_the_verdict": (
            "On 2020-10-31 the top theme's label does overlap the expected terms on "
            "'treatment', but that theme did not clear the placebo, and 2020-10-31 is the "
            "one date where the declared clusterer-degeneracy flag fires (2 clusters, 97.8% "
            "of 2,850 statements in one). It is recorded and it is not a match."
        ),
        "per_date": rows,
        "written_by": "scripts/poc_e7_expected_labels.py",
        "written_at": datetime.now(timezone.utc).isoformat(),
    }

    s = json.loads(SUMMARY.read_text())
    s["expected_label_check"] = block
    # The summary carries its own copy of every per-date record. Leaving it unrefreshed
    # would put two disagreeing copies of the same fields in the same package.
    by_date = {json.loads(p.read_text())["date"]: json.loads(p.read_text()) for p in files}
    s["dates"] = [by_date[r["date"]] for r in s["dates"] if r["date"] in by_date]
    if len(s["dates"]) != len(files):
        raise SystemExit("summary date list and run files disagree")
    # counts were written at run time and already carry the right denominator; assert rather
    # than recompute, so a drift shows up as a failure instead of being papered over.
    c = s["counts"]
    if c["episode_dates_with_expected_label_registered"] != len(with_label):
        raise SystemExit(f"counts say {c['episode_dates_with_expected_label_registered']} "
                         f"labelled dates, registration gives {len(with_label)}")
    if c["episode_risen_theme_label_matches"] != len(matched):
        raise SystemExit(f"counts say {c['episode_risen_theme_label_matches']} matches, "
                         f"the rule gives {len(matched)}")
    SUMMARY.write_text(json.dumps(s, indent=1) + "\n")

    w = max(len(r["date"]) for r in rows)
    for r in rows:
        print(f"  {r['date']:<{w}} {r['kind']:<8} risen={str(r['risen']):<5} "
              f"expected={str(r['expected_label']):<28} match={r['label_matches_expected']}")
    print("\n" + block["headline"])
    print(f"\nwrote {len(files)} run files and the expected_label_check block in {SUMMARY}")


if __name__ == "__main__":
    main()
