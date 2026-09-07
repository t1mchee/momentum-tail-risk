"""exp-059: deterministic firm-level aggregation of harvest labels, stated before the batch runs.

The registration requires the passage-label -> firm-label rule to exist in code BEFORE any
batch submits, so that the aggregation cannot be tuned after the labels are seen. This module
is that statement. It contains arithmetic and set logic only -- no model call, no threshold
that was not written down here first.

The registered quantity is attribution coverage: the proportion of named-theme member firms
whose own filed text, available at the naming date, discloses the exposure the name asserts.
The named exposure for exp-059 is the AI-compute / data-center demand bet, which the frozen
harvest schema carries as the anchored theme ``ai_compute``.

Rule, fixed here
----------------
Passage level (from one harvest record's anchored read of ``ai_compute``):
  * ``exposed``   -- direction is one of hurt_by_rising / benefits_from_rising /
                     exposed_direction_unstated / hedged_neutral, AND magnitude >= 1,
                     AND the verbatim quote passes the quote gate against the passage text.
                     A claim whose quote cannot be found is treated as not stated, never
                     as evidence.
  * ``denied``    -- direction is stated_immaterial with a grounded quote. A denial is a
                     statement, and the firm-level three-state carries it separately.
  * ``not_stated``-- anything else (not_discussed, ungrounded quote, missing record).

Firm level: ANY-POSITIVE over the firm's passages available at the as-of date.
  * ``exposed``    if any passage is exposed;
  * ``denied``     else if any passage is denied;
  * ``not_stated`` otherwise -- including the case of a firm with no passages at all,
                    which counts in the denominator: silence is the finding, not missing data.

The registered proportion is #exposed / #theme members, computed over passages available at
the NAMING DATE (2025-01-24 end of day, UTC). Post-naming passages in the slice (8-Ks through
2025-03-31) are harvested for description but excluded from the registered quantity by the
``as_of`` filter here.
"""

from __future__ import annotations

import pandas as pd

NAMED_THEME = "ai_compute"
NAMING_DATE_CUTOFF = pd.Timestamp("2025-01-24T23:59:59Z")

#: Directions that state the exposure exists (whatever its sign or hedging).
EXPOSED_DIRECTIONS = frozenset(
    {"hurt_by_rising", "benefits_from_rising", "exposed_direction_unstated", "hedged_neutral"})
DENIED_DIRECTION = "stated_immaterial"


def passage_label(harvest: dict, quote_grounded: bool) -> str:
    """One passage's three-state read of the named theme. Pure function of the record.

    ``harvest`` is the ``harvest`` dict of one cached/collected record (schema harvest-v3);
    ``quote_grounded`` is the mechanical quote-gate verdict for the named theme's quote,
    checked against the passage text by the caller.
    """
    for th in harvest.get("themes", []):
        if th.get("theme") != NAMED_THEME:
            continue
        d = th.get("direction")
        if d in EXPOSED_DIRECTIONS and int(th.get("magnitude", 0)) >= 1 and quote_grounded:
            return "exposed"
        if d == DENIED_DIRECTION and quote_grounded:
            return "denied"
        return "not_stated"
    return "not_stated"


def firm_labels(passages: pd.DataFrame, theme_members: list[str],
                *, as_of: pd.Timestamp = NAMING_DATE_CUTOFF) -> pd.DataFrame:
    """ANY-POSITIVE aggregation to firm level over passages available at ``as_of``.

    ``passages`` needs columns: ticker, available_at (tz-aware or naive UTC), label
    (exposed/denied/not_stated from :func:`passage_label`). Firms in ``theme_members``
    with no rows get ``not_stated`` and stay in the denominator.
    """
    df = passages.copy()
    av = pd.to_datetime(df["available_at"], utc=True)
    cut = pd.Timestamp(as_of)
    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    df = df[av <= cut]
    out = []
    for t in sorted(theme_members):
        sub = df[df["ticker"] == t]
        if (sub["label"] == "exposed").any():
            lab = "exposed"
        elif (sub["label"] == "denied").any():
            lab = "denied"
        else:
            lab = "not_stated"
        out.append({"ticker": t, "firm_label": lab,
                    "n_passages_asof": int(len(sub)),
                    "n_exposed": int((sub["label"] == "exposed").sum()),
                    "n_denied": int((sub["label"] == "denied").sum())})
    return pd.DataFrame(out)


def coverage_proportion(firms: pd.DataFrame) -> float:
    """The registered quantity: #exposed / #theme members."""
    return float((firms["firm_label"] == "exposed").mean()) if len(firms) else float("nan")
