"""The canonical problem definition, printed verbatim at the head of every PM-facing output.

Required element 1 of the brief asks for the monitored exposure, the reversal target, the
horizon, the intended user and the decision informed. Two products answer that brief -- the
nightly text brief and the daily X-ray -- and until this module existed they answered it
differently, or not at all. One source of truth, one block, both headers.

The awkward part of that block is the book. Three momentum constructions are live in this
repo at once, and they are NOT the same portfolio. The repo's own registered finding
(clm-construction, exp-002) is that how the portfolio is built changes which episodes appear
in it -- August 2007 is nearly twice as severe among large companies as in the headline
construction, and January 2021 disappears entirely under equal weighting. A module that
quietly unified the three would be manufacturing agreement that the evidence says does not
exist. So the block prints all three, names which panels each one carries, and flags the
divergence as a divergence.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import severity as _sv

# --------------------------------------------------------------------------------------
# The declared horizon
# --------------------------------------------------------------------------------------

#: Trading days. The registered headline: `clm-severity-forecastable` reproduces at
#: `--horizon 10 --quantile 0.05`, `severity.HORIZONS` calls 10d the headline and the frozen
#: crash thresholds are 10-day levels. Asserted rather than remembered, below.
DECLARED_HORIZON_DAYS: int = 10

#: The declared left-tail quantile. `severity.QUANTILES` calls 0.05 the reported headline.
DECLARED_QUANTILE: float = 0.05

#: The forecast step the sealed holdout's power statement fixed, in trading days. Overlapping
#: 10-day windows are not independent observations, so the registration named a step that makes
#: consecutive forecasts disjoint and predicted roughly forty forecasts and about two expected
#: breaches from it. The first implementation used all 865 daily forecasts instead and reported
#: a spurious failure; the defence logged at the time was that a registered sampling scheme has
#: to be executed by the code rather than remembered by the person running it. This constant is
#: that defence. A deviation from it is now a diff.
DECLARED_FORECAST_STEP_DAYS: int = 21

assert DECLARED_FORECAST_STEP_DAYS > DECLARED_HORIZON_DAYS, (
    "the forecast step must exceed the horizon or consecutive windows overlap, which is the "
    "error the step exists to prevent")

assert DECLARED_HORIZON_DAYS in _sv.HORIZONS, "declared horizon is not one the model fits"
assert DECLARED_QUANTILE in _sv.QUANTILES, "declared quantile is not one the model fits"
assert DECLARED_QUANTILE in _sv.CRASH_THRESHOLD_10D, (
    "declared quantile has no frozen 10-day crash threshold")

#: The registered crash level at the declared horizon and quantile, and its deeper sibling.
DECLARED_LEVEL: float = _sv.CRASH_THRESHOLD_10D[DECLARED_QUANTILE]
DEEP_LEVEL: float = _sv.CRASH_THRESHOLD_10D[0.01]
THRESHOLD_VINTAGE: str = _sv.CRASH_THRESHOLD_VINTAGE


# --------------------------------------------------------------------------------------
# Book identity
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Book:
    """One momentum construction actually in use, with the panels it carries."""
    key: str
    tag: str            #: one-character marker used in the header
    label: str
    universe: str
    sort: str
    source: str
    carries: str


BOOKS: tuple[Book, ...] = (
    Book(
        key="french_wml", tag="S",
        label="Ken French daily momentum (WML)",
        universe="CRSP US common stock, NYSE breakpoints",
        sort="value-weighted top-minus-bottom decile on 12-2 prior return, 1926-present",
        source="mba.tuck.dartmouth.edu Ken French library, vintaged on disk",
        carries="conditional VaR/ES, exceedance probability, Shapley drivers, "
                "odds-conditioning panel",
    ),
    Book(
        key="conferred_iwv", tag="C",
        label="IWV panel legs, recomputed at the as-of date",
        universe="iShares IWV (Russell 3000) holdings price panel",
        sort="12-1 total return, split-corrected, top/bottom decile struck ON the as-of date",
        source="factor.cleanlegs.momentum + .legs, computed per brief",
        carries="conferred market-beta tilt, placebo entitlement test, component ES",
    ),
    Book(
        key="xray_month_end", tag="X",
        label="IWV panel legs, frozen at the prior month-end",
        universe="iShares IWV (Russell 3000) holdings price panel",
        sort="12-1 decile sort struck at the prior CALENDAR MONTH-END and held",
        source="data/processed/leg_members.pkl via factor.legs.build",
        carries="news themes, story-beta, attention saturation, 13F crowding, named bet, "
                "vol-scaled sizing spine",
    ),
)

BOOK_BY_KEY = {b.key: b for b in BOOKS}

#: Stated once, printed in both products. The honest reading of what the three books are.
DIVERGENCE = (
    "THE THREE BOOKS ARE NOT THE SAME PORTFOLIO AND ARE NOT UNIFIED HERE. [C] and [X] share a "
    "universe and a signal but strike the sort on different days -- [C] at the as-of date, [X] "
    "at the prior month-end -- so their members differ by construction. [S] is a different "
    "universe and a different weighting entirely. The repo's own registered finding "
    "(clm-construction, exp-002) is that construction changes WHICH crashes exist in the "
    "series: August 2007 reads -8.37% among large companies against -4.53% in the headline "
    "construction, and January 2021 reads -6.88% among large companies and -0.89% weighting "
    "every stock equally, where it disappears. There is no neutral construction, so the choice "
    "is stated rather than averaged away. Read every number against the book tagged beside it.")

REVERSAL_TARGET = (
    f"a {DECLARED_HORIZON_DAYS}-day cumulative WML return at or below the registered "
    f"{DECLARED_QUANTILE:.0%} level {DECLARED_LEVEL:+.4f} (deep tail: the 1% level "
    f"{DEEP_LEVEL:+.4f}); thresholds frozen {THRESHOLD_VINTAGE}")

INTENDED_USER = (
    "a PM or factor risk manager already holding US cross-sectional momentum, who has to size "
    "and hedge it today -- not a signal researcher looking for an entry")

DECISION = (
    "how much momentum risk to carry over the next "
    f"{DECLARED_HORIZON_DAYS} trading days, what the current book is actually levered to, and "
    "which named condition would refute that read. NOT an entry or exit timing decision: six "
    "registered programs in this repo independently found no exploitable pre-event warning, "
    "and no panel in either product is a timing signal.")


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

#: The block wraps to this rather than running off, because a header nobody can read is not a
#: header. Deliberately independent of the caller's ``comment`` prefix: if the wrap width moved
#: with the prefix, the two products would break their lines in different places and the block
#: would stop being byte-identical, which is the one property it exists to have. 74 leaves room
#: for the brief's two-space prefix inside its 76-column rule.
WRAP = 74


#: Every field's value starts in the same column, so the block reads as a table.
LABEL_W = 18


def _field(label: str, text: str, comment: str) -> list[str]:
    """One ``label  value`` field, wrapped with a hanging indent under the value column."""
    import textwrap
    label = label.ljust(LABEL_W)
    pad = " " * (2 + len(label) + 2)
    first = f"{comment}  {label}  "
    # break_on_hyphens=False: without it `clm-construction` and `CALENDAR MONTH-END` split
    # across lines, which turns a citation a reader is meant to grep for into two half-words.
    body = textwrap.wrap(text, width=max(WRAP - len(pad), 30),
                         break_on_hyphens=False) or [""]
    return [first + body[0]] + [comment + pad + ln for ln in body[1:]]


def header_lines(product_book: str, *, resolved: dict[str, str] | None = None,
                 comment: str = "  ") -> list[str]:
    """The canonical block. The DEFINITION lines are byte-identical in both products.

    There is deliberately no "you are here" marker: a marker would make the two headers
    differ, and every panel in both products already carries its own ``[S]``/``[C]``/``[X]``
    book tag, which is the finer-grained answer anyway. ``product_book`` is validated and
    kept because a caller that cannot name its own book has no business printing this block.

    ``resolved`` maps a book key to a short live fact (leg counts, formation date) MEASURED
    on this run, so the header states the book that actually formed rather than the one that
    was supposed to. Those lines are the only ones that differ between the two outputs, and
    they are measurements rather than definition.
    """
    if product_book not in BOOK_BY_KEY:
        raise KeyError(f"unknown book {product_book!r}; known: {sorted(BOOK_BY_KEY)}")
    resolved = resolved or {}
    c = comment
    L = [f"{c}PROBLEM DEFINITION — canonical, identical in both PM-facing outputs"]
    L += _field("Monitored exposure",
                "US equity cross-sectional momentum, long winners / short losers, held as a "
                "book to be sized rather than traded as a signal.", c)
    L += _field("Books live",
                f"{len(BOOKS)} constructions, deliberately not unified:", c)
    for b in BOOKS:
        L += [f"{c}    [{b.tag}] {b.label}"]
        for k, val in (("universe", b.universe), ("sort", b.sort),
                       ("source", b.source), ("carries", b.carries)):
            L += _field(f"      {k:9}", val, c)
        if b.key in resolved:
            L += _field(f"      {'this run':9}", resolved[b.key], c)
    L += _field("Divergence", DIVERGENCE, c)
    L += _field("Reversal target", REVERSAL_TARGET, c)
    L += _field("Declared horizon",
                f"{DECLARED_HORIZON_DAYS} trading days at the {DECLARED_QUANTILE:.0%} quantile "
                "(the registered headline; the term structure at 1d and 21d is printed beside "
                "it, never instead of it)", c)
    L += _field("Intended user", INTENDED_USER, c)
    L += _field("Decision informed", DECISION, c)
    return L


def header_markdown(product_book: str, *, resolved: dict[str, str] | None = None) -> str:
    """The same block, wrapped for the X-ray's markdown."""
    body = "\n".join(header_lines(product_book, resolved=resolved, comment=""))
    return "## Problem definition (canonical)\n\n```\n" + body + "\n```"
