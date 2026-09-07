"""The regression set: six dates the system must produce a brief for, and why each is there.

These are not a sample. They are chosen so that a change breaking any ONE of them breaks
something specific and nameable, which is what makes them useful as a regression test rather
than as evidence. Five are drawn from the project's own episode catalogue and its documented
books; the sixth is today, because a monitor that has only ever run on history is a backtest.

Every date is constrained to a year the holdings panel can actually serve. The panel is
monthly before 2013 and carries gaps in 2017 and 2018, so a date outside the usable set would
fail for a data reason and teach nothing about the pipeline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GoldenDate:
    as_of: str
    label: str
    why: str
    #: What a failure here would mean, so a red test is diagnostic rather than just red.
    breaks_if: str


GOLDEN: tuple[GoldenDate, ...] = (
    GoldenDate(
        "2019-09-06", "the rate reversal",
        "The motivating episode. A -11.7% drawdown peaking 2019-08-27, on a book the project "
        "documents as a rates bet wearing ten sector labels.",
        "the conferred tilt, the naming stage, or the translation layer stops reproducing the "
        "one case where the answer is independently known"),
    GoldenDate(
        "2021-01-29", "the squeeze",
        "A different mechanism entirely: the loser leg carried the squeeze cohort, so the "
        "book's fragility was on the short side rather than in a shared macro exposure.",
        "the system has learned only the rates story and generalises it to books that do not "
        "carry one"),
    GoldenDate(
        "2020-05-22", "the deepest drawdown on record here",
        "-22.3% over ten days, the largest in the catalogue within servable years. A rebound "
        "episode rather than a crowded rotation.",
        "the asymmetry channel stops separating a panic rebound from a positioning unwind"),
    GoldenDate(
        "2023-01-19", "a recent episode",
        "-12.9%, well after every design decision in this project was made, so it functions "
        "as an out-of-design check rather than a case anything was tuned on.",
        "the pipeline has been fitted to the episodes it was built alongside"),
    GoldenDate(
        "2015-06-30", "a calm control",
        "No episode. The brief must say so plainly and produce a low reading rather than "
        "manufacturing a narrative from an unremarkable book.",
        "the system finds a story everywhere, which is the failure mode a model in the loop "
        "makes most likely"),
    GoldenDate(
        "TODAY", "live",
        "Whatever today is. A monitor that has only ever run on history is a backtest, and "
        "the could-not-measure blocks shrinking across successive live briefs is the most "
        "honest evidence of monitoring available.",
        "the system works only on dates whose outcome is already known"),
)


def resolve(g: GoldenDate, today: dt.date | None = None) -> pd.Timestamp:
    """TODAY resolves at call time; every other date is fixed."""
    if g.as_of == "TODAY":
        return pd.Timestamp(today or dt.date.today())
    return pd.Timestamp(g.as_of)


def all_dates(today: dt.date | None = None) -> list[pd.Timestamp]:
    return [resolve(g, today) for g in GOLDEN]
