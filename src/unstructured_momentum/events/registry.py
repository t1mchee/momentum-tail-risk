"""Hand-curated registry of momentum drawdown episodes, with mechanism labels.

Why a hand registry when a detector exists
------------------------------------------
The detector in `events.episodes` finds episodes from returns. This registry records what
is *known* about them from the literature and the contemporaneous record: the trigger, the
mechanism, and the citation. The two are deliberately independent, and the Stage-0 gate is
that they agree where they should — and, importantly, **disagree where they should**.

The inverted gate
-----------------
The obvious gate ("the detector must recover Aug 2007") is wrong, and adopting it would
corrupt the registry. Aug 2007 sits at the **5.97th percentile** of modern 10-day WML moves
and clears no 1% or 2.5% threshold in any of twelve momentum constructions; it only appears
at a 5% cut alongside ~97 other "events". Loosening the detector until it appears would be
fitting the instrument to a prior.

Aug 2007 was a **joint multi-factor deleveraging**: averaging standardized 10-day moves
across wml/hml/smb/st_rev puts it at the 0.10th percentile, second worst of the modern era.
So the correct gate is two-sided:

    Aug 2007 must be ABSENT from the momentum-only detector
    Aug 2007 must be PRESENT in the joint-deleveraging channel

Passing both validates the two-channel taxonomy for free, at Stage 0.

Mechanism labels
----------------
``A_panic_rebound``  weak prior 2y market, high vol, market rallies during the drawdown.
                     Momentum loses through its conditional negative beta. This is the
                     regime Daniel-Moskowitz's panic-state model is built for.
``B_crowded``        benign prior market state, unremarkable vol, market ~flat during.
                     A positioning unwind or leadership rotation. DM is structurally blind
                     here — measured: before 2019-09-09 the panic-state model predicted
                     VaR5 of -3.48% against an unconditional -4.86%, i.e. LESS alarmed,
                     because bear=0. Realised was -5.03%.
``J_joint``          the whole long-short factor complex degrosses together; momentum is
                     collateral damage rather than the subject.
``mixed``            does not classify cleanly. Kept as ``mixed`` rather than forced — how
                     many land here is evidence about whether the taxonomy holds.

Labels are assigned from the documented trigger and contemporaneous market state, NOT from
the size of the momentum drawdown, so the label is independent of the thing being predicted.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Episode:
    key: str
    trough: dt.date
    mechanism: str
    trigger: str
    note: str
    citation: str = ""
    tier: str = ""  # design / validate / sealed, filled from config
    daily_panel: bool = False  # is stock-level daily holdings coverage available?
    #: Whether the MOMENTUM-only detector should surface this, stated explicitly rather
    #: than derived from `mechanism`. March 2020 is a joint deleveraging that ALSO
    #: registers marginally on momentum (1 of 12 constructions), so deriving the
    #: expectation from the label alone produces a false gate failure.
    expect_momentum_detect: bool = True
    #: Horizon (trading days) at which the episode registers. The event set is
    #: horizon-dependent: Nov 2022 is present at 1d (-11.5%) and 5d (-15.8%) and ABSENT
    #: at 10d and 21d because it partially reversed. A 10-day-only detector silently
    #: drops short-sharp catalyst events while keeping slower ones.
    detect_horizon: int = 10


#: Modern-era episodes. Deliberately not exhaustive: these are the ones with a documented
#: trigger in the literature or contemporaneous record. Undocumented detector hits stay in
#: the detector output and out of here.
EPISODES: tuple[Episode, ...] = (
    Episode(
        "1999_04_value_rotation", dt.date(1999, 4, 21), "B_crowded",
        "Cyclical/value rotation late in the dot-com melt-up",
        "Prior 2y market +55.7%; benign state, market roughly flat through the drawdown.",
    ),
    Episode(
        "2000_03_dotcom_peak", dt.date(2000, 3, 21), "mixed",
        "Nasdaq peak and initial unwind of growth leadership",
        "Prior 2y +30.1%. Straddles regimes; kept as mixed rather than forced.",
    ),
    Episode(
        "2001_01_fed_intermeeting", dt.date(2001, 1, 17), "B_crowded",
        "Unscheduled FOMC intermeeting rate cut, 3 Jan 2001",
        "Strongest measured Type B in the modern sample: -11.65 sigma vol-standardized, "
        "visible in ALL twelve momentum constructions, market +0.9% through the window, "
        "prior 2y only -3.4%. Beaten-down losers ripped on the surprise cut.",
        "Daniel & Moskowitz (2016) discuss optionality of the loser leg",
    ),
    Episode(
        "2001_04_rebound", dt.date(2001, 4, 19), "A_panic_rebound",
        "Sharp bear-market rally",
        "Prior 2y -22.8%, market +14.1% during. Canonical panic-rebound signature.",
    ),
    Episode(
        "2002_11_rebound", dt.date(2002, 11, 25), "A_panic_rebound",
        "Post-October-2002-low rally",
        "Prior 2y -40.9%, market +4.5% during.",
    ),
    Episode(
        "2007_08_quant_quake", dt.date(2007, 8, 9), "J_joint",
        "Simultaneous deleveraging of quantitative market-neutral books, 6-9 Aug 2007",
        "NOT a momentum event in daily factor data: 5.97th pctile momentum-only, clearing "
        "no 1%/2.5% threshold in any of 12 constructions. Joint channel puts it at the "
        "0.10th pctile (HML -6.53 sigma, WML -5.05, ST_Rev -4.52, SMB -3.00), 2nd worst "
        "modern joint move after March 2020. Round-tripped by 10 Aug.",
        "Khandani & Lo (2007, 2011); AQR 'The August of Our Discontent'",
        expect_momentum_detect=False,
    ),
    Episode(
        "2008_12_crisis", dt.date(2008, 12, 8), "A_panic_rebound",
        "Post-Lehman bear rally",
        "Prior 2y -44.7%, market +21.4% during.",
    ),
    Episode(
        "2009_03_canonical", dt.date(2009, 3, 23), "A_panic_rebound",
        "March 2009 bottom and violent low-quality rally",
        "The canonical momentum crash: prior 2y -51.6%, market +20.4% during, trailing "
        "vol 44.7%. Loser-decile beta above 3 on the Daniel-Moskowitz decomposition.",
        "Daniel & Moskowitz (2016), 'Momentum Crashes', JFE 122(2)",
    ),
    Episode(
        "2016_03_low_quality", dt.date(2016, 3, 7), "mixed",
        "Oil/commodity rebound driving a low-quality rally",
        "Prior 2y +5.9%. Tests the taxonomy's edge; does not classify cleanly.",
    ),
    Episode(
        "2019_09_momentum_unwind", dt.date(2019, 9, 11), "B_crowded",
        "Rate-driven value rotation, 9-11 Sept 2019",
        "THE MOTIVATING CASE. -6.03 sigma vol-standardized, worst in the modern sample, "
        "yet the JOINT channel reads +0.59 (a non-event) — exactly complementary to 2007. "
        "Market flat on 9-10 Sept (+0.07%, +0.16%) while the momentum residual fell "
        "-1.76% and -1.56% on 8-9bp rate moves. Book was a rate-sensitive quality/bond-"
        "proxy bet that GICS reported as 'diversified' (sector HHI 50th pctile). "
        "Panic-state model was LESS alarmed than unconditional (VaR5 -3.48% vs -4.86%) "
        "because bear=0; realised 10d was -5.03%. Winner-leg comomentum: 87th pctile.",
        "Lou & Polk (2022) comomentum; Barroso & Santa-Clara (2015)",
        daily_panel=True,
    ),
    Episode(
        "2020_03_covid", dt.date(2020, 3, 20), "J_joint",
        "COVID crash and the March 2020 liquidity event",
        "Worst joint deleveraging of the modern era: joint_z -7.65, with ST_Rev at "
        "-17.68 sigma. Momentum itself comparatively mild (wml +0.50 sigma) but it DOES "
        "register marginally on the momentum detector: -14.5% in 1 of 12 constructions. "
        "Joint and momentum labels are not mutually exclusive.",
        daily_panel=True, expect_momentum_detect=True,
    ),
    Episode(
        "2020_06_reopening", dt.date(2020, 6, 8), "mixed",
        "Reopening/low-quality rally",
        "Top-3 worst 10-day WML episode in the modern sample (-38.0% in the worst "
        "construction). Prior 2y +7.4%, market +9.8% during.",
        daily_panel=True,
    ),
    Episode(
        "2020_11_vaccine", dt.date(2020, 11, 9), "B_crowded",
        "Pfizer vaccine efficacy announcement, 9 Nov 2020",
        "Second-worst single day in a century of WML (-14.39%). A single scheduled-ish "
        "binary catalyst flipping leadership. Market-neutral tilt rate beta was at the "
        "5th pctile (bond-proxy-like) going in.",
        daily_panel=True,
    ),
    Episode(
        "2021_03_rates_rotation", dt.date(2021, 3, 5), "B_crowded",
        "Long-rate backup driving growth-to-value rotation",
        "Prior 2y +49.3%, market -2.8% during. Tilt rate beta at the 4.7th pctile.",
        daily_panel=True,
    ),
    Episode(
        "2022_11_cpi", dt.date(2022, 11, 10), "B_crowded",
        "Softer-than-expected October CPI print, 10 Nov 2022",
        "One of the ten worst single days in a century of WML (-6.68%). Macro-print "
        "catalyst; market-wide, so not book-specific. HORIZON-DEPENDENT: present at 1d "
        "(-11.5%) and 5d (-15.8%), ABSENT at 10d and 21d because it partially reversed "
        "(10d cum to 2022-11-23 was -5.81%). Registers at detect_horizon=5.",
        daily_panel=True, detect_horizon=5,
    ),
)


def frame() -> pd.DataFrame:
    """Registry as a dataframe, with tier assignment from config."""
    from ..config import tier_of

    rows = []
    for e in EPISODES:
        rows.append(
            {
                "key": e.key,
                "trough": pd.Timestamp(e.trough),
                "mechanism": e.mechanism,
                "trigger": e.trigger,
                "note": e.note,
                "citation": e.citation,
                "tier": tier_of(e.trough).value,
                "daily_panel": e.daily_panel,
                "expect_momentum_detect": e.expect_momentum_detect,
                "detect_horizon": e.detect_horizon,
            }
        )
    return pd.DataFrame(rows).sort_values("trough").reset_index(drop=True)


def gate_detector_agreement(
    detected: pd.DataFrame,
    *,
    tolerance_days: int = 21,
    momentum_only: bool = True,
) -> pd.DataFrame:
    """Stage-0 gate: does the detector recover the registry where it should?

    ``momentum_only=True`` means these are momentum-channel detections, in which case
    ``J_joint`` episodes are EXPECTED TO BE MISSING and their absence is a pass, not a
    failure. Set False when checking the joint channel, where the expectation inverts.
    """
    reg = frame()
    troughs = pd.DatetimeIndex(detected["trough_at"]) if len(detected) else pd.DatetimeIndex([])

    rows = []
    for _, e in reg.iterrows():
        if len(troughs):
            gap = min(abs((troughs - e["trough"]).days))
            found = bool(gap <= tolerance_days)
        else:
            gap, found = None, False

        expected = bool(e["expect_momentum_detect"]) if momentum_only else (
            e["mechanism"] == "J_joint"
        )
        rows.append(
            {
                "key": e["key"],
                "trough": e["trough"].date(),
                "mechanism": e["mechanism"],
                "found": found,
                "expected_found": expected,
                "gate_pass": found == expected,
                "detect_horizon": int(e["detect_horizon"]),
                "nearest_gap_days": gap,
            }
        )
    return pd.DataFrame(rows)
