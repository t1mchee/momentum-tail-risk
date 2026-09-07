"""Tests of whether text-adjacent information actually leads momentum reversals.

The evidence so far says financial news is *reactive*: the Sept 2019 evidence pack is
dominated by articles published after the unwind, and 82% of pre-event keyword matches
came from a single content farm. Before concluding that text has no forward-looking value,
two more defensible hypotheses deserve a proper test.

**H1 -- official-sector commentary leads.** Regulators write about crowding and leverage
before it breaks. The BIS Quarterly Review, IMF GFSR and Fed FSR are timestamped,
free, and long-running. The test is whether crowding language in an issue published
*before* an episode is elevated relative to its own history.

**H2 -- the catalyst calendar leads.** A reversal need not be forecast from narrative if
its trigger is on a schedule. Jan 2001 was an FOMC action; Nov 2022 was a CPI print. The
FOMC calendar is published a year ahead, so "days until the next decision" is genuinely
forward-looking with no leakage at all -- it is not a forecast, it is a fact about the
future.

The power problem, which dominates everything
----------------------------------------------
There are roughly 24 distinct modern-era reversal episodes. `minimum_detectable_lift`
makes the consequence concrete: against a 15% base rate at N=24, only an effect of about
**2.2x or larger** can reach p<0.05. Anything weaker is invisible no matter how good the
data or the method.

That is the binding constraint on this entire project, and it is worth stating before any
result rather than after. It is also the main reason the design avoids fitting a
flexible model: with 24 events, a model with more than a handful of parameters is
memorising, and a validation that reports only a point estimate is hiding its own
uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def near_dates(dates, calendar, window_days: int) -> np.ndarray:
    """Boolean: is each date within ``window_days`` of any calendar date?"""
    cal = pd.DatetimeIndex(calendar)
    return np.array(
        [bool((np.abs((cal - pd.Timestamp(d)).days) <= window_days).any()) for d in dates]
    )


def calendar_clustering(
    event_dates,
    calendar,
    all_trading_days,
    *,
    windows: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Do events cluster near a calendar more than chance?

    The base rate is computed on *every trading day in the sample*, not assumed. That
    matters: FOMC dates are frequent enough that a naive "were events near a meeting?"
    reading looks impressive at any window. The comparison is against how often a randomly
    chosen day is also near one.

    Reports an exact binomial p-value rather than a normal approximation, because at
    N~24 the approximation is poor exactly in the tail that decides the question.
    """
    rows = []
    for w in windows:
        base = float(near_dates(all_trading_days, calendar, w).mean())
        hits = int(near_dates(event_dates, calendar, w).sum())
        n = len(event_dates)
        p = stats.binomtest(hits, n, base, alternative="greater").pvalue if n else np.nan
        rows.append(
            {
                "window_days": w,
                "n_events": n,
                "hits": hits,
                "expected": n * base,
                "event_rate": hits / n if n else np.nan,
                "base_rate": base,
                "lift": (hits / n / base) if n and base else np.nan,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def minimum_detectable_lift(
    n_events: int, base_rate: float, *, alpha: float = 0.05
) -> dict:
    """Smallest effect this sample size could detect at ``alpha``.

    Reported alongside every null result. "We found no significant effect" is close to
    meaningless without it: at N=24 a genuine 1.6x effect would fail to reach significance
    almost always, so a null is uninformative about effects below the detection floor.
    """
    for k in range(0, n_events + 1):
        if stats.binomtest(k, n_events, base_rate, alternative="greater").pvalue < alpha:
            return {
                "n_events": n_events,
                "base_rate": base_rate,
                "min_hits": k,
                "min_event_rate": k / n_events,
                "min_lift": k / n_events / base_rate,
                "alpha": alpha,
            }
    return {
        "n_events": n_events,
        "base_rate": base_rate,
        "min_hits": None,
        "min_lift": np.inf,
        "alpha": alpha,
    }


def summarise(result: pd.DataFrame, mdl: dict) -> str:
    """Human-readable verdict that refuses to overclaim in either direction."""
    best = result.loc[result["p_value"].idxmin()]
    lines = [
        f"N = {int(best['n_events'])} events.",
        f"Best window +/-{int(best['window_days'])}d: {int(best['hits'])} hits vs "
        f"{best['expected']:.1f} expected, lift {best['lift']:.2f}x, p = {best['p_value']:.3f}.",
    ]
    if best["p_value"] < 0.05:
        lines.append("Significant at the 5% level.")
    else:
        lines.append(
            f"NOT significant. At this sample size a lift below {mdl['min_lift']:.1f}x "
            f"cannot reach p<{mdl['alpha']}, so this null does not rule out a real effect "
            f"of the observed size -- the test is underpowered, not the hypothesis refuted."
        )
    return " ".join(lines)


def grid_test(
    build_events,
    calendar,
    all_trading_days,
    *,
    horizons: tuple[int, ...] = (5, 10, 21),
    percentiles: tuple[float, ...] = (1.0, 2.5, 5.0),
    window_days: int = 3,
    min_events: int = 8,
) -> pd.DataFrame:
    """Repeat a clustering test across the whole event-definition grid.

    The single most dangerous way to run this test is once, on one threshold and one
    horizon, and report the result. Event definitions are researcher degrees of freedom:
    with three horizons and three thresholds there are nine ways to define "a reversal",
    and at alpha=0.05 roughly one of them is expected to look significant by chance alone.

    Reporting the whole grid makes that visible. A real effect shows up as a *consistently*
    elevated lift across cells; a spurious one shows up as one lucky cell surrounded by
    noise. ``build_events(horizon, percentile) -> DatetimeIndex``.
    """
    rows = []
    for hz in horizons:
        for pc in percentiles:
            ev = build_events(hz, pc)
            if len(ev) < min_events:
                continue
            r = calendar_clustering(ev, calendar, all_trading_days, windows=(window_days,))
            row = r.iloc[0].to_dict()
            row.update({"horizon": hz, "percentile": pc})
            rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        # Benjamini-Hochberg across the grid. Without it, "one cell reached p<0.05" reads
        # as evidence when it is the expected yield of nine tests.
        m = len(out)
        ranked = out.sort_values("p_value").reset_index(drop=True)
        ranked["bh_threshold"] = 0.05 * (ranked.index + 1) / m
        ranked["survives_bh"] = ranked["p_value"] <= ranked["bh_threshold"]
        out = ranked
    return out


def grid_verdict(grid: pd.DataFrame) -> str:
    """Interpret a grid honestly, weighting consistency over the best cell."""
    if not len(grid):
        return "No cell had enough events to test."
    med = grid["lift"].median()
    n_sig = int((grid["p_value"] < 0.05).sum())
    n_bh = int(grid["survives_bh"].sum())
    below_one = int((grid["lift"] < 1.0).sum())

    verdict = (
        f"{len(grid)} event definitions tested. Median lift {med:.2f}x; "
        f"{below_one} cells below 1.0x. {n_sig} cell(s) reached p<0.05 uncorrected, "
        f"{n_bh} survive Benjamini-Hochberg."
    )
    if n_bh == 0 and med < 1.3:
        verdict += (
            " The effect does not survive expansion of the event set: the lift is not "
            "consistently positive and the uncorrected hits are within what nine tests "
            "produce by chance. Treated as refuted rather than underpowered."
        )
    return verdict


def run_fomc_test(
    event_dates,
    all_trading_days,
    *,
    scheduled_only: bool = True,
) -> tuple[pd.DataFrame, dict, str]:
    """H2 end to end against the FOMC calendar.

    ``scheduled_only`` defaults to True because unscheduled intermeeting actions are
    surprises. Counting them would credit a *known calendar* with anticipating events that
    were not on it -- the calendar cannot have warned anyone about a decision nobody knew
    was coming.
    """
    from ..data import fedcal

    cal = fedcal.scheduled_only() if scheduled_only else fedcal.load()
    result = calendar_clustering(event_dates, cal["date"], all_trading_days)
    base = float(result.loc[result["p_value"].idxmin(), "base_rate"])
    mdl = minimum_detectable_lift(len(event_dates), base)
    return result, mdl, summarise(result, mdl)
