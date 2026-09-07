"""Render one date's run as the page a portfolio manager reads.

The could-not-measure block is not an appendix. It sits in the body, because the difference
between "measured and quiet" and "never measured" is the difference between a monitor and a
decoration, and only the brief can tell the reader which one they are holding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from . import contract
from .dates import GOLDEN, resolve
from .run import OUT, RunResult, run

RULE = "─" * 76


def _wrap(text: str, indent: str = "  ") -> list[str]:
    """Wrap a disclosure to the rule. A caveat that runs off the page is not a caveat."""
    import textwrap
    return [indent + ln for ln in textwrap.wrap(text, width=len(RULE) - len(indent))]


def render(res: RunResult) -> str:
    v, gaps = res.values, res.could_not_measure
    L = [RULE,
         f"MOMENTUM REVERSAL RISK — {res.as_of.date()}",
         f"generated {res.generated_at:%Y-%m-%d %H:%M} UTC   "
         f"coverage {res.coverage:.0%} of stages",
         RULE, ""]

    # Element 1, printed before any number, from the one module both products read. The live
    # facts under each book are measured on this run rather than asserted, so a header can
    # never describe a book that did not form.
    resolved: dict[str, str] = {}
    if "conferred" in v:
        c = v["conferred"]
        resolved["conferred_iwv"] = (f"{c['n_leg']} names per leg, formation window closing "
                                     f"{c['available_at']}")
    if "severity" in v and v["severity"].get("estimates"):
        # The DECLARED horizon's row, not whichever one happens to be first.
        e0 = next((e for e in v["severity"]["estimates"]
                   if e["horizon_days"] == contract.DECLARED_HORIZON_DAYS
                   and abs(e["quantile"] - contract.DECLARED_QUANTILE) < 1e-9),
                  v["severity"]["estimates"][0])
        resolved["french_wml"] = (f"{e0['n_train']:,} training observations at "
                                  f"{e0['horizon_days']}d, fitted through {res.as_of.date()}")
    if "named_bet" in v:
        nb = v["named_bet"]
        resolved["xray_month_end"] = (
            f"book formed {nb['book_as_of']}, read from the X-ray for {nb['xray_date']}"
            + (f" ({nb['stale_days']}d before this brief)" if nb["stale_days"] else ""))
    else:
        resolved["xray_month_end"] = (
            "could_not_measure: " + gaps.get("named_bet", "X-ray not read on this run"))
    L += contract.header_lines("french_wml", resolved=resolved) + [""]

    if "conferred" in v:
        c = v["conferred"]
        pct = c["beta_spread_pctile"]
        pct_s = f"{pct:.0%} percentile" if pct == pct else "percentile unavailable"
        posture = "DEFENSIVE" if c["beta_spread"] < 0 else "PRO-CYCLICAL"
        L += ["WHAT THE SORT JUST INHERITED   [C]",
              f"  Conferred market-beta spread   {c['beta_spread']:+.3f}   ({pct_s})",
              f"  Winner-leg beta / loser-leg    {c['beta_winner']:.3f} / {c['beta_loser']:.3f}",
              f"  Conferred 10y-change spread    {c['d10y_spread']:+.4f}",
              f"  Leg size                       {c['n_leg']} names",
              f"  Posture                        {posture}",
              f"  Estimated on the formation window closing {c['available_at']}; nothing here",
              "  uses data from after that date."]
        # The entitlement test, per date. Without it the block states a spread without saying
        # whether a book nobody sorted would have shown one.
        pbp, pbc = c.get("placebo_pctile"), c.get("placebo_clears")
        base = c.get("placebo_base_rate")
        if pbp is not None and pbp == pbp:
            L += [f"  Against 100 placebo books matched to this book's size profile, the sort's",
                  f"  spread sits at the {pbp:.0f}th percentile -- "
                  f"{'ABOVE' if pbc else 'NOT above'} the placebo 95th."]
        if base is not None and base == base:
            k, n = c.get("placebo_k"), c.get("placebo_n")
            ne, ci = c.get("placebo_neff"), c.get("placebo_ci") or [float("nan")] * 2
            head = (f"  Base rate: the sort clears its own placebo 95th in {k} of {n} months "
                    f"({base:.0%})" if k is not None
                    else f"  Base rate: the sort clears its own placebo 95th in {base:.0%}")
            L += [head]
            if ne == ne and ci[0] == ci[0]:
                L += [f"    effective sample {ne:.0f} of {n} after a lag-1 autocorrelation of "
                      f"{c.get('placebo_autocorr', float('nan')):.3f};",
                      f"    95% block interval [{ci[0]:.3f}, {ci[1]:.3f}]"]
            L += ["  so this is a regularity and not a constant. Months are adjacent and the",
                  "  indicator persists, so the effective sample is far below the month count",
                  "  and the naive binomial interval would be much too tight."]
        # A different WINDOW and a different estimator, printed under its own label rather than
        # blended with the line above.
        pbs, pn = c.get("post_beta_spread"), c.get("post_n_obs")
        if pbs is not None:
            L += [f"  Post-formation, 3m after the sort   {pbs:+.3f}   "
                  f"(median per-name beta, {pn} snapshots)"]
        elif c.get("post_unavailable"):
            L += [f"  Post-formation, 3m after the sort   could_not_measure: "
                  f"{c['post_unavailable']}"]
        # What the sort inherited, NAMED. This is the X-ray's read, on the X-ray's own book
        # [X], and the block says so rather than letting the reader assume it describes the
        # [C] legs above it. The LLM's own DECLINE is printed verbatim when it declines.
        L += ["", "  AND WHAT THE STORY IS   [X] — from the X-ray, a different book"]
        if "named_bet" in v:
            b = v["named_bet"]
            gate = "validity-gated" if b["gated"] else "NOT validity-gated (top ungated theme)"
            L += [f"    X-ray {b['xray_date']}"
                  + (f", {b['stale_days']}d before this brief" if b["stale_days"] else "")
                  + f"; book formed {b['book_as_of']}; "
                  f"{b['n_gated']} of {b['n_themes']} themes gated",
                  f"    Top theme      {b['n']} names, {b['side']}-side leg, {gate}",
                  "    Members        " + ", ".join(b["members"][:20])
                  + (" …" if b["n"] and b["n"] > 20 else ""),
                  f"    Story-beta     {b['story_beta']:+.3f}   "
                  f"(loading percentile {b['load_pct']} against within-book pseudo-themes)",
                  f"    Saturation     {b['saturation']:.1%} of the book's attention   "
                  f"(percentile {b['sat_pct']})"]
            if b.get("crowd_top10") is not None:
                L += [f"    13F crowding   top-10 filer share {b['crowd_top10']:.3f}   "
                      f"(percentile {b['crowd_pct']}, quarter {b['crowd_quarter']})"]
                for f_ in (b.get("top_filers") or [])[:3]:
                    L += [f"      {f_['rank']}. {f_['filer'][:44]:44} "
                          f"{f_['share']:5.1%} of theme 13F value  (CIK {f_['cik']})"]
                if b.get("top_filers"):
                    L += ["      names AS FILED, long-only, 45-day lag; who owns the story, "
                          "not evidence of crowding"]
            else:
                L += ["    13F crowding   could_not_measure: no 13F quarter available at "
                      "this date for this theme"]
            L += [f"    Rotation door  "
                  f"{'ARMED' if b['rotation_armed'] else 'quiet'}",
                  "    NAMED BET      " + (b["bet_name"] or "(no name returned)")]
            L += ["    Naming is an LLM annotation with a mandatory DECLINE option and is "
                  "never a number;",
                  "    it names the bet the sort built. It does not measure it."]
        else:
            L += [f"    could_not_measure: {gaps.get('named_bet', 'X-ray stage did not run')}"]
        L += [""]

    # The decision line, printed BEFORE the description of the state, because a page whose
    # first number is a diagnosis invites the reader to admire the diagnosis.
    if "sizing" in v:
        z = v["sizing"]
        L += ["HOW MUCH TO HOLD   [S] — the only line here that is a decision"]
        L += [f"  126d realised vol      {z['rv126_ann']:.1%} annualised",
              f"  Vol-scaled leverage    {z['leverage']:.2f}x  against a "
              f"{z['target_vol']:.0%} target"]
        if z.get("leverage_10d_ago"):
            d = z["leverage"] - z["leverage_10d_ago"]
            L += [f"  Change over 10d        {d:+.2f}x  (from "
                  f"{z['leverage_10d_ago']:.2f}x)"]
        L += [f"  Book return            21d {z['wml_21d']:+.2%}, 5d {z['wml_5d']:+.2%}",
              "  Barroso-Santa-Clara constant-volatility scaling. No conditioning and no",
              "  model: trailing volatility beat every elaborate alternative this project",
              "  tested against it, and a sizing rule is the honest product of that finding.",
              "  The X-ray computes the same quantity on the [X] book, which is the one that",
              "  actually gets sized; those numbers are NOT interchangeable with these.", ""]

    if "severity" in v:
        c = v["severity"]
        L += [f"HOW BAD IT COULD GET   [S] — declared horizon "
              f"{contract.DECLARED_HORIZON_DAYS}d at the "
              f"{contract.DECLARED_QUANTILE:.0%} quantile"]
        for e in c["estimates"]:
            L += [f"  {int(e['quantile']*100):>2}% VaR over {e['horizon_days']:>2}d  "
                  f"{e['var_conditional']:+.4f}   ES {e['es_conditional']:+.4f}   "
                  f"(unconditional VaR {e['var_unconditional']:+.4f})"]
        L += [f"  {c['equivalence']}"]

        # A conditional estimate BELOW its own unconditional baseline is the model saying the
        # state makes things safer. That is a real reading and sometimes right, but it is also
        # exactly what this model printed on 2019-09-06, six sessions before a reversal that
        # breached the conditional VaR. The state variables that drive the calm reading are
        # built for bear-market rebounds and are structurally blind to a crowded rotation from
        # a benign state. The page says so itself rather than leaving the reader to notice
        # that the smaller number is the more alarming one.
        _dec = next((e for e in c["estimates"]
                     if e["horizon_days"] == contract.DECLARED_HORIZON_DAYS
                     and abs(e["quantile"] - contract.DECLARED_QUANTILE) < 1e-9), None)
        if _dec and _dec["var_conditional"] > _dec["var_unconditional"]:
            L += ["", "  ⚠ LESS ALARMED THAN ITS OWN BASELINE. At the declared horizon the",
                  f"  conditional VaR {_dec['var_conditional']:+.4f} is milder than the",
                  f"  unconditional {_dec['var_unconditional']:+.4f}: the model is claiming this",
                  "  state is SAFER than an average day. Treat that as a claim to check, not a",
                  "  reassurance. The conditioning variables are built for bear-market rebounds",
                  "  and do not see a crowded rotation from a calm state; on 2019-09-06 this",
                  "  same line printed and the realised 5-day move was worse than the",
                  "  conditional expected shortfall."]
        cp = c.get("crash_probability")
        if cp:
            L += ["  Chance of a 10-day loss beyond the registered threshold, beside the base",
                  "  rate that threshold was defined to produce:"]
            for r in cp["rows"]:
                arrow = "BELOW" if r["conditional_prob"] < r["base_rate"] else "above"
                L += [f"    beyond {r['level']:+.4f}   {r['conditional_prob']:.3f}   "
                      f"{arrow} the {r['base_rate']:.3f} base rate"]
            L += [f"    thresholds frozen: {cp['vintage']}"]
        cov = c.get("coverage_by_state")
        if cov:
            L += ["  Measured coverage by state, 5% VaR at 10d, on this run's own backtest:"]
            for r in cov:
                L += [f"    {r['state']:>5}  breach rate {r['breach_rate']:.4f} against "
                      f"{r['expected']:.2f} expected, n {int(r['n']):,}  "
                      f"(uncond p {r['p_uc']:.3f}, independence p {r['p_ind']:.3f})"]
        for name, why in sorted(c.get("unavailable", {}).items()):
            L += [f"  {name:30} could_not_measure: {why}"]
        L += [""]

    if "drivers" in v:
        c = v["drivers"]
        L += ["WHAT THE ESTIMATE IS MADE OF   [S] Shapley, [C] component ES"]
        sh = c.get("shapley")
        if sh:
            L += [f"  Shapley over model terms, exhaustive over {sh['n_subsets']} subsets; "
                  f"baseline {sh['baseline']:+.4f} -> fitted {sh['full']:+.4f}"]
            for t_ in sh["terms"]:
                L += [f"    {t_['term']:22} {t_['shapley']:+.5f}   "
                      f"(term value now {t_['value_now']:+.4f})"]
            L += [f"    contributions reconstruct the fit to "
                  f"{sh['reconstruction_error']:.1e}"]
        ce = c.get("component_es")
        if ce:
            L += [f"  Names carrying the tail: book ES {ce['book_es']:+.4f} over "
                  f"{ce['n_tail_days']} worst days of {ce['n_names']} constituents"]
            L += ["    " + ", ".join(f"{x['ticker']} {x['contribution']:+.5f}"
                                     for x in ce["top"][:10])]
        cats = c.get("catalysts")
        if cats is not None:
            L += [f"  Scheduled in the next 30 days: {len(cats) or 'none'}"]
            for e in cats:
                L += [f"    {e['date']}  {e['kind']}  -> {e['driver']}"]
        for name, why in sorted(c.get("unavailable", {}).items()):
            L += [f"  {name:30} could_not_measure: {why}"]
        L += [""]

    if "crowding" in v:
        c = v["crowding"]
        # No single book tag is truthful here: the four components cover different universes
        # and different parts of the book, which is why they are never blended into one
        # crowding number. Each carries its own scope on its own line.
        L += ["POSITIONING   [mixed — components differ in universe and coverage; "
              "never blended]"]
        k = c["components"].get("comomentum")
        if k:
            L += [f"  Comomentum, winner-minus-loser  {k['spread']:+.4f}   "
                  f"({k['pctile_spread']:.0%} of {k['n_history']} observations)",
                  f"    {k['scope']}"]
        k = c["components"].get("short_volume")
        if k:
            L += [f"  Short volume share              {k['ratio']:.3f}   "
                  f"({k['change']:+.3f} over 21 sessions)"]
        k = c["components"].get("mtum_flow")
        if k:
            L += [f"  MTUM shares outstanding         {k['shares_outstanding']:,.0f}   "
                  f"({k['pct_change_21d']:+.1%} over 21 sessions)"]
        k = c["components"].get("skew")
        if k:
            L += [f"  Basket 30d implied vol          {k['basket_iv30']:.3f}   "
                  f"implied correlation {k['implied_correlation']:.3f}",
                  f"    {k['n_captured']} captured / {k['n_skipped']} skipped, "
                  f"{k['n_losers_captured']} on the loser leg; capture began {k['capture_start']}"]
        for name, why in sorted(c["unavailable"].items()):
            L += [f"  {name:30} could_not_measure: {why}"]
        L += [""]

    if "odds_conditioning" in v:
        c = v["odds_conditioning"]
        L += ["WHAT THE PUBLISHED CONDITIONERS SAY ABOUT THE ODDS   [S]",
              "  Indicator                   state, point-in-time     historical crash-",
              "                                                      frequency ratio"]
        for r in c["rows"]:
            passed = "survived our attack" if r["survives"] else "FAILED our attack"
            ratio = f"{r['ratio']:.2f}x" if r.get("ratio") is not None else "n/a"
            L += [f"  {r['indicator']:26}  {r['state']:22}  {ratio:>7}   ({passed})",
                  f"    value {r['value']:+.4f} on {r['as_of_row']}, "
                  f"{r['n_history']:,} observations to date"
                  + (f", {r['stale_days']}d stale" if r["stale_days"] else "")
                  + (f"; surrogate rank {r['placebo_rank']}" if r.get("placebo_rank")
                     is not None else ""),
                  f"    {r['source']}"]
        for name, why in sorted(c["unavailable"].items()):
            L += [f"  {name:26}  could_not_measure: {why}"]
        L += _wrap(c["disclosure"], "  ") + _wrap(c["recipe_note"], "  ") + [""]

    if "channels" in v:
        c = v["channels"]
        L += ["WHICH MECHANISM THE STATE LOOKS LIKE"]
        for k in ("wml_asymmetry", "loser_beta_dn", "joint_z", "factor_dispersion",
                  "eig_concentration"):
            if k in c["values"]:
                L += [f"  {k.replace('_', ' '):28} {c['values'][k]:+.4f}   "
                      f"({c['percentiles'][k]:.0%} of history to date)"]
        L += [f"  measured on {c['as_of_row']} from {c['n_history']:,} days of history"
              + (f", {c['stale_days']}d stale" if c["stale_days"] else ""),
              f"  {c['scope']}", ""]

    if "composition" in v:
        c = v["composition"]
        L += ["WHAT THE BOOK IS MADE OF",
              f"  Largest named group   {c['largest_group']}",
              f"  Stability             {c['agreement']}",
              f"  Size                  median {c['median_names']:.0f} names across "
              f"{c['median_sectors']:.0f} GICS sectors",
              f"  From the book formed  {c['from_book']} ({c['book_age_days']}d before this brief)",
              "  Beats a sector- and size-matched control; roughly three fifths of the",
              "  partition's structure is still sector.", ""]

    if "translation" in v:
        t = v["translation"]
        L += ["WHAT IT MEANS   [C] — the operational read of the CONFERRED tilt.",
              "  This is NOT the X-ray's named bet, which is a different book and is",
              "  reported above; if that stage did not run, this block does not replace it.",
              f"  Driver          {t['driver']}   (confidence {t['confidence']}/10)",
              f"  Transmission    {t['transmission']}",
              f"  Hedge           {t['hedge']}",
              f"                  {t['hedge_rationale']}"]
        for c_ in t["catalysts"][:4]:
            L.append(f"  Catalyst        {c_}")
        L += ["", "  FALSIFIER — logged now, scored mechanically later",
              f"    {t['falsifier']}",
              f"    {t['falsifier_refutes']}", ""]

    if gaps:
        L += ["COULD NOT MEASURE", ""]
        for k, why in gaps.items():
            L.append(f"  {k:14s} {why}")
        L += ["", "  These are gaps, not zeros. A quiet reading from a stage that did not run",
              "  is not a quiet market.", ""]
    L.append(RULE)
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="brief")
    p.add_argument("--as-of", default="today")
    p.add_argument("--golden", action="store_true", help="run the whole regression set")
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--write", action="store_true", help="also write to data/processed/briefs")
    a = p.parse_args(argv)

    dates = ([resolve(g) for g in GOLDEN] if a.golden
             else [pd.Timestamp(dt.date.today() if a.as_of == "today" else a.as_of)])
    worst = 0
    for d in dates:
        res = run(d, translate=not a.no_translate)
        text = render(res)
        print(text)
        if a.write:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"brief_{d.date()}.txt").write_text(text)
        if not res.values:
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main())
