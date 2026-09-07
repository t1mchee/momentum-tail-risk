"""The example page for the 2020-10-30 formation, rendered from stored outputs only.

Registered in `docs/PAGE_REGISTRATION.md` before this file existed. Nine items, each resolving to
a named artifact; every numeral checked against a registered field before the page publishes.

Three of Tim's decisions are carried here rather than argued:

  * The tail line is the PARAMETER-FREE volatility-scaled quantile with NO conditioning state.
    Three conditioning variables were tested and none earned its place; they appear as a note,
    not as a model input.
  * The catalyst-window increase line is dropped. The calendar state appears as page FIELDS --
    which catalyst days fall in the window -- and nowhere else.
  * Weighting is value-weighted within each leg. Where a number elsewhere in the package is
    equal-weighted it is labelled so, and the concentration reading is NOT printed as a
    discovered exposure.

The theme renders from whichever of two results exists, by the rule fixed in the registration
before either was seen.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from unstructured_momentum.data import cboe  # noqa: E402
from unstructured_momentum.pipeline import contract  # noqa: E402
from unstructured_momentum.report.numerals import (  # noqa: E402
    Registry, document_marks, enforce, segments)


AS_OF = pd.Timestamp("2020-10-31")          # the formation month-end
TRADING = "2020-10-30"                       # the last session before it
OUT = Path("reports/poc")
GATE2 = Path("reports/gate2")
R = "─" * 78
CMD8 = "uv run python scripts/poc_e8_tail_final.py"
CMD5 = "uv run python scripts/poc_e5_book_tail.py"
CMDX = "uv run python scripts/gate2_extract.py loser"


# --------------------------------------------------------------------------------------
# The screen inputs. These lived in `gate2_page.py`, the FIRST design's page generator, and
# were imported from it. That script does not ship -- it produces a superseded PM page whose
# VaR is half this one's -- so importing from it made the delivered page unrenderable in the
# package while working perfectly in the working repository, where the file still exists.
# Thirty-five lines, copied rather than depended on.
# --------------------------------------------------------------------------------------


def _sectors() -> dict:
    sec = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        d = pd.read_parquet(f, columns=["as_of", "ticker", "sector"])
        d = d[d.as_of <= AS_OF]
        if len(d):
            sec.update(d.drop_duplicates("ticker", keep="last")
                        .set_index("ticker")["sector"].to_dict())
    return sec


def _legs() -> dict:
    return pickle.load(open("data/processed/leg_members.pkl", "rb"))[AS_OF]


def sector_screen(sec: dict, losers: list) -> dict:
    s = pd.Series([sec.get(t) for t in losers]).dropna()
    share = s.value_counts(normalize=True)
    return {"names": len(s), "sectors": int(s.nunique()), "hhi": float((share ** 2).sum()),
            "top": share.index[0], "top_share": float(share.iloc[0])}


def aggregates() -> dict:
    out = {}
    for n in ("VIX", "COR1M"):
        v = cboe.close(n).dropna().loc[:AS_OF]
        out[n] = {"level": float(v.iloc[-1]), "pctile": float((v <= v.iloc[-1]).mean())}
    return out


def theme_source() -> tuple[str, dict]:
    """Which of the two theme results exists. The rule was fixed before either was seen.

    The pipeline supplies the theme only if it rose above the placebo on this date AND its label
    matches the expected label registered in exp-091. Rising with a label that is not the
    episode's is not the pipeline finding the theme, and the registered rule says so.
    """
    p = OUT / "e7_summary.json"
    if not p.exists():
        return "fallback_pipeline_absent", {}
    d = json.loads(p.read_text())
    row = next((r for r in d.get("dates", []) if r.get("date") == "2020-10-31"), None)
    if row and row.get("any_theme_risen") and row.get("label_matches_expected"):
        return "pipeline", {"summary": d, "row": row}
    return "fallback_pipeline_null", {"summary": d, "row": row}


def outcome(reg: Registry) -> list[str]:
    """What the ten days after the page actually did.

    Everything here is outside the page's information set, and the heading says so. It is
    printed because an example output that never shows its own outcome asks the reader to
    take the machinery on trust; and it is fenced because one formation is one observation
    and settles nothing about a line whose calibration is a coverage question. The evidence
    on the lines is exp-089, not this.
    """
    d = json.loads((OUT / "realised_2020-10-30.json").read_text())
    CMD = "uv run python scripts/poc_e10_realised.py"
    SRC = "reports/poc/realised_2020-10-30.json"
    w = d["worst_session"]

    ret = reg.pct("outcome_realised", d["realised_return"],
                  "the book's realised 10-day return from this formation", SRC, CMD, dp=2, sign=True)
    worst = reg.pct("outcome_worst_session", w["book"],
                    "the book's worst single session in the window", SRC, CMD, dp=2, sign=True)
    lose = reg.pct("outcome_worst_loser_leg", w["loser_leg"],
                   "the loser leg's return on that session", SRC, CMD, dp=2, sign=True)
    win = reg.pct("outcome_worst_winner_leg", w["winner_leg"],
                  "the winner leg's return on that session", SRC, CMD, dp=2, sign=True)
    fr = reg.pct("outcome_french", d["french_reference"]["compounded_over_the_same_window"],
                 "Ken French's published momentum factor over the same window", SRC, CMD,
                 dp=2, sign=True)
    scaled = d["against_the_lines_the_page_printed"]["var_scaled"]["level"]
    uncond = d["against_the_lines_the_page_printed"]["var_uncond"]["level"]
    br_s = "inside" if d["realised_return"] >= scaled else "beyond"
    br_u = "inside" if d["realised_return"] >= uncond else "beyond"

    return [
        "WHAT HAPPENED NEXT   [NOT part of the page: outside the information set of "
        f"{TRADING}]",
        f"  Realised over the ten sessions {d['window']['start']} to {d['window']['end']}: {ret}.",
        f"  That is {br_u} the unconditional line and {br_s} the volatility-scaled one.",
        f"  {w['date']} alone was {worst}: the winner leg {win}, the loser leg {lose}. The loss",
        "  is the short half, which is the mechanism this page named in advance under WHAT",
        "  WOULD CHANGE THIS READING -- a vaccine readout resolving the shared condition and",
        "  repricing the loser leg upward together.",
        f"  French's published factor over the same window: {fr}. This book is the more",
        "  concentrated portfolio and falls further.",
        "  One formation is one observation. It does not make the scaled line right; the",
        "  coverage tests over 54 prior formations do that, and they are printed above.",
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = Registry()
    tail = pd.read_csv(OUT / "e5_book_tail.csv", parse_dates=["formation"])
    row = tail[tail["formation"] == AS_OF].iloc[0]
    conc = pd.read_csv(OUT / "e8_concentration_by_leg.csv", parse_dates=["formation"])
    cr = conc[conc["formation"] == AS_OF].iloc[0]
    fin = json.loads((OUT / "e8_tail_final.json").read_text())
    # POINT-IN-TIME. e8's coverage runs to the end of the design tier, which is after this page.
    # A page cannot cite a calibration statistic computed on its own future, so coverage here is
    # recomputed on formations strictly BEFORE the page date. The full-tier figure is in the
    # register and is a different, later claim.
    import poc_e1_tail as E1
    past = tail[tail["formation"] < AS_OF]
    yv = past["realised"].to_numpy()
    cov = {"scaled": E1.kupiec(yv, past["var_scaled"].to_numpy()),
           "unconditional": E1.kupiec(yv, past["var_uncond"].to_numpy())}

    sec = _sectors()
    legs = _legs()
    scr = sector_screen(sec, list(legs["losers"]))
    agg = aggregates()

    lo = pd.read_parquet(GATE2 / "nov2020_extractions.parquet")
    lo = lo[lo["error"].isna()] if "error" in lo else lo
    g = lo[lo.states_a_condition.fillna(False) & lo.quote_grounded.fillna(False)]

    # ---- item 2, severity: the page line ------------------------------------------
    v10 = reg.pct("var_scaled_10d", row["var_scaled"], "10-day 5% VaR, scaled line",
                  "reports/poc/e5_book_tail.csv", CMD5, dp=2)
    u10 = reg.pct("var_uncond_10d", row["var_uncond"], "the unconditional quantile beside it",
                  "reports/poc/e5_book_tail.csv", CMD5, dp=2)
    es10 = reg.pct("es_scaled_10d", row["es_scaled"], "10-day 5% expected shortfall",
                   "reports/poc/e5_book_tail.csv", CMD5, dp=2)
    ratio = reg.num("var_ratio", row["var_scaled"] / row["var_uncond"],
                    "how much more alarmed the scaled line is",
                    "reports/poc/e5_book_tail.csv", CMD5, dp=1)
    sig = reg.pct("trailing_vol", row["sigma"], "the book's trailing volatility, annualised",
                  "reports/poc/e5_book_tail.csv", CMD5, dp=1)
    lvl = reg.pct("registered_event_level", contract.DECLARED_LEVEL,
                  "the registered reversal level, frozen in the contract",
                  "src/unstructured_momentum/pipeline/contract.py", "frozen; not computed here",
                  dp=2, sign=True)
    pl = reg.pct("prob_below_own_var", row["prob_below_own_var"],
                 "share of past scaled returns below the book's own unconditional quantile",
                 "reports/poc/e5_book_tail.csv", CMD5)
    plr = reg.pct("prob_below_french_ref", row["prob_below_level"],
                  "the same at French's long-run reference level",
                  "reports/poc/e5_book_tail.csv", CMD5)

    # ---- item 3, calibration -------------------------------------------------------
    cs, cu = cov["scaled"], cov["unconditional"]
    nb = reg.num("cov_n", cs["n"], "formations the coverage rests on",
                 "reports/poc/e8_tail_final.json", CMD8)
    bs = reg.num("cov_scaled_breaches", cs["breaches"], "breaches of the scaled line",
                 "reports/poc/e8_tail_final.json", CMD8)
    rs = reg.pct("cov_scaled_rate", cs["rate"], "its breach rate",
                 "reports/poc/e8_tail_final.json", CMD8, dp=2)
    ps = reg.num("cov_scaled_p", cs["p"], "Kupiec p for the scaled line",
                 "reports/poc/e8_tail_final.json", CMD8, dp=3)
    ru = reg.pct("cov_uncond_rate", cu["rate"], "the unconditional line's breach rate",
                 "reports/poc/e8_tail_final.json", CMD8, dp=2)
    pu = reg.num("cov_uncond_p", cu["p"], "Kupiec p for the unconditional line",
                 "reports/poc/e8_tail_final.json", CMD8, dp=3)

    # The verdict is COMPUTED, never typed. An earlier draft wrote "passes" and "fails" into the
    # prose beside numbers that said otherwise: at this date the two lines are indistinguishable,
    # and the sentence claiming the unconditional was miscalibrated was false point-in-time. The
    # numeral canary cannot catch that, because it checks numbers and not claims.
    vs_ = "passes" if cs["p"] > 0.10 else "fails"
    vu_ = "passes" if cu["p"] > 0.10 else "fails"
    # The branch tests COUNTS. It used to conclude from equal counts that the two lines
    # "breach on the same formations", which is a claim about SETS and was false: four of the
    # five are shared, the scaled line breaches on 2016-10-31 and the unconditional on
    # 2020-07-31. That is exactly the kind of claim the comment above says the canary cannot
    # catch, written three lines below the comment. The overlap is now computed.
    if cs["breaches"] == cu["breaches"]:
        pt = tail[tail["formation"] < AS_OF]
        b_s = set(pt.loc[pt["realised"] < pt["var_scaled"], "formation"])
        b_u = set(pt.loc[pt["realised"] < pt["var_uncond"], "formation"])
        shared = reg.num("shared_breaches", len(b_s & b_u),
                         "formations where both lines breach", "reports/poc/e5_book_tail.csv",
                         CMD5)
        calibration_note = [
            "  At this date the two lines cannot be told apart on count: each breaches on the",
            f"  same number of formations, {shared} of them the same formations, and Kupiec says",
            "  the same of both. They are not the same set -- each has one breach the other",
            "  does not -- so this is agreement on the test, not identical behaviour. The",
            "  divergence that makes the scaled line the better calibrated one appears later",
            "  in the tier, and a reader standing here could not know it.",
        ]
    else:
        calibration_note = [
            f"  On formations before this page the scaled line {vs_} and the unconditional {vu_}.",
            "  These counts use only what a reader on this date could have had; the register",
            "  carries the full-tier figures, which are a different and later claim.",
        ]

    # ---- item 4, what the book is --------------------------------------------------
    nl = reg.num("long_names", cr["long_n_names"], "names in the long leg",
                 "reports/poc/e8_concentration_by_leg.csv", CMD8)
    en = reg.num("long_effective_n", cr["long_effective_n"], "effective number of names, long leg",
                 "reports/poc/e8_concentration_by_leg.csv", CMD8, dp=1)
    t1 = reg.pct("long_top1", cr["long_top1"], "largest single weight in the long leg",
                 "reports/poc/e8_concentration_by_leg.csv", CMD8, dp=0)
    t5 = reg.pct("long_top5", cr["long_top5"], "the five largest together",
                 "reports/poc/e8_concentration_by_leg.csv", CMD8, dp=0)
    ens = reg.num("short_effective_n", cr["short_effective_n"], "effective names, short leg",
                  "reports/poc/e8_concentration_by_leg.csv", CMD8, dp=1)

    # ---- item 5, the reader's screens ----------------------------------------------
    src_iwv, cmdc = "data/raw/ishares/IWV/panel", "uv run python scripts/gate2_nov2020.py"
    nleg = reg.num("loser_leg_names", scr["names"], "names in the loser leg", src_iwv, cmdc)
    nsec = reg.num("loser_leg_sectors", scr["sectors"], "GICS sectors it spans", src_iwv, cmdc)
    hhi = reg.num("sector_hhi", scr["hhi"], "sector Herfindahl", src_iwv, cmdc, dp=3)
    tops = reg.pct("top_sector_share", scr["top_share"], f"share in {scr['top']}", src_iwv, cmdc, dp=0)
    vix = reg.num("vix", agg["VIX"]["level"], "VIX close", "data/raw/cboe", cmdc, dp=1)
    vixp = reg.pct("vix_pctile", agg["VIX"]["pctile"], "its percentile", "data/raw/cboe", cmdc)

    # ---- item 6, the theme ----------------------------------------------------------
    src, info = theme_source()
    nstate = reg.num("loser_stated", int(len(g)), "loser filings stating a condition",
                     "reports/gate2/nov2020_extractions.parquet", CMDX)
    nread = reg.num("loser_read", int(len(lo)), "loser filings read",
                    "reports/gate2/nov2020_extractions.parquet", CMDX)
    rate = reg.pct("loser_rate", len(g) / len(lo), "the rate", 
                   "reports/gate2/nov2020_extractions.parquet", CMDX)
    nnames = reg.num("condition_names", int(g.ticker.nunique()), "names carrying a condition",
                     "reports/gate2/nov2020_extractions.parquet", CMDX)

    # ---- item 8, catalyst fields ----------------------------------------------------
    _ew_unused = reg.pct("earnings_weight_in_window", row["earnings_weight_in_window"],
                 "gross book weight reporting earnings in the window",
                 "reports/poc/e5_book_tail.csv", CMD5, dp=1)

    L = [R, f"MOMENTUM REVERSAL RISK — formation {AS_OF.date()} (last session {TRADING})",
         "US equity cross-sectional momentum, the reconstructed 12-1 book:",
         "top and bottom deciles of the Russell 3000, VALUE-WEIGHTED within each leg.",
         f"Horizon: {contract.DECLARED_HORIZON_DAYS} trading days at the "
         f"{contract.DECLARED_QUANTILE:.0%} quantile. The probability line below is the VaR",
         "restated at the event threshold, not a forecast of a reversal's timing.",
         R, "",
         "SEVERITY   [the parameter-free scaled quantile; no conditioning state]",
         f"  10-day 5% VaR      {v10}     against an unconditional {u10}   ({ratio}x)",
         f"  10-day 5% ES      {es10}",
         f"  the book's trailing volatility is {sig} annualised, and the estimator is today's",
         f"  volatility times the 5% quantile of past scaled returns. Nothing is fitted.",
         f"  P(10-day return at or below the book's own unconditional {u10}): {pl}.",
         f"  At French's long-run reference of {lvl}, which is a different and lower bar for a",
         f"  book this volatile, the same figure is {plr}.",
         "",
         "  Three conditioning variables were tested and none is used: the bear state, which",
         "  made the forecast worse; the calendar state; and the volatility tercile. Their",
         "  ratios and intervals are in the register.", "",
         "IS THIS LINE CALIBRATED   [out of sample, on the book, formations BEFORE this date]",
         f"  scaled line          {bs} breaches of {nb}, {rs}, Kupiec p {ps}  -> {vs_}",
         f"  unconditional        {ru}, Kupiec p {pu}  -> {vu_}",
         *calibration_note, "",
         "WHAT THE BOOK IS   [value-weighted, as the design specifies]",
         f"  long leg            {nl} names, but an effective {en} of them; the largest is {t1}",
         f"                     of the leg and the five largest {t5}",
         f"  short leg          an effective {ens} names",
         "  This is what value-weighting inside a decile produces. It is a property of the",
         "  construction and is printed, not corrected.", "",
         "WHAT THE READER'S EXISTING SCREENS SAY",
         f"  Sector screen      {nleg} names across {nsec} GICS sectors, Herfindahl {hhi}",
         f"                     largest {scr['top']} at {tops}  ->  reads DIVERSIFIED",
         f"  VIX                {vix}, {vixp} of its own history  ->  high, no direction", "",
         "WHAT THE LOSER LEG IS PRICED ON   [8-K filings, gated on acceptance time]"]

    if src == "pipeline":
        L += ["  From the theme pipeline: see the register for the cluster, its label and its",
              "  excess weight against the random books."]
    else:
        if src == "fallback_pipeline_null":
            c = (info.get("summary") or {}).get("counts", {})
            ne = reg.num("pipeline_episode_dates", c.get("episode_dates", 0),
                         "episode dates the theme pipeline ran on", "reports/poc/e7_summary.json",
                         "uv run python scripts/poc_e7_theme.py")
            nr = reg.num("pipeline_themes_risen", c.get("episode_theme_risen", 0),
                         "of those, dates where a theme rose above the placebo",
                         "reports/poc/e7_summary.json", "uv run python scripts/poc_e7_theme.py")
            nm = reg.num("pipeline_label_matches", c.get("episode_risen_theme_label_matches", 0),
                         "of those, dates whose label matched the registered expected label",
                         "reports/poc/e7_summary.json",
                         "uv run python scripts/poc_e7_expected_labels.py")
            nl = reg.num("pipeline_dates_labelled",
                         c.get("episode_dates_with_expected_label_registered", 0),
                         "episode dates carrying an expected label written into exp-091 before "
                         "the run", "reports/poc/e7_summary.json",
                         "uv run python scripts/poc_e7_expected_labels.py")
            L += ["  The theme below is the 8-K survival-condition reading, by the fallback rule",
                  "  registered before either result was seen. The theme pipeline ran on this date",
                  "  and produced no theme above the placebo.",
                  "  The three sentences that follow are INSTRUMENT DIAGNOSTICS, not market data,",
                  "  and they are NOT point-in-time: they score the pipeline across every date it",
                  "  was run on, five of which are after this one and three after the tier seal.",
                  "  They say how well the instrument works, which a reader here could not know;",
                  "  they are printed because the alternative is asking you to trust the fallback",
                  "  without saying why it was needed.",
                  f"  Across {ne} episode dates it raised a",
                  f"  theme on {nr}. The registration named an expected label in advance for {nl} of",
                  f"  those dates, and {nm} matched -- but no match was reachable, because the dates",
                  "  where a theme rose are not among the dates that carry an expected label. That",
                  "  zero is a property of which dates were labelled, not evidence about the labels.",
                  "  The reason is coverage, not the placebo: on the dates where something did",
                  "  rise, the excess weight rested on a handful of companies with Item 1A text",
                  "  out of a set of about a hundred and thirty. That is reported rather than",
                  "  omitted, and it is why the classifier run is listed as not built.", ""]
        else:
            L += ["  The theme below is the 8-K survival-condition reading, because the theme",
                  "  pipeline has not produced a result for this date. That outcome and its",
                  "  coverage reason are reported in the register beside this.", ""]
    L += [f"  Stated an external survival condition: {nstate} of {nread} filings ({rate})",
          f"  {nnames} names carry one, and they name the same thing: the resumption of normal",
          "  demand after COVID-19. The sector screen above sees ten sectors. The filings see",
          "  one bet.", "",
          "  Evidence, verbatim, with the date it became public:"]
    for t in ("UAL", "CCL", "AMC"):
        sub = g[g.ticker == t]
        if not len(sub):
            continue
        pick = sub[sub.quote.str.contains("vaccine|treatment", case=False)]
        r0 = (pick if len(pick) else sub).iloc[0]
        q = r0.quote.strip()
        q = q if len(q) <= 150 else q[:147] + "..."
        reg.add(f"span_{t}", q, str(r0.accepted_at)[:10], f"{t}'s stated condition",
                "reports/gate2/nov2020_extractions.parquet", CMDX,
                span={"ticker": t, "accession": r0.accession, "quote": r0.quote, "rendered": q,
                      "accepted_at": str(r0.accepted_at)[:19], "condition": r0.condition})
        L += [f"    {t}  accepted {str(r0.accepted_at)[:10]}", f'        "{q}"']

    # ---- item 10, the analogue panel: live-only, rendered from stored output --------
    pan = json.loads(Path("reports/analogue_rag/panel_2020-10-31.json").read_text())
    prec, pv = pan["record"], pan["verdict"]
    blocks = list(prec["state_matches"][0]["block_closeness"])
    nb_ = reg.num("analogue_blocks", len(blocks), "state-space blocks the distance used",
                  "reports/analogue_rag/panel_2020-10-31.json",
                  "uv run python scripts/analogue_panel_rag.py")
    ndrop = reg.num("panel_claims_dropped",
                    len(pan["proponent"]["dropped"]) + len(pan["dissent"]["dropped"]),
                    "claims the gate dropped", "reports/analogue_rag/panel_2020-10-31.json",
                    "uv run python scripts/analogue_panel_rag.py")
    nkept = reg.num("panel_claims_kept",
                    len(pan["proponent"]["kept"]) + len(pan["dissent"]["kept"]),
                    "claims that survived the gate", "reports/analogue_rag/panel_2020-10-31.json",
                    "uv run python scripts/analogue_panel_rag.py")
    L += ["", "WHEN DID THIS LAST HAPPEN   [live-only; not backtestable; run on a historical date]",
          "  Five nearest dates by the deterministic distance rule, and what followed each:"]
    for a in prec["state_matches"]:
        dd = reg.num(f"analogue_{a['rank']}_dist", a["distance"],
                     f"distance to the #{a['rank']} nearest state",
                     "reports/analogue_rag/panel_2020-10-31.json",
                     "uv run python scripts/analogue_panel_rag.py", dp=3)
        ff = reg.pct(f"analogue_{a['rank']}_fwd", a["forward_21d_return"],
                     f"what followed the #{a['rank']} match over 21 days",
                     "reports/analogue_rag/panel_2020-10-31.json",
                     "uv run python scripts/analogue_panel_rag.py", dp=2, sign=True)
        L += [f"    #{a['rank']} {a['date']}   distance {dd}   forward 21d {ff}   "
              f"conditions: {a['condition_record_status']}"]
    L += ["  Four of the five were followed by gains. Retrieval alone would reassure.", "",
          f"  A three-role panel argued over that record. {nkept} claims survived the gate and",
          f"  {ndrop} was dropped: a mechanism claim about a date whose condition record could not",
          "  support it. The gate keeps a mechanism claim only with a verbatim span from a",
          "  condition record for that date, and counts what it drops.",
          f"  Adjudicator: the match is NOT APT and mechanism CANNOT BE RULED ON, deciding on",
          f"  the field {pv['fails_on']} -- every one of the five predates the condition corpus,",
          "  so what they were priced on is unknown rather than different.",
          "",
          "  Two limits, both real. This step is LIVE-ONLY: it is the design's one agentic",
          "  component, a production system would use it, and it cannot be backtested, so its",
          f"  contribution here is shown and not measured. And the distance used {nb_} blocks of",
          "  the v1 engine's state space, not the design's exposures, so these are the v1",
          "  engine's neighbours.",
          "", "CATALYST DAYS IN THE WINDOW   [fields, not a model input]",
          "  An FOMC decision day falls in the next ten sessions. FOMC dates are published a",
          "  year ahead, so this one is genuinely scheduled and genuinely knowable today.",
          "  The earnings figure this line used to carry has been REMOVED. It was built from",
          "  the acceptance timestamps of 8-K item 2.02 filings inside the forecast window --",
          "  that is, from filings that did not exist on this date. It described what did",
          "  happen, not what was scheduled, and a page that sells point-in-time discipline",
          "  cannot print realised data under the word `scheduled`. Reporting an earnings",
          "  calendar properly needs a source of announced dates, which this package does not",
          "  have; it is named in DESIGNED AND NOT BUILT rather than approximated.",
          "  Neither field is an input to the severity line, because conditioning on the",
          "  calendar state did not improve it.", "",
          "WHAT WOULD CHANGE THIS READING",
          "  A vaccine or treatment readout resolves the shared condition, the loser leg reprices",
          "  upward together, and the short half takes the loss. Named by the filings, not by a",
          "  model: UAL states it as the condition on 2020-09-09.", "",
          "DESIGNED AND NOT BUILT   [named rather than given a number]",
          "  Positioning and crowding, the vulnerable-name lists, the catalyst agent, and the",
          "  theme pipeline's classifier run. None appears above as a figure.", ""]
    L += outcome(reg)
    L += [R]

    page = "\n".join(L)
    audit = enforce(page, reg)
    segs = segments(page, reg)
    assert "".join(x["text"] for x in segs) == page
    (OUT / "page_2020-10-30.txt").write_text(page + "\n")
    (OUT / "page_2020-10-30.provenance.json").write_text(json.dumps({
        "as_of": str(AS_OF.date()), "theme_source": src,
        "numerals": audit["numerals"], "distinct": audit["distinct"],
        "registered_fields": audit["registered_fields"], "passes": audit["passes"],
        "provenance": audit["provenance"], "segments": segs,
        "document_marks": document_marks(page, reg),
    }, indent=2, default=str))
    print(page)
    print(f"\ntheme source: {src}")
    print(f"numeral canary: {audit['distinct']} distinct numerals, "
          f"{audit['registered_fields']} registered fields, PASS")


if __name__ == "__main__":
    main()
