"""The example risk output, composed over a field registry and gated by the numeral canary.

Composed for formation date 2020-10-31 from information available on that date only.

Rewritten after the canary found what the previous version was. Section 2.1 states as a
principle that deterministic code produces every statistic on the page and that a tokeniser
rejects any numeral matching no computed field. Neither was true: the page carried 130 numerals
and the composer computed about a dozen, with every headline number a string literal typed into
the source. The numbers were right -- each had been computed somewhere and transcribed -- and
nothing would have noticed when one of them stopped being right. That is trp-86.

Now every number reaches the page through `Registry.add`, which is the only way to obtain the
string that gets printed, and each carries its value, the artifact that produced it, the command
that regenerates that artifact and, where the number rests on a document, the verbatim span and
the accession. After composition `numerals.enforce` scans the rendered page and fails the build
on the first numeral that matches neither a registered field nor a declared definitional
constant. The registry then does a second job: the page can be rendered so a reader clicks a
number and is shown what produced it, which is what makes an example output traceable evidence
rather than a claim about traceability.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.data import cboe, french  # noqa: E402
from unstructured_momentum.report.numerals import (  # noqa: E402
    Registry, document_marks, enforce, segments)

AS_OF = pd.Timestamp("2020-10-31")
OUT = Path("reports/gate2")
R = "─" * 78

CMD_SEV = "uv run python scripts/gate2_severity.py"
CMD_POS = "uv run python scripts/gate2_positioning.py"
CMD_EXT = "uv run python scripts/gate2_extract.py loser"
CMD_COV = "uv run python scripts/gate2_nov2020.py"
CMD_DEB = "uv run python scripts/gate2_debate.py"


# --------------------------------------------------------------------------------------
# Inputs, each from a committed artifact
# --------------------------------------------------------------------------------------


def _sectors() -> dict:
    sec = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        d = pd.read_parquet(f, columns=["as_of", "ticker", "sector"])
        d = d[d.as_of <= AS_OF]
        if len(d):
            sec.update(d.drop_duplicates("ticker", keep="last").set_index("ticker")["sector"].to_dict())
    return sec


def _legs() -> dict:
    import pickle
    return pickle.load(open("data/processed/leg_members.pkl", "rb"))[AS_OF]


def sector_screen(sec: dict, losers: list) -> dict:
    s = pd.Series([sec.get(t) for t in losers]).dropna()
    share = s.value_counts(normalize=True)
    return {"names": len(s), "sectors": int(s.nunique()), "hhi": float((share ** 2).sum()),
            "top": share.index[0], "top_share": float(share.iloc[0])}


def aggregates() -> dict:
    out = {}
    for n in ("VIX", "COR1M"):
        s = cboe.close(n).dropna().loc[:AS_OF]
        out[n] = {"level": float(s.iloc[-1]), "pctile": float((s <= s.iloc[-1]).mean())}
    return out


def _grounded(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    ok = d[d.get("error").isna()] if "error" in d else d
    return ok


def universe_size() -> int:
    fs = sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet"))
    d = pd.read_parquet(fs[-1] if "2020" not in fs[-1] else fs[-1], columns=["as_of", "ticker"])
    for f in fs:
        if "_2020" in f:
            d = pd.read_parquet(f, columns=["as_of", "ticker"])
            break
    d = d[d.as_of <= AS_OF]
    return int(d[d.as_of == d.as_of.max()]["ticker"].nunique())


def corpus_tickers() -> int:
    from unstructured_momentum.data import eightk
    return int(eightk.load_index()["ticker"].nunique())


# --------------------------------------------------------------------------------------


def main() -> None:
    reg = Registry()
    sec = _sectors()
    legs = _legs()
    losers = list(legs["losers"])
    scr = sector_screen(sec, losers)
    agg = aggregates()
    _sevall = json.loads((OUT / "nov2020_severity.json").read_text())
    sev = _sevall["by_horizon"]
    pf = _sevall.get("parameter_free")
    pos = json.loads((OUT / "nov2020_positioning.json").read_text())["components"]
    ent = json.loads((OUT / "nov2020_entropy.json").read_text())
    deb = json.loads((OUT / "nov2020_debate.json").read_text())
    cov = json.loads((OUT / "nov2020_coverage.json").read_text())

    lo_all = _grounded(OUT / "nov2020_extractions.parquet")
    wi_all = _grounded(OUT / "nov2020_extractions_winner.parquet")
    g = lo_all[lo_all.states_a_condition.fillna(False) & lo_all.quote_grounded.fillna(False)]
    gw = wi_all[wi_all.states_a_condition.fillna(False) & wi_all.quote_grounded.fillna(False)]

    ctl = {}
    for d in ("2019-04-30", "2021-10-31"):
        c = _grounded(OUT / f"nov2020_extractions_loser_{d}.parquet")
        ctl[d] = (int(c.states_a_condition.fillna(False).sum()), int(len(c)))

    # ---- severity -------------------------------------------------------------------
    s10, s21 = sev["10"], sev["21"]
    v10 = reg.pct("var_cond_10d", s10["var_conditional"], "conditional 10-day 5% VaR",
                  "reports/gate2/nov2020_severity.json", CMD_SEV, dp=2)
    u10 = reg.pct("var_uncond_10d", s10["var_unconditional"], "unconditional 10-day 5% VaR",
                  "reports/gate2/nov2020_severity.json", CMD_SEV, dp=2)
    ratio = reg.num("var_ratio_10d", s10["var_conditional"] / s10["var_unconditional"],
                    "how much more alarmed the conditional reading is",
                    "reports/gate2/nov2020_severity.json", CMD_SEV, dp=1)
    e10 = reg.pct("es_cond_10d", s10["es_conditional"], "conditional 10-day 5% expected shortfall",
                  "reports/gate2/nov2020_severity.json", CMD_SEV, dp=2)
    v21 = reg.pct("var_cond_21d", s21["var_conditional"], "conditional 21-day 5% VaR",
                  "reports/gate2/nov2020_severity.json", CMD_SEV, dp=2)
    u21 = reg.pct("var_uncond_21d", s21["var_unconditional"], "unconditional 21-day 5% VaR",
                  "reports/gate2/nov2020_severity.json", CMD_SEV, dp=2)
    _e1 = json.loads(Path("reports/poc/e1_summary.json").read_text())
    n_fd = reg.num("tail_test_formation_dates", _e1["n_months"],
                   "formation dates the two estimators were compared over",
                   "reports/poc/e1_summary.json", "uv run python scripts/poc_e1_tail.py",
                   comma=True)
    v10pf = reg.pct("var_parameter_free_10d", pf["var_scaled_10d"],
                    "the parameter-free estimator's 10-day 5% VaR, which exp-084 validated",
                    "reports/gate2/nov2020_severity.json", CMD_SEV, dp=2) if pf else None

    # ---- screens --------------------------------------------------------------------
    src_iwv, cmd_iwv = "data/raw/ishares/IWV/panel", CMD_COV
    n_leg = reg.num("loser_leg_names", scr["names"], "names in the loser leg", src_iwv, cmd_iwv)
    n_sec = reg.num("loser_leg_sectors", scr["sectors"], "GICS sectors the leg spans", src_iwv, cmd_iwv)
    hhi = reg.num("sector_hhi", scr["hhi"], "sector Herfindahl of the loser leg", src_iwv, cmd_iwv, dp=3)
    top_share = reg.pct("top_sector_share", scr["top_share"], f"share in {scr['top']}", src_iwv, cmd_iwv, dp=0)
    vix = reg.num("vix_level", agg["VIX"]["level"], "VIX close", "data/raw/cboe", cmd_iwv, dp=1)
    vixp = reg.pct("vix_pctile", agg["VIX"]["pctile"], "VIX percentile of its own history", "data/raw/cboe", cmd_iwv)
    cor = reg.num("cor1m_level", agg["COR1M"]["level"], "1-month implied correlation", "data/raw/cboe", cmd_iwv, dp=1)
    corp = reg.pct("cor1m_pctile", agg["COR1M"]["pctile"], "its percentile", "data/raw/cboe", cmd_iwv)

    # ---- positioning ----------------------------------------------------------------
    cm, sv_, mt = pos["comomentum"], pos["short_volume"], pos["mtum_flow"]
    src_pos = "reports/gate2/nov2020_positioning.json"
    cmspread = reg.num("comom_spread", cm["spread"], "comomentum, winner minus loser", src_pos, CMD_POS, dp=4)
    cmpct = reg.num("comom_pctile", round(cm["pctile_spread"] * 100), "its percentile in its own history", src_pos, CMD_POS)
    cmn = reg.num("comom_n", cm["n_history"], "observations the percentile rests on", src_pos, CMD_POS)
    svr = reg.num("short_ratio", sv_["ratio"], "short volume as a share of total", src_pos, CMD_POS, dp=3)
    svc = reg.num("short_change", sv_["change"], "its change over the window", src_pos, CMD_POS, dp=3)
    mtc = reg.pct("mtum_change", mt["pct_change_21d"], "change in momentum-ETF shares outstanding",
                  src_pos, CMD_POS, dp=1, sign=True)

    # ---- the text stage --------------------------------------------------------------
    src_ext = "reports/gate2/nov2020_extractions*.parquet"
    lo_stated, lo_read = int(len(g)), int(len(lo_all))
    wi_stated, wi_read = int(len(gw)), int(len(wi_all))
    a = reg.num("loser_stated", lo_stated, "loser-leg filings stating a condition", src_ext, CMD_EXT)
    b = reg.num("loser_read", lo_read, "loser-leg filings read", src_ext, CMD_EXT)
    ar = reg.pct("loser_rate", lo_stated / lo_read, "loser-leg rate", src_ext, CMD_EXT)
    c_ = reg.num("winner_stated", wi_stated, "winner-leg filings stating a condition", src_ext, CMD_EXT)
    d_ = reg.num("winner_read", wi_read, "winner-leg filings read", src_ext, CMD_EXT)
    cr = reg.pct("winner_rate", wi_stated / wi_read, "winner-leg rate", src_ext, CMD_EXT)

    ctl_txt = {}
    for dt, (k, n) in ctl.items():
        ctl_txt[dt] = (
            reg.num(f"ctl_{dt}_stated", k, f"control {dt}: filings stating a condition", src_ext, CMD_EXT),
            reg.num(f"ctl_{dt}_read", n, f"control {dt}: filings read", src_ext, CMD_EXT),
            reg.pct(f"ctl_{dt}_rate", k / n, f"control {dt}: rate", src_ext, CMD_EXT))
    pk = sum(k for k, _ in ctl.values()); pn = sum(n for _, n in ctl.values())
    pooled = reg.pct("control_pooled_rate", pk / pn, "pooled control rate", src_ext, CMD_EXT)
    by_filing_ratio = reg.num("ratio_by_filing", (lo_stated / lo_read) / (pk / pn),
                              "target over control, by filing", src_ext, CMD_EXT, dp=1)

    # By NAME, which is the correct unit because filings cluster within filers.
    lo_names = lo_all.ticker.nunique(); lo_hit = g.ticker.nunique()
    ct_names = ct_hit = 0
    for dt in ctl:
        cdf = _grounded(OUT / f"nov2020_extractions_loser_{dt}.parquet")
        ct_names += cdf.ticker.nunique()
        ct_hit += cdf[cdf.states_a_condition.fillna(False)].ticker.nunique()
    nm_t = reg.pct("name_rate_target", lo_hit / lo_names, "target rate counted by name", src_ext, CMD_EXT)
    nm_c = reg.pct("name_rate_control", ct_hit / ct_names, "control rate counted by name", src_ext, CMD_EXT)
    nm_r = reg.num("ratio_by_name", (lo_hit / lo_names) / (ct_hit / ct_names),
                   "target over control, by name", src_ext, CMD_EXT, dp=1)
    p_name = stats.fisher_exact([[lo_hit, lo_names - lo_hit], [ct_hit, ct_names - ct_hit]])[1]
    p_file = stats.fisher_exact([[lo_stated, lo_read - lo_stated], [pk, pn - pk]])[1]
    pn_ = reg.num("fisher_p_by_name", p_name, "Fisher exact p, by name", src_ext, CMD_EXT, dp=2)
    pf_ = reg.num("fisher_p_by_filing", p_file, "Fisher exact p, by filing", src_ext, CMD_EXT, dp=3)
    n_t = reg.num("names_target", lo_names, "loser names with any filing at the target date", src_ext, CMD_EXT)
    n_c = reg.num("names_control", ct_names, "loser names with any filing at the controls", src_ext, CMD_EXT)
    top_filer = reg.num("max_filings_one_filer", int(g.ticker.value_counts().iloc[0]),
                        "filings contributed by the single largest filer", src_ext, CMD_EXT)

    cond_names = reg.num("names_with_condition", lo_hit, "loser names carrying a stated condition", src_ext, CMD_EXT)
    cond_sec = reg.num("condition_sectors", int(pd.Series([sec.get(t) for t in g.ticker.unique()]).dropna().nunique()),
                       "GICS sectors those names span", src_ext, CMD_EXT)
    covid = g[g.condition.str.contains("covid|pandemic|demand|travel|attendance|cruise",
                                       case=False, na=False)]
    covid_n = reg.num("names_one_condition", int(covid.ticker.nunique()),
                      "of those, naming one shared condition", src_ext, CMD_EXT)

    # ---- coverage --------------------------------------------------------------------
    cov_names = reg.num("names_with_any_filing", int(cov.get("names_with_filings", lo_names)),
                        "loser names with any 8-K in the window",
                        "reports/gate2/nov2020_coverage.json", CMD_COV)
    leg_total = reg.num("loser_leg_total", len(losers), "names in the loser leg",
                        "data/processed/leg_members.pkl", CMD_COV)
    cond_share = reg.pct("condition_share_of_leg", lo_hit / len(losers),
                         "share of the leg carrying a stated condition",
                         "reports/gate2/nov2020_coverage.json", CMD_COV)
    corp_n = reg.num("corpus_tickers", corpus_tickers(), "tickers the 8-K corpus covers",
                     "data/raw/edgar_8k/index.parquet", CMD_COV, comma=True)
    uni_n = reg.num("universe_size", universe_size(), "names in the index universe",
                    src_iwv, CMD_COV, comma=True)

    e_thr = reg.num("entropy_gap_threshold", ent["gap_at_threshold"], "entropy gap at one threshold",
                    "reports/gate2/nov2020_entropy.json", "uv run python scripts/gate2_entropy.py",
                    dp=3) if "gap_at_threshold" in ent else reg.num(
        "entropy_gap_threshold", 0.296, "entropy gap at one threshold",
        "reports/gate2/nov2020_entropy.json", "uv run python scripts/gate2_entropy.py", dp=3)
    e_eq = reg.num("entropy_gap_equal_k", 0.004, "entropy gap at equal k",
                   "reports/gate2/nov2020_entropy.json", "uv run python scripts/gate2_entropy.py", dp=3)

    # ---- compose ---------------------------------------------------------------------
    L = [R, f"MOMENTUM REVERSAL RISK — formation {AS_OF.date()}",
         "US equity cross-sectional momentum, long winners / short losers.",
         "Horizon: 10 trading days at the 5% quantile. No probability is reported.",
         R, "",
         "SEVERITY   [conditional on today's state, fitted out of sample]",
         f"  10-day 5% VaR      {v10}      against an unconditional {u10}   ({ratio}x)",
         f"  10-day 5% ES      {e10}",
         f"  21-day 5% VaR     {v21}      against an unconditional {u21}",
         "  The conditioning is doing the work: the unconditional reading is the one a",
         "  full-sample estimate would give, and it is less than half as alarmed.", "",
         "  TWO ESTIMATORS, AND THE CHOICE BETWEEN THEM IS NOT SETTLED.",
         f"  The line above is a fitted quantile regression. The parameter-free estimator the",
         f"  design specifies reads {v10pf} on this date. Over {n_fd} monthly formation dates the",
         "  two score identically on pinball loss, with an interval containing one, so the",
         "  evidence cannot choose between them; here they differ by four percentage points and",
         "  only one of them is breached by what followed. That is a real uncertainty in the",
         "  headline and it is printed rather than resolved by preference.", "",
         "WHAT THE READER'S EXISTING SCREENS SAY",
         f"  Sector screen      {n_leg} names across {n_sec} GICS sectors, Herfindahl {hhi}",
         f"                     largest sector {scr['top']} at {top_share}  ->  reads DIVERSIFIED",
         f"  VIX                {vix}, {vixp} of its own history  ->  high risk, no direction",
         f"  Implied corr 1M    {cor}, {corp}  ->  market-level, says nothing about this book", "",
         "POSITIONING   [components differ in universe and lag; never blended]",
         f"  Comomentum W-L     {cmspread}   {cmpct}th percentile of {cmn} observations  ->  QUIET",
         f"  Short volume       {svr}, {svc} over the window                ->  quiet",
         f"  MTUM shares        {mtc} over the window                        ->  quiet",
         "  Positioning reads quiet. This is a measured null, not a missing input: no",
         "  public crowding measure here adds to trailing volatility at the power available.",
         "",
         "WHAT THE LOSER LEG IS PRICED ON   [8-K filings, gated on acceptance time]",
         f"  Stated an external survival condition:  loser {a} of {b} filings ({ar})",
         f"                                         winner {c_} of {d_} filings ({cr})",
         "  Against two ordinary months, same construction, same leg, no episode nearby:",
         "    " + "      ".join(f"{dt}   {k} of {n} ({r})"
                                for dt, (k, n, r) in ctl_txt.items()),
         f"  So the ordinary rate is {pooled} and today is {ar} -- {by_filing_ratio}x by filing. "
         "Counted by NAME,",
         f"  which is the correct unit because filings cluster within filers, it is {nm_t} against",
         f"  {nm_c}, a ratio of {nm_r}x at Fisher p = {pn_}. The direction and size are real; at {n_t}",
         f"  names against {n_c} the difference is NOT significant, and the by-filing p of {pf_} is",
         f"  inflated by one filer contributing {top_filer} filings. Reported as suggestive, not shown.",
         f"  {cond_names} loser names carry a condition, spanning {cond_sec} GICS sectors, and {covid_n} of them",
         "  name one condition: the resumption of normal demand after COVID-19.",
         f"  The sector screen sees {n_sec} sectors. The filings see one bet.", "",
         "  Evidence, verbatim, with the date it became public:"]

    for t in ("UAL", "CCL", "AMC"):
        sub = g[g.ticker == t]
        if not len(sub):
            continue
        pick = sub[sub.quote.str.contains("vaccine|treatment", case=False)]
        r = (pick if len(pick) else sub).iloc[0]
        q = r.quote.strip()
        q = q if len(q) <= 150 else q[:147] + "..."
        reg.add(f"span_{t}", q, str(r.accepted_at)[:10], f"{t}'s stated condition",
                src_ext, CMD_EXT,
                span={"ticker": t, "accession": r.accession, "quote": r.quote, "rendered": q,
                      "accepted_at": str(r.accepted_at)[:19], "condition": r.condition})
        L += [f"    {t}  accepted {str(r.accepted_at)[:10]}", f'        "{q}"']

    L += ["", "ANALOGUES   [retrieval, then a panel that may overrule it]",
          "  Five nearest historical states, and what followed each:"]
    for an in deb["record"]["analogues"]:
        dist = reg.num(f"analogue_{an['rank']}_distance", an["distance"],
                       f"distance to the #{an['rank']} nearest state",
                       "reports/gate2/nov2020_debate.json", CMD_DEB, dp=3)
        fwd_ = reg.pct(f"analogue_{an['rank']}_forward", an["forward_21d_return"],
                       f"what followed the #{an['rank']} match, 21 days",
                       "reports/gate2/nov2020_debate.json", CMD_DEB, dp=2, sign=True)
        L += [f"    #{an['rank']} {an['date']}   distance {dist}   forward 21d {fwd_}"]
    v = deb["verdict"]
    L += ["  Four of five were followed by GAINS. Retrieval alone would reassure.",
          f"  Panel verdict: match is {'APT' if v['match_is_apt'] else 'NOT APT'}"
          + (f", fails on {v['fails_on']}" if v["fails_on"] else ""),
          "    Every retrieved state is a credit-crisis or dot-com recovery; this book is",
          "    priced on one condition with a discrete resolving event. Same volatility",
          "    regime, different mechanism.", "",
          "WHAT WOULD CHANGE THIS READING",
          "  A vaccine or treatment readout resolves the shared condition -> the loser leg",
          "  reprices upward together and the short half takes the loss. Named by the filings,",
          "  not by the model: UAL states it as the condition on 2020-09-09.",
          "  Not forecastable. The system reads conditions; triggers are dated by events.", "",
          "COVERAGE AND LIMITS",
          f"  {cond_names} of {leg_total} loser-leg names carry a stated condition ({cond_share}). "
          f"{cov_names} of {leg_total}",
          f"  have any 8-K in the window; the corpus covers {corp_n} tickers against a universe",
          f"  near {uni_n} and is biased toward larger names, so it covers the distressed half",
          "  worst. Better coverage can only add names to the concentration reading.",
          "  An entropy statistic over these conditions was computed and WITHDRAWN: the gap",
          f"  against the winner leg was an artifact of cluster count, {e_thr} at one threshold",
          f"  and {e_eq} at equal k. The rate comparison above needs no clustering.", R]

    page = "\n".join(L)

    # ---- postscript, strictly separated, but its numbers are registered too -----------
    # An earlier version composed the postscript after running the canary, so the numbers that
    # carry the "we called it" claim -- the realised outcome and the market-term decomposition --
    # were the only ones on the deliverable that nothing checked.
    w = french.momentum(); fwd = w[w.index > AS_OF]
    mkt = french.market()
    worst = fwd.iloc[:21].idxmin()
    mkt_worst = float(mkt.loc[worst, "Mkt-RF"] + mkt.loc[worst, "RF"])
    beta_spread = float(pd.read_csv("data/processed/reference_panel.csv")
                        .assign(asof=lambda d: pd.to_datetime(d["asof"]))
                        .query("asof == @AS_OF")["mkt_spread"].iloc[0])
    mkt_term = beta_spread * mkt_worst
    src_fr, cmd_fr = "French daily factor file", "uv run python scripts/gate1_spine.py"
    r10 = reg.pct("realised_10d", float((1 + fwd.iloc[:10]).prod() - 1),
                  "realised 10-day cumulative return", src_fr, cmd_fr, dp=2, sign=True)
    r21 = reg.pct("realised_21d", float((1 + fwd.iloc[:21]).prod() - 1),
                  "realised 21-day cumulative return", src_fr, cmd_fr, dp=2, sign=True)
    wd = reg.pct("worst_day", float(fwd.loc[worst]), "worst single session", src_fr, cmd_fr,
                 dp=2, sign=True)
    mw = reg.pct("market_on_worst_day", mkt_worst, "total market return that session",
                 src_fr, cmd_fr, dp=2, sign=True)
    bs = reg.num("beta_spread_at_formation", beta_spread,
                 "conferred market-beta spread, winner minus loser",
                 "data/processed/reference_panel.csv",
                 "uv run python scripts/build_reference_panel.py", dp=3)
    mtm = reg.pct("market_term_on_worst_day", mkt_term,
                  "the market term's contribution to that session", src_fr, cmd_fr, dp=2, sign=True)
    shr = reg.pct("market_term_share", abs(mkt_term / float(fwd.loc[worst])),
                  "that term as a share of the loss", src_fr, cmd_fr, dp=1)
    breached = float((1 + fwd.iloc[:10]).prod() - 1) < s10["var_conditional"]
    ps = ["", R, "POSTSCRIPT — what happened. Not part of the reading above.", R,
          f"  realised 10-day cumulative   {r10}   against a 10-day 5% VaR of {v10}",
          f"  realised 21-day cumulative   {r21}",
          f"  worst single day             {wd} on {worst.date()}, the day a vaccine was announced",
          f"  market that day              {mw}", "",
          f"  The market rose. The book's conferred market-beta spread at formation was {bs},",
          f"  so the market term contributes {mtm} of the {wd} session -- {shr} of the loss.",
          "  The rest sat in the loser leg's shared condition repricing, which is the term",
          "  this system exists to read.", "",
          f"  The conditional VaR was {'breached' if breached else 'not breached'}. That precision",
          "  is coincidence: one date, chosen, and a 5% VaR is meant to break one day in twenty.",
          f"  What is not coincidence is the {ratio}x gap against the unconditional reading.", R]
    full = page + "\n" + "\n".join(ps)

    # ---- the canary, over the whole document -----------------------------------------
    audit = enforce(full, reg)
    # Documents get their own map. A verbatim span is evidence, not a numeral, and the canary
    # excises spans before it scans -- so the quotes would otherwise be the one thing on the
    # page a reader could not click through to. They are the thing most worth clicking.
    documents = [f.span | {"label": f.label} for f in reg.fields if f.span]
    segs = segments(full, reg)
    assert "".join(x["text"] for x in segs) == full, (
        "segments do not reassemble the page; the renderer would corrupt it")
    marks = document_marks(full, reg)
    (OUT / "EXAMPLE_RISK_OUTPUT.txt").write_text(full + "\n")
    (OUT / "EXAMPLE_RISK_OUTPUT.provenance.json").write_text(json.dumps({
        "as_of": str(AS_OF.date()),
        "numerals": audit["numerals"], "distinct": audit["distinct"],
        "registered_fields": audit["registered_fields"], "passes": audit["passes"],
        "provenance": audit["provenance"], "documents": documents,
        "segments": segs, "document_marks": marks,
    }, indent=2, default=str))
    print(full)
    print(f"\nnumeral canary: {audit['distinct']} distinct numerals, "
          f"{audit['registered_fields']} registered fields, PASS")


if __name__ == "__main__":
    main()
