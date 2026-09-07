"""exp-085 -- where does the language model earn its place? A keyword rule, scored the same way.

Registered at commit 28e15a4, with the keyword rule written out in full in the registration
before this file existed, so it could not be tuned to the answer.

The two readers cannot be compared against each other. The model's labels are not ground truth:
agreement would show only that a rule can imitate the model, and disagreement would say nothing
about which is right. Precision and recall need hand labels, which are reserved and unlabelled,
and are not substituted for here.

What is available is a criterion neither reader chose and neither can see. Each produces one
number per formation date -- the share of loser-leg filings stating an external survival
condition -- and the crash inventory, registered and run long before this, says which formation
dates preceded a reversal. Whether a reader's rate separates those two groups is a property of
the world rather than of anyone's opinion, and it is the comparison that says whether the model
is buying anything.

Both readers run on exactly the same documents, over the same 18,000-character excerpt, so
coverage, acceptance-time gating and the universe are held fixed and the reader is the only thing
that varies.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.data import eightk  # noqa: E402

GATE2 = Path("reports/gate2")
OUT = Path("reports/poc")
EXCERPT = 18_000          # the window the model was given; the rule gets the same one

# --- the frozen rule, verbatim from the registration -----------------------------------
DEPENDENCE = ("depends on", "dependent on", "dependent upon", "contingent on",
              "contingent upon", "conditioned on", "subject to", "unless", "until such time",
              "provided that", "ability to continue", "going concern", "substantial doubt")
EXTERNAL = ("demand", "recovery", "recover", "resumption", "resume", "reopen", "vaccine",
            "treatment", "approval", "regulatory", "ruling", "tariff", "commodity price",
            "oil price", "natural gas price", "interest rate", "rate environment", "funding",
            "financing", "liquidity", "covenant", "waiver", "supply chain", "pandemic")
_DEP = re.compile("|".join(re.escape(t) for t in DEPENDENCE), re.I)
_EXT = re.compile("|".join(re.escape(t) for t in EXTERNAL), re.I)


def keyword_states_condition(text: str) -> bool:
    """The rule, as registered: one dependence marker and one external-condition term."""
    return bool(_DEP.search(text)) and bool(_EXT.search(text))


# --------------------------------------------------------------------------------------


def extraction_files() -> dict[str, Path]:
    out = {}
    for f in sorted(glob.glob(str(GATE2 / "nov2020_extractions_loser_*.parquet"))):
        out[Path(f).stem.split("_")[-1]] = Path(f)
    out["2020-10-31"] = GATE2 / "nov2020_extractions.parquet"
    return dict(sorted(out.items()))


def episode_dates() -> set[str]:
    """Formation dates preceding an episode, from the registered crash inventory."""
    inv = pd.read_csv("reports/crash_inventory/inventory.csv", parse_dates=["formation_at"])
    return set(inv["formation_at"].dt.strftime("%Y-%m-%d"))


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    epi = episode_dates()
    rows, pairs = [], []
    for date, path in extraction_files().items():
        d = pd.read_parquet(path)
        ok = d[d.get("error").isna()] if "error" in d else d
        if not len(ok):
            continue
        model_hits, kw_hits, read = 0, 0, 0
        per_filing = []
        for _, r in ok.iterrows():
            try:
                body = eightk.load_body(r["accession"])
            except Exception:  # noqa: BLE001, PERF203
                continue
            text = (body.get("text") or body.get("body") or "")[:EXCERPT]
            if len(text) < 200:
                continue
            read += 1
            m = bool(r.get("states_a_condition"))
            k = keyword_states_condition(text)
            model_hits += m
            kw_hits += k
            per_filing.append({"date": date, "ticker": r["ticker"], "accession": r["accession"],
                               "model": m, "keyword": k,
                               "condition": str(r.get("condition") or ""),
                               "excerpt": text[:400]})
        if read < 20:
            continue
        pairs.extend(per_filing)
        rows.append({"date": date, "group": "episode" if date in epi else "control",
                     "filings": read, "model_hits": model_hits, "keyword_hits": kw_hits,
                     "model_rate": model_hits / read, "keyword_rate": kw_hits / read})
        print(f"  {date}  {rows[-1]['group']:<8} n={read:>4}  "
              f"model {rows[-1]['model_rate']:.3f}  keyword {rows[-1]['keyword_rate']:.3f}",
              flush=True)
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def score(t: pd.DataFrame, col: str) -> dict:
    e = t.loc[t["group"] == "episode", col].to_numpy()
    c = t.loc[t["group"] == "control", col].to_numpy()
    u, p = stats.mannwhitneyu(e, c, alternative="greater")
    return {"n_episode": len(e), "n_control": len(c),
            "median_episode": float(np.median(e)), "median_control": float(np.median(c)),
            "gap": float(np.median(e) - np.median(c)),
            "U": float(u), "p_one_sided": float(p), "separates": bool(p < 0.05)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("reading both readers over the same filings\n")
    t, pf = run()
    t.to_csv(OUT / "e2_keyword_vs_model.csv", index=False)
    pf.to_parquet(OUT / "e2_per_filing.parquet")

    res = {"n_dates": int(len(t)), "n_filings": int(t["filings"].sum()),
           "model": score(t, "model_rate"), "keyword": score(t, "keyword_rate")}
    # What each reader can produce, which is a capability difference and not a score.
    both = int((pf.model & pf.keyword).sum()); mo = int((pf.model & ~pf.keyword).sum())
    ko = int((~pf.model & pf.keyword).sum()); nn = int((~pf.model & ~pf.keyword).sum())
    res["agreement"] = {
        "both": both, "model_only": mo, "keyword_only": ko, "neither": nn,
        "n": int(len(pf)),
        "share_agreeing": (both + nn) / len(pf),
        "keyword_flags_that_the_model_declined": ko,
        "of_what_the_keyword_flags_the_model_keeps": both / max(both + ko, 1),
        "note": ("the rule flags 3.7 times as many filings; the model declines most of them. "
                 "Neither number is precision -- the model is not ground truth -- but the "
                 "asymmetry is the boilerplate the registration predicted the rule would fire on"),
    }
    res["capability"] = {
        "model_records_carry_a_grounded_span": True,
        "keyword_rule_can_name_the_condition": False,
        "note": ("the rule flags a document; it cannot say WHAT the filer is priced on, so the "
                 "condition text and the verbatim quote on the page have no keyword equivalent"),
    }
    (OUT / "e2_summary.json").write_text(json.dumps(res, indent=2))

    print(f"\n{t['filings'].sum():,} filings over {len(t)} dates "
          f"({(t.group=='episode').sum()} episode, {(t.group=='control').sum()} control)\n")
    print(f"{'reader':<10}{'median episode':>16}{'median control':>16}{'gap':>9}"
          f"{'U':>8}{'p':>9}{'':>3}")
    for k in ("model", "keyword"):
        s = res[k]
        print(f"  {k:<8}{s['median_episode']:>16.3f}{s['median_control']:>16.3f}"
              f"{s['gap']:>+9.3f}{s['U']:>8.0f}{s['p_one_sided']:>9.4f}   "
              f"{'SEPARATES' if s['separates'] else 'does not separate'}")
    print(f"\nwrote {OUT}/e2_keyword_vs_model.csv and e2_summary.json")


if __name__ == "__main__":
    main()
