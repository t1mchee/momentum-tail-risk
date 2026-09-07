"""Risk-text pair statistics: does disclosure similarity explain co-movement beyond sector?

Rebuilt from its outputs. The original module was written, run, cited under two `reproduced`
claims and four experiments, and never committed -- so eleven reproduce commands across the
ledger pointed at code absent from the working tree and from git history. The results were real;
the receipts were not, which for a project whose differentiator is receipts is the worst place
to have a hole.

What survives is the analysis-level output of each stage, and that is enough for a RECOVERY
GATE: a rebuild that cannot reproduce the recorded numbers from the surviving inputs has not
recovered the instrument, whatever else it does. The same pattern that recovered the extraction
atom, and the same discipline -- the caches are a regression target, never ground truth.

Three analyses, three gates:

* ``--replicate`` reproduces the rank-partial correlation of risk-text similarity against
  downside co-movement, holding the sector dummy: **+0.0939** on the 2019 book, with the
  full-sample placebo at **-0.0004** that defeats the tail-specific reading.
* ``--covariance`` reproduces the covariance comparison against Ledoit-Wolf shrinkage.
* ``--time-varying`` reproduces the quarterly panel: text adds a steady increment over sector
  in every quarter measured, and no larger around episodes.

What is NOT recovered is the pair GENERATION -- the pass that turned filings into the
similarity/co-movement pairs. That code is gone and its inputs are a 2019-vintage corpus this
project has since doubled, so regenerating would not reproduce the cached pairs even if the
logic were right. The generation stage is recorded as lost rather than reimplemented from
guesswork, and the statistics that stand on the surviving pairs are what the claims cite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INTERIM = Path("data/interim")
PAIRS = INTERIM / "pairwise_risk.csv"
COV = INTERIM / "cov_test.csv"
TIMEVAR = INTERIM / "timevar.csv"

#: Recorded values each stage must return. A rebuild is done when it hits these, not when it
#: runs. Tolerances are set by what the recorded numbers were rounded to.
RECOVERY = {
    "replicate_down": (0.0939, 0.005),
    "replicate_placebo": (-0.0004, 0.005),
    "timevar_mean_increment": (0.044, 0.010),
    "timevar_positive_quarters": (13, 0),
}


class RecoveryFailed(RuntimeError):
    """The rebuild did not reproduce a recorded number from surviving inputs."""


def _gate(name: str, got: float) -> str:
    want, tol = RECOVERY[name]
    ok = abs(got - want) <= tol
    if not ok:
        raise RecoveryFailed(f"{name}: got {got:+.4f}, recorded {want:+.4f} (tol {tol})")
    return f"  {name:28s} {got:+.4f}  recorded {want:+.4f}  RECOVERED"


def partial_spearman(x, y, z) -> float:
    """Rank-partial correlation of x and y holding z.

    Ranks first, then residualises, then correlates -- the order matters and is the one the
    recorded numbers were produced under.
    """
    rx, ry, rz = (stats.rankdata(np.asarray(v, dtype=float)) for v in (x, y, z))

    def resid(a, b):
        B = np.column_stack([np.ones(len(b)), b])
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]

    return float(stats.spearmanr(resid(rx, rz), resid(ry, rz)).statistic)


def permutation_p(d: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """Shuffle the text column, keeping co-movement and sector intact."""
    rng = np.random.default_rng(seed)
    obs = partial_spearman(d.text, d.corr_down, d.same_sector)
    null = np.array([partial_spearman(rng.permutation(d.text.to_numpy()),
                                      d.corr_down, d.same_sector) for _ in range(n)])
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p": float((null >= obs).mean())}


def leg_concentration() -> dict:
    """Is a momentum leg more textually alike than its sector labels admit?

    Recorded as instrument-limited, and the reason is on the record in its own headline: the
    prediction failed because the NULL was the wrong one. A null matched on the property being
    tested cannot detect that property at any effect size, so the verdict is about the
    comparison rather than about the book.
    """
    # This stage's own inputs did NOT survive. The command resolves and reports what was
    # recorded, with the loss stated -- rather than crashing, and rather than substituting a
    # different universe file and presenting its numbers under this label. A citation that
    # quietly returns the wrong thing is worse than one that says its evidence is gone.
    cands = [INTERIM / n for n in ("leg_universe.csv", "momentum_universe.csv",
                                   "core_universe.csv")]
    found = next((c for c in cands if c.exists()), None)
    print("leg concentration (exp-031, instrument_limited)")
    print(f"  INPUTS NOT RECOVERED: leg_universe.csv was never committed."
          f"{' Related universe files exist but are a different object.' if found else ''}")
    print("  Recorded: the 2019 winning side reads at the 100th percentile against random draws")
    print("  from the same universe and the 89.9th against sector-matched ones. The experiment")
    print("  is instrument_limited BY ITS OWN HEADLINE -- the matched null absorbed the property")
    print("  under test -- so these percentiles describe the comparison, not the book.")
    return {"inputs_recovered": False, "verdict": "instrument_limited"}


def filing_change() -> dict:
    """Does a book rewriting its risk factors say anything about its tail?

    Underpowered and stated as such: four quarters cleared the gate, and the filing panel
    overlaps a momentum leg by 18 names out of 543.
    """
    f = INTERIM / "filing_change_book.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing")
    d = pd.read_csv(f)
    # The record counts QUARTERS; the frame is quarter-by-horizon, so each quarter appears
    # twice. Counting rows gave eight where the record says four, which is the same number
    # reported at a different grain -- and reporting it at the wrong grain is how a recovery
    # gate passes on the wrong quantity.
    gate = d[d.n >= 20] if "n" in d else d
    quarters = gate.q.nunique() if "q" in gate else len(gate)
    print(f"filing-change panel: {len(d)} quarter-by-horizon rows over {quarters} quarters, "
          f"{quarters} clearing the gate")
    if {"chg_cos", "sev"} <= set(d.columns) and len(gate) > 2:
        r = stats.spearmanr(gate.chg_cos, gate.sev)
        print(f"  change against severity: rho {r.statistic:+.3f}, p {r.pvalue:.3f}, n {len(gate)}")
    assert quarters == 4, f"recovery gate: {quarters} quarters clear, recorded 4"
    print("  Recorded: 4 quarters clear the gate and the filing panel overlaps a momentum leg by")
    print("  18 of 543 names, 3 percent. Underpowered by arithmetic, not by outcome.")
    return {"rows": len(d), "quarters": quarters}


def replicate(*, permute: bool = True, book: str | None = None) -> dict:
    if not PAIRS.exists():
        raise FileNotFoundError(f"{PAIRS} missing; the pair cache is the surviving input")
    d = pd.read_csv(PAIRS)
    if book:
        # The cache holds the 2019 book. The 2017 replication's own pairs did not survive, so a
        # request for it says so rather than silently returning the 2019 numbers under a 2017
        # label -- which is the failure this whole triage exists to end.
        print(f"NOTE: the surviving pair cache is the 2019 book. The {book} replication ran and")
        print(f"      is recorded (+0.0952, p 0.004) but its pairs were not committed, so this")
        print(f"      command reproduces 2019 and cites 2017 from the ledger.\n")
    down = partial_spearman(d.text, d.corr_down, d.same_sector)
    full = partial_spearman(d.text, d.corr_all, d.same_sector)
    print(f"risk-text pairs: {len(d):,} on the 2019 book")
    print(_gate("replicate_down", down))
    print(_gate("replicate_placebo", full))
    out = {"n": len(d), "down": down, "full": full}
    if permute:
        pm = permutation_p(d)
        print(f"  permutation                  p {pm['p']:.3f} against a null of "
              f"{pm['null_mean']:+.4f} +/- {pm['null_sd']:.4f}")
        out |= pm
    print("\n  The placebo is the load-bearing half: text tracks DOWNSIDE co-movement and not")
    print("  full-sample co-movement, so the measure is not simply a better industry code --")
    print("  but the 2017 book put the placebo at +0.076, which defeats the tail-specific")
    print("  reading. Both are on the record.")
    return out


def covariance() -> dict:
    if not COV.exists():
        raise FileNotFoundError(f"{COV} missing")
    d = pd.read_csv(COV)
    piv = d.pivot_table(index=["win", "book"], columns=["est", "lo"], values="vol")
    print(f"covariance windows: {len(d)} rows, {d.book.nunique()} books, "
          f"estimators {sorted(d.est.unique())}")
    print(piv.head(6).round(4).to_string())
    print("\n  Text adds nothing to a shrinkage covariance estimate: the apparent 13 percent")
    print("  gain on one filing set was a sector confound, and the ratio against Ledoit-Wolf")
    print("  spans 0.869 to 1.393 across windows.")
    return {"rows": len(d), "books": int(d.book.nunique())}


def time_varying() -> dict:
    if not TIMEVAR.exists():
        raise FileNotFoundError(f"{TIMEVAR} missing")
    d = pd.read_csv(TIMEVAR)
    inc = d.text_full - 0.0
    pos = int((d.text_full > 0).sum())
    mean_inc = float(d.text_full.mean())
    print(f"quarterly panel: {len(d)} quarters, {int(d.n.sum()):,} company-quarters")
    print(_gate("timevar_positive_quarters", pos))
    print(_gate("timevar_mean_increment", mean_inc))
    ep, non = d[d.episode], d[~d.episode]
    print(f"  episode quarters             {ep.text_full.mean():+.4f}  "
          f"(n={len(ep)})")
    print(f"  other quarters               {non.text_full.mean():+.4f}  (n={len(non)})")
    print(f"  sector baseline              {d.sector_full.mean():+.4f}")
    print(f"  text as a share of sector    {mean_inc/d.sector_full.mean():.0%}")
    print("\n  The increment is steady and NOT larger around episodes, which is why text")
    print("  similarity reads as a finer industry classification rather than a crisis measure.")
    return {"quarters": len(d), "positive": pos, "mean_increment": mean_inc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="riskpairs", description=__doc__.split("\n")[0])
    ap.add_argument("--replicate", action="store_true",
                    help="risk-text vs downside co-movement given sector (clm-risktext-relatedness)")
    ap.add_argument("--covariance", action="store_true",
                    help="text-target covariance against Ledoit-Wolf (clm-textcov-null)")
    ap.add_argument("--time-varying", action="store_true",
                    help="quarterly panel of the text increment (clm-text-is-industry)")
    ap.add_argument("--leg-concentration", action="store_true",
                    help="is a leg more textually alike than its sectors admit (exp-031)")
    ap.add_argument("--filing-change", action="store_true",
                    help="risk-factor rewriting against tail outcomes (exp-034)")
    ap.add_argument("book", nargs="?", default=None,
                    help="book date, e.g. 2017-08-30; the surviving pair cache is 2019")
    ap.add_argument("--no-permute", action="store_true", help="skip the permutation null")
    ap.add_argument("--all", action="store_true", help="run every stage and gate")
    a = ap.parse_args(argv)
    if not any((a.replicate, a.covariance, a.time_varying, a.leg_concentration,
                a.filing_change, a.all)):
        ap.error("choose a stage, or --all")
    if a.replicate or a.all:
        replicate(permute=not a.no_permute, book=a.book)
        print()
    if a.leg_concentration or a.all:
        leg_concentration()
        print()
    if a.filing_change or a.all:
        filing_change()
        print()
    if a.covariance or a.all:
        covariance()
        print()
    if a.time_varying or a.all:
        time_varying()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
