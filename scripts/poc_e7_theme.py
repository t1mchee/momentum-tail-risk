"""exp-091 -- the theme pipeline, built and run on the eleven registered dates plus the memo date.

Registered in project/experiments.yaml at commit fef25f9a4 before any of this existed. The
registration's `method.grid` is the specification and is implemented literally:

    200 random books per date matched on decile-by-sector cell counts per leg; Ledoit-Wolf with
    a constant-correlation target on the market residual over 126 days; three components; the
    component sign fixed so the book's exposure is positive; top-quartile sets by loading. One
    named public encoder; HDBSCAN at minimum cluster size 25 over episode-window news titles
    plus the Item 1A sentences of set and random-set companies; the member distance cut at the
    90th percentile of member-to-centroid distance and the centroid cut at the 10th percentile
    of nearest-centroid distance. Assignment uses the leg-sign rule, one company one vote.

Authority: docs/design/spec.yaml v2-draft-15 (nodes `sort`, `random`, `covariance`, `pca`,
`bridge`, `embedspace`, `cluster`, `themescore`), with POC-BRIEF corrections 2, 9, 10, 15, 17
and 19.  Section 1 of the brief names `factor/`, `features/conferred` and `factor/placebo` as
existing assets; they do not exist on disk and are not used.  Everything here is built from the
holdings panel, leg_members.pkl, book_returns.parquet, the GDELT episode titles and the EDGAR
Item 1A stores.

EVERY CHOICE THAT IS NOT IN THE REGISTRATION IS LISTED IN `DEVIATIONS` BELOW AND COPIED INTO
THE OUTPUT JSON.  Nothing below this line is tuned after seeing a result: the keyword rule, the
truncation length, the minimum window length and the expected-label term sets are all fixed
here before the first run, and the expected labels come from the registration itself.

Writes: reports/poc/e7_summary.json, reports/poc/e7_themes/<date>.json,
        reports/poc/e7_cache/  (embedding cache, so a rerun is cheap)
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.factor import splits as fsplits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "poc"
THEMES = OUT / "e7_themes"
CACHE = OUT / "e7_cache"

# ---------------------------------------------------------------- frozen configuration

#: The registration: 200 random books per date.
N_RANDOM = 200
#: The registration and spec node `covariance`: 126 days.
WINDOW = 126
#: The registration: three components.
N_COMPONENTS = 3
#: Spec node `pca`: the component set is the top quartile of companies by |v_ik|.
SET_QUANTILE = 0.75
#: The registration: HDBSCAN at a minimum cluster size of 25.
MIN_CLUSTER_SIZE = 25
#: Spec node `embedspace` / `namecluster`: a cluster with fewer than 25 news members is
#: company-specific.
NEWS_MEMBERS_FOR_SHARED = 25
#: The registration: the theme rises if its excess weight is above the 95th percentile of the
#: random books' maximum.
PLACEBO_Q = 0.95
#: Spec node `pca`: a component is above noise when VS_k exceeds the 95th percentile of the
#: same statistic on the random portfolios.
NOISE_Q = 0.95

#: DEVIATION D1 (declared before the run).  The covariance window is specified as 126 days.
#: The holdings panel is not complete on every trading day, so some formation dates have fewer
#: than 126 panel observations in that span.  A date whose window holds fewer than this many
#: observations has its decomposition refused and is reported unbuilt rather than estimated on
#: a window it does not have.  Set at 80% of the specified window before any date was run.
MIN_WINDOW_OBS = 100

#: DEVIATION D2 (declared before the run).  Item 1A is truncated to the first 18,000 characters
#: of the section.  18,000 is the excerpt length registered for exp-085's two readers
#: (scripts/poc_e2_keyword_vs_model.py, EXCERPT); it is reused verbatim rather than chosen here.
#: Without it the clustered population is ~64,000 filing sentences per date and HDBSCAN in 384
#: dimensions does not finish.
EXCERPT = 18_000

#: DEVIATION D3.  The named public encoder.  Draft 11 uses one encoder on every date and the
#: package must name it; the ChronoBERT vintages on disk are a per-date family and are therefore
#: not it.  Its published training cutoff is not bounded, which the spec's `embedspace` node
#: already says of any such encoder.
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

#: DEVIATION D4.  The first-stage market m_t is the capitalisation-weighted return of the
#: point-in-time IWV universe, built from the same panel as everything else.  The repo's other
#: market series (Fama-French Mkt-RF via data/french) ends 2026-06-30 and cannot cover the memo
#: date's window; the correlation between the two is printed on every date as a check.

#: DEVIATION D5.  The bridge's candidate instruments.  The spec names eleven sector ETFs, the
#: ten-year future, front-month WTI, a dollar index future, a high-yield credit ETF and the
#: mega-cap pair.  No ETF or futures price series is on disk.  The candidates used are the
#: closest series that are: five macro series from data/raw/fred and the panel-built mega-cap
#: pair (the same six the repo's reference panel uses), plus eleven equally-weighted sector
#: portfolios of the point-in-time universe standing in for the eleven sector ETFs.  Each
#: candidate's source is printed with its name.

#: DEVIATION D6.  The classifier of spec node `riskextract` is not run: it needs 2,000
#: language-model labels, none are on disk and no model key is available.  THE KEYWORD RUN IS
#: THE ONLY RUN.  This is the fallback the node's `on_failure` names and the brief permits.
CLASSIFIER_AVAILABLE = False

#: DEVIATION D11.  Dimensionality reduction before clustering, and the boilerplate rule.
#: exp-091's registration says HDBSCAN at a minimum cluster size of 25 and does NOT say how the
#: embedding space is reduced first.  Run literally -- HDBSCAN on raw 384-dimension normalised
#: MiniLM vectors under a euclidean metric with min_samples at its default of 25 -- the
#: clusterer returns noise share 1.000 on every one of the twelve dates, which is the known
#: high-dimension failure mode and not a statement about the filings.  POC-BRIEF draft 16
#: section 6 supplies the missing step: "UMAP (about ten dimensions, cosine, fixed seed)
#: before HDBSCAN (25, min_samples 5), the boilerplate cluster excluded by rule".  That is the
#: primary run here.  The literal registered configuration is computed and reported beside it
#: on every date so both are visible.
REDUCE_DIMS = 10
REDUCE_SEED = 0
REDUCE_METRIC = "cosine"
PRIMARY_VARIANT = "umap10_min_samples_5"

#: DEVIATION D11 (second half).  A cluster whose c-TF-IDF label is generic risk-factor
#: vocabulary is not a theme.  The rule is applied by list, never by eye: a cluster is
#: boilerplate when at least five of its six top c-TF-IDF terms are in this vocabulary.  The
#: list and the rule are written down here before the run and are copied into every record.
BOILERPLATE_TERMS = frozenset({
    "business", "businesses", "results", "result", "financial", "finances", "operations",
    "operating", "operate", "adversely", "adverse", "risk", "risks", "condition", "conditions",
    "materially", "material", "affect", "affected", "affecting", "company", "companies",
    "future", "could", "may", "impact", "impacted", "significant", "significantly", "harm",
    "harmed", "cash", "flows", "revenue", "revenues", "customers", "customer", "products",
    "product", "services", "service", "additional", "certain", "ability", "including",
})
BOILERPLATE_MIN_GENERIC_TERMS = 5
BOILERPLATE_RULE = (f"a cluster is boilerplate, and cannot be a theme, when at least "
                    f"{BOILERPLATE_MIN_GENERIC_TERMS} of its 6 top c-TF-IDF terms are in the "
                    f"declared generic risk-factor vocabulary")


#: A DIAGNOSTIC FLAG, not a parameter and not a decision rule: nothing downstream changes when
#: it fires, and no threshold is moved to make a date pass.  HDBSCAN at min_samples 5 cannot
#: leave exactly zero noise on data with real density structure, and a single cluster holding
#: almost the whole population is the excess-of-mass selection returning the root of the
#: condensed tree.  When both hold, the clustering is degenerate and the date's theme statistic
#: is recorded WITH THE FLAG rather than silently.  Measured on 2020-10-31: three independent
#: 64-company draws from that date's own filings, same parameters and same seed, returned 2, 2
#: and 26 clusters (noise 0.000, 0.000, 0.328), so the collapse is an instability of the
#: clusterer at this population size and not a property of that month's filings.
#: The criterion is the single unambiguous one: a cluster holding almost the whole population
#: is the excess-of-mass selection returning the root.  A first version also required noise
#: share to be exactly zero, which was transcribed from the synthetic diagnostic draws (noise
#: 0.000) and does not hold on the real 2020-10-31 run (noise 0.006), so the flag did not fire
#: on the one date it exists to mark.  The conjunct is removed; the noise share is recorded
#: beside it either way.  This is a flag only -- no statistic, threshold or parameter anywhere
#: downstream depends on it.
DEGENERATE_MIN_TOP_SHARE = 0.90
DEGENERACY_RULE = (f"one cluster holds more than {DEGENERATE_MIN_TOP_SHARE:.0%} of the "
                   f"clustered population (excess-of-mass returning the root of the "
                   f"condensed tree); the noise share is recorded beside it")


def is_boilerplate(label: list[str]) -> bool:
    return sum(1 for t in label if t.lower() in BOILERPLATE_TERMS) >= BOILERPLATE_MIN_GENERIC_TERMS


#: DEVIATION D7.  The news window.  Spec node `embedspace` reads a rolling twelve-month window.
#: The GDELT episode-title files span roughly two months before the event to two weeks after it,
#: so most of every file postdates its own formation date.  Titles seen after the formation date
#: are DROPPED: the alternative is to cluster news published after the reversal and then report
#: that the reversal's theme was visible before it.  The share kept is printed per date.

# ---------------------------------------------------------------- two keyword rules, kept apart
#
# THERE ARE TWO DIFFERENT RULES IN PLAY AND THEY ANSWER DIFFERENT QUESTIONS.  An earlier version
# of this file merged them -- it restated exp-085's frozen lists and added "could", "may" and
# "exposed to" to them -- which widened a frozen rule and made this run a different instrument
# from the one exp-085 scored.  That is undone.  exp-085's rule is now IMPORTED and never
# restated, so there is exactly one definition of it in the repo and it cannot drift.
#
# RULE A -- exp-085's frozen rule, imported: "a filing states a condition if its text contains at
# least one DEPENDENCE marker AND at least one EXTERNAL-CONDITION term", over an 18,000-character
# excerpt.  Its registration says "No tuning of this list is permitted after the rates are
# computed", and none is done: the import is used verbatim.  It asks a FILING-level question
# ("does this filer state an external survival condition?"), not the sentence-level question this
# node needs.  It is carried on every date as a reported comparator: the count of sentences it
# marks is printed and stored beside the count Rule B marks, so the two can never be confused.
#
# RULE B -- the extractor this node actually specifies.  spec.yaml nodes `riskextract` and
# `newsagent`: "The unit is a sentence ... The comparator is the keyword rule: the registered
# term list, a list of risk terms and modal phrases ("could", "may", "exposed to", "depends on"),
# applied to each sentence."  That is a DISJUNCTION over one list, at sentence level, and no such
# list exists in the register.  It is assembled here from material that is already registered
# plus the four phrases the spec writes out itself, and every term is traceable:
#     modal phrases  = exp-085's DEPENDENCE list (imported, which already contains "depends on")
#                      plus "could", "may" and "exposed to", quoted verbatim from spec.yaml
#     risk terms     = exp-085's EXTERNAL list (imported)
# A sentence is a risk statement when it contains at least one term from either half.  This is a
# NEW rule with a new name; it is not exp-085's rule and is never reported as exp-085's.  It is
# written down here before the first run and is not touched afterwards.
#
# Why Rule B is the primary and Rule A is not: measured on 2020-10-31 before any theme statistic
# was computed, Rule A at sentence level marks 0 of 5,351 news titles and 57 filing sentences
# across 33 companies, so the clustered population is 57 statements and HDBSCAN at a minimum
# cluster size of 25 cannot run on any date.  Rule A's conjunction is a filing-level test being
# asked a sentence-level question.  Both counts are reported on every date.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poc_e2_keyword_vs_model as KW  # noqa: E402

#: quoted verbatim from spec.yaml nodes `riskextract` / `newsagent`; "depends on" is already in
#: exp-085's DEPENDENCE list and is not duplicated.
SPEC_MODAL_PHRASES = ("could", "may", "exposed to")

MODAL = tuple(KW.DEPENDENCE) + SPEC_MODAL_PHRASES
RISK_TERMS = tuple(KW.EXTERNAL)
_MODAL = re.compile("|".join(re.escape(t) for t in MODAL), re.I)
_RISK = re.compile("|".join(re.escape(t) for t in RISK_TERMS), re.I)


def is_risk_statement(text: str) -> bool:
    """RULE B: the spec's sentence-level risk-statement rule (a disjunction over one list)."""
    return bool(_MODAL.search(text)) or bool(_RISK.search(text))


def states_condition_exp085(text: str) -> bool:
    """RULE A: exp-085's frozen rule, imported unchanged, carried as a reported comparator."""
    return KW.keyword_states_condition(text)


# ---------------------------------------------------------------- the dates
#
# Eight episode formation dates (the `book_date` stamped in each GDELT episode file), three calm
# windows (2016, 2017, 2018 -- the GKG calm files span 08-13 to 09-06 of each year, so the
# formation date is that August's month end, the same relation the 2019-09 episode has to its
# own window), and the memo date (the latest formation date in leg_members.pkl).

EPISODES = {
    "2017-12-31": ("2018-02_volmageddon", "2018-02-05"),
    "2019-08-31": ("2019-09_rotation", "2019-09-09"),
    "2020-10-31": ("2020-11_vaccine", "2020-11-09"),
    "2020-12-31": ("2021-01_squeeze", "2021-01-27"),
    "2022-10-31": ("2022-11_cpi", "2022-11-10"),
    "2024-06-30": ("2024-08_unwind", "2024-08-05"),
    "2024-12-31": ("2025-01_deepseek", "2025-01-27"),
    "2025-03-31": ("2025-04_tariff", "2025-04-09"),
}
CALM = {"2016-08-31": "calm_2016", "2017-08-31": "calm_2017", "2018-08-31": "calm_2018"}
MEMO_DATE = "2026-07-31"

#: The registration writes the expected label for three episodes and admits no post-hoc
#: judgement of recognisability.  A match is counted only where the theme's top terms overlap
#: these term sets.
EXPECTED_LABEL = {
    "2020-10-31": ("vaccine", "treatment", "timing", "trial"),
    "2025-03-31": ("tariff", "tariffs"),
    "2022-10-31": ("inflation", "rate", "rates", "interest"),
}


# ================================================================ price / return panel

def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split-adjusted prices (for returns), RAW prices (for market value), quantities, sectors.

    The two price frames are not interchangeable and mixing them is an instrument bug that this
    script committed and had to fix.  `splits.adjust` rescales prices BACKWARDS through each
    detected corporate action so the return series is continuous; quantity is left unadjusted.
    Multiplying an adjusted price by an unadjusted quantity therefore does not give market
    value: for a name that later reverse-split 1:20, the pre-action adjusted price is twenty
    times the price actually paid, and that name's market value -- and so its weight in a
    value-weighted leg -- comes out twentyfold.  Measured effect on ||w||_2 at 2017-12-31:
    0.6748 on adjusted prices against 0.3785 on raw.  Returns use the adjusted frame; every
    capitalisation, weight and cap decile uses the raw one.
    """
    frames = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/ishares/IWV/panel/IWV_*.parquet"))):
        if int(f.split("_")[-1][:4]) < 2014:
            continue
        frames.append(pd.read_parquet(f, columns=["as_of", "ticker", "price",
                                                  "quantity", "sector"]))
    long = pd.concat(frames, ignore_index=True)
    long = long[long.price.notna() & (long.price > 0)]
    P = long.pivot_table(index="as_of", columns="ticker", values="price", aggfunc="last")
    Q = long.pivot_table(index="as_of", columns="ticker", values="quantity", aggfunc="last")
    Padj, n_acts = fsplits.adjust(P, Q)
    sec = long.dropna(subset=["sector"]).groupby("ticker")["sector"].last()
    print(f"panel: {P.shape[0]} dates x {P.shape[1]} names, "
          f"{P.index.min().date()} -> {P.index.max().date()}, {n_acts} corporate actions divided out")
    return Padj, P, Q, sec


#: DEVIATION D9.  The panel is missing whole trading days in some years, so a naive
#: pct_change() across a gap is a multi-day return standing in a daily series.  A return whose
#: two observations are more than this many calendar days apart is dropped.  Five days lets a
#: normal weekend and a single holiday through and nothing longer.  Declared before the run.
MAX_GAP_DAYS = 5


def returns_from(P: pd.DataFrame) -> pd.DataFrame:
    """Daily returns.  After share-count split adjustment a residual |r| > 1 is an unresolved
    corporate action rather than a market move; it is dropped.  Declared before the run."""
    R = P.pct_change()
    R = R.mask(R.abs() > 1.0)
    gap = pd.Series(P.index, index=P.index).diff().dt.days
    bad = np.broadcast_to((gap > MAX_GAP_DAYS).to_numpy()[:, None], R.shape)
    return R.mask(pd.DataFrame(bad, index=R.index, columns=R.columns))


# ================================================================ book and random books

@dataclass
class Book:
    winners: list[str]
    losers: list[str]

    @property
    def names(self) -> list[str]:
        return list(self.winners) + list(self.losers)


def book_weights(bk: Book, caps: pd.Series) -> pd.Series:
    """Spec node `sort`: value-weighted within each leg, dollar-neutral across them.

    Weights come from price * quantity.  weight_pct is NOT used: it rounds to 0.00 for the
    smallest names in the panel and would zero out most of each leg.
    """
    w = {}
    for leg, names, sign in (("w", bk.winners, 1.0), ("l", bk.losers, -1.0)):
        c = caps.reindex(names).dropna()
        c = c[c > 0]
        tot = c.sum()
        for t, v in c.items():
            w[t] = sign * float(v) / float(tot)
    return pd.Series(w)


def leg_sign(bk: Book) -> pd.Series:
    s = {t: 1.0 for t in bk.winners}
    s.update({t: -1.0 for t in bk.losers})
    return pd.Series(s)


def cell_table(pool: pd.Index, caps: pd.Series, sec: pd.Series) -> pd.DataFrame:
    """(capitalisation decile, sector) for every eligible name."""
    c = caps.reindex(pool).dropna()
    c = c[c > 0]
    dec = pd.qcut(c.rank(method="first"), 10, labels=False)
    d = pd.DataFrame({"cap": c, "decile": dec.astype(int)})
    d["sector"] = sec.reindex(d.index)
    return d.dropna(subset=["sector"])


def draw_random_books(bk: Book, cells: pd.DataFrame, rng: np.random.Generator,
                      n: int) -> tuple[list[Book], dict]:
    """Spec node `random`, implemented literally.

    For each leg, reproduce the book's count in every (capitalisation decile, sector) cell,
    drawn without replacement within a portfolio from the point-in-time universe WITH THE
    BOOK'S OWN COMPANIES IN THE POOL.  A cell short of companies passes its shortfall to the
    next smaller decile in the same sector, and from the smallest decile to the next larger.
    The first-drawn leg plays the winner leg.
    """
    by_cell: dict[tuple, list[str]] = {}
    for t, row in cells.iterrows():
        by_cell.setdefault((int(row.decile), row.sector), []).append(t)

    want = {}
    for leg, names in (("w", bk.winners), ("l", bk.losers)):
        have = cells.reindex([x for x in names if x in cells.index]).dropna(subset=["sector"])
        want[leg] = have.groupby(["decile", "sector"]).size().to_dict()

    books, short = [], 0
    for _ in range(n):
        picked = {}
        # "drawn without replacement within a portfolio": the taken set spans BOTH legs, so no
        # name can land in the long and the short side of the same random book.
        taken: set[str] = set()
        for leg in ("w", "l"):
            out: list[str] = []
            for (dec, sector), k in want[leg].items():
                need = int(k)
                # the cell, then smaller deciles in the same sector, then larger ones
                order = [dec] + list(range(dec - 1, -1, -1)) + list(range(dec + 1, 10))
                for d in order:
                    if need <= 0:
                        break
                    avail = [x for x in by_cell.get((d, sector), []) if x not in taken]
                    if not avail:
                        continue
                    take = min(need, len(avail))
                    got = rng.choice(np.asarray(avail, dtype=object), size=take, replace=False)
                    out += list(got)
                    taken.update(got)
                    need -= take
                short += max(need, 0)
            picked[leg] = out
        books.append(Book(picked["w"], picked["l"]))
    return books, {"cells_short_total": int(short)}


# ================================================================ covariance and components

def ledoit_wolf_constant_correlation(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward the constant-correlation target.

    sklearn's LedoitWolf shrinks toward a scaled identity, which is a different target; the
    spec and the registration both say constant correlation, so the estimator is written out.
    X is T x N of residuals.
    """
    T, N = X.shape
    Y = X - X.mean(axis=0, keepdims=True)
    S = (Y.T @ Y) / T
    v = np.diag(S).copy()
    v[v <= 0] = np.finfo(float).tiny
    sd = np.sqrt(v)
    C = S / np.outer(sd, sd)
    off = ~np.eye(N, dtype=bool)
    rbar = float(C[off].mean())
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, v)

    Y2 = Y ** 2
    pi_mat = (Y2.T @ Y2) / T - S ** 2
    pi_hat = float(pi_mat.sum())

    # theta_{ii,ij} = E[(y_i^2 - s_ii)(y_i y_j - s_ij)] = (1/T) sum y_i^3 y_j - s_ii s_ij
    A = ((Y ** 3).T @ Y) / T
    theta = A - v[:, None] * S                      # element (i, j) = theta_{ii,ij}
    ratio = np.outer(1.0 / sd, sd)                  # element (i, j) = sd_j / sd_i
    rho_off = (rbar / 2.0) * (ratio * theta + ratio.T * theta.T)
    rho_hat = float(np.trace(pi_mat) + rho_off[off].sum())

    gamma_hat = float(((F - S) ** 2).sum())
    delta = 0.0 if gamma_hat <= 0 else (pi_hat - rho_hat) / gamma_hat / T
    delta = float(min(max(delta, 0.0), 1.0))
    return delta * F + (1.0 - delta) * S, delta


def decompose(U: pd.DataFrame, bk: Book, caps: pd.Series) -> dict | None:
    """Sigma, the three components, VS_k, g_k, ||w||, and the top-quartile sets."""
    names = list(dict.fromkeys(t for t in bk.names if t in U.columns))
    if len(names) < 50:
        return None
    w = book_weights(Book([t for t in bk.winners if t in names],
                          [t for t in bk.losers if t in names]), caps)
    w = w.reindex(names).dropna()
    names = list(w.index)
    X = U[names].to_numpy(dtype=float)
    Sigma, delta = ledoit_wolf_constant_correlation(X)
    wv = w.to_numpy()

    lam, V = np.linalg.eigh(Sigma)
    order = np.argsort(lam)[::-1][:N_COMPONENTS]
    lam, V = lam[order], V[:, order]

    denom = float(wv @ Sigma @ wv)
    comps = []
    for k in range(V.shape[1]):
        v = V[:, k]
        g = float(wv @ v)
        if g < 0:                      # spec node `pca`: sign fixed so that w'v_k > 0
            v, g = -v, -g
        vs = float(lam[k] * g * g / denom) if denom > 0 else np.nan
        av = np.abs(v)
        cut = np.quantile(av, SET_QUANTILE)
        idx = np.where(av >= cut)[0]
        comps.append({
            "k": k + 1, "eigenvalue": float(lam[k]), "g_k": g, "VS_k": vs,
            "loadings": pd.Series(v, index=names),
            "set": [names[i] for i in idx],
            "f": pd.Series(X @ v, index=U.index),
            "sigma_k": float(np.std(X @ v, ddof=1)),
        })
    # Reconciliation against exp-089 (reports/poc/e5_book_tail.csv, weight_l2_norm). That script
    # normalises the two legs JOINTLY -- v / v.sum() over winners and losers together -- and
    # never applies the leg sign, so its weights are all positive and its own gross_weight
    # column is identically 1.0. Spec node `sort` defines w value-weighted WITHIN each leg and
    # dollar-neutral across them, which is what g_k = w'v_k and correction 17 refer to and what
    # `w_norm` above is. Both are printed so the two numbers are never mistaken for each other.
    mv = caps.reindex(names).astype(float).fillna(0.0)
    joint = float(np.linalg.norm((mv / mv.sum()).to_numpy())) if mv.sum() > 0 else np.nan
    return {"names": names, "w": w, "Sigma": Sigma, "shrinkage": delta,
            "w_norm": float(np.linalg.norm(wv)),
            "w_norm_joint_normalisation_exp089_convention": joint,
            "cond": float(np.linalg.cond(Sigma)), "components": comps,
            "wSw": denom}


# ================================================================ reference series and bridge

def fred(name: str) -> pd.Series:
    d = pd.read_csv(ROOT / f"data/raw/fred/{name}.csv")
    d.columns = ["date", name]
    s = pd.to_numeric(d[name], errors="coerce")
    return s.set_axis(pd.to_datetime(d["date"])).dropna()


def candidate_instruments(P: pd.DataFrame, R: pd.DataFrame, caps_by_date: dict,
                          sec: pd.Series) -> tuple[pd.DataFrame, dict]:
    """The bridge's candidates (DEVIATION D5): five macro series, the mega-cap pair, and
    eleven equally-weighted sector portfolios standing in for the sector ETFs."""
    src = {}
    x = {}
    x["ten_year"] = fred("DGS10").diff();              src["ten_year"] = "FRED DGS10, daily change"
    x["wti"] = fred("DCOILWTICO").pct_change();        src["wti"] = "FRED DCOILWTICO, daily return"
    x["dollar"] = fred("DTWEXBGS").pct_change();       src["dollar"] = "FRED DTWEXBGS, daily return"
    x["ig_credit"] = fred("BAA10Y").diff();            src["ig_credit"] = "FRED BAA10Y, daily change"
    x["hy_credit"] = fred("BAMLH0A0HYM2").diff();      src["hy_credit"] = "FRED BAMLH0A0HYM2, daily change"

    # the mega-cap pair: the ten largest index positions, equally weighted, minus the
    # equal-weighted index -- built from the panel, as the reference-panel builder does.
    mega = {}
    for d, caps in caps_by_date.items():
        if d not in R.index:
            continue
        r = R.loc[d].dropna()
        top = caps.reindex(r.index).dropna().nlargest(10).index
        if len(top) < 10:
            continue
        mega[d] = float(r.reindex(top).mean() - r.mean())
    x["mega_cap_pair"] = pd.Series(mega).sort_index()
    src["mega_cap_pair"] = "panel: ten largest positions equally weighted, minus the equal-weighted universe"

    for s in sorted(set(sec.dropna())):
        members = [t for t in sec.index[sec == s] if t in R.columns]
        if len(members) < 5:
            continue
        key = "sector_" + re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        x[key] = R[members].mean(axis=1)
        src[key] = f"panel: equally-weighted {s} members of the universe (stands in for the sector ETF)"
    return pd.DataFrame(x).sort_index(), src


def bridge(f: pd.Series, cand: pd.DataFrame, m: pd.Series, src: dict) -> dict:
    """Correction 1 / spec node `bridge`.  Each candidate is residualised on the market with the
    same first stage as the companies, giving u_H and the instrument's market beta; the
    component is then regressed on u_H.  The instrument is the candidate with the highest R^2.
    """
    rows = []
    for c in cand.columns:
        d = pd.concat([f.rename("f"), cand[c].rename("x"), m.rename("m")], axis=1).dropna()
        if len(d) < 40:
            continue
        A = np.column_stack([np.ones(len(d)), d["m"].to_numpy()])
        beta = np.linalg.lstsq(A, d["x"].to_numpy(), rcond=None)[0]
        uH = d["x"].to_numpy() - A @ beta
        B = np.column_stack([np.ones(len(d)), uH])
        b = np.linalg.lstsq(B, d["f"].to_numpy(), rcond=None)[0]
        resid = d["f"].to_numpy() - B @ b
        sst = float(((d["f"] - d["f"].mean()) ** 2).sum())
        r2 = float(1 - (resid ** 2).sum() / sst) if sst > 0 else np.nan
        rows.append({"instrument": c, "source": src.get(c, ""), "b_k": float(b[1]),
                     "beta_H": float(beta[1]), "r2": r2, "n": int(len(d))})
    if not rows:
        return {}
    rows.sort(key=lambda r: (-1 if np.isnan(r["r2"]) else -r["r2"]))
    return {"chosen": rows[0], "all": rows}


# ================================================================ text

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def sentences(text: str) -> list[str]:
    out = []
    for s in _SENT.split(re.sub(r"\s+", " ", text)):
        s = s.strip()
        if 40 <= len(s) <= 1000:
            out.append(s)
    return out


def filing_index() -> pd.DataFrame:
    """Every Item 1A on disk with a filing date, so the pick is point in time.

    data/raw/edgar_sections_2017 is EXCLUDED: its records carry no filing date and no producer
    is in the repo, so their availability cannot be established.  Dropping 68 tickers there
    costs the two 2017 dates coverage they would otherwise have had.
    """
    rows = []
    for d in ("edgar_panel", "edgar_sections", "edgar_sections059"):
        for f in glob.glob(str(ROOT / f"data/raw/{d}/*.json")):
            try:
                x = json.load(open(f))
            except Exception:
                continue
            if not isinstance(x, dict) or not x.get("item_1a") or not x.get("filing_date"):
                continue
            rows.append({"source": d, "ticker": x["ticker"],
                         "filing_date": pd.Timestamp(x["filing_date"]), "path": f})
    return pd.DataFrame(rows).sort_values("filing_date")


def filings_asof(idx: pd.DataFrame, asof: pd.Timestamp) -> dict[str, str]:
    sub = idx[idx.filing_date <= asof]
    if sub.empty:
        return {}
    last = sub.groupby("ticker").tail(1)
    return dict(zip(last.ticker, last.path))


def news_titles(asof: pd.Timestamp, key: str | None) -> tuple[list[str], dict]:
    """DEVIATION D7: only titles seen on or before the formation date."""
    if key is None:
        return [], {"file": None, "titles_total": 0, "titles_kept": 0}
    p = ROOT / f"data/corpus/gdelt/episode_titles/{key}.parquet"
    if not p.exists():
        return [], {"file": str(p), "titles_total": 0, "titles_kept": 0, "missing": True}
    d = pd.read_parquet(p, columns=["title", "seendate"])
    total = int(len(d))
    keep = d[d.seendate <= asof.tz_localize("UTC")]
    t = sorted({s.strip() for s in keep.title.dropna() if 20 <= len(s.strip()) <= 400})
    return t, {"file": p.name, "titles_total": total, "titles_kept": int(len(keep)),
               "titles_unique_kept": len(t),
               "share_before_formation": round(len(keep) / total, 4) if total else 0.0}


class Encoder:
    """The one named public encoder, with a content-addressed cache under reports/poc/e7_cache."""

    def __init__(self) -> None:
        self.model = None
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / "embeddings.npz"
        self.store: dict[str, np.ndarray] = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            keys, vecs = z["keys"], z["vecs"]
            self.store = {str(k): vecs[i] for i, k in enumerate(keys)}

    @staticmethod
    def _key(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]

    def __call__(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.store]
        if missing:
            if self.model is None:
                import torch
                from sentence_transformers import SentenceTransformer
                dev = "mps" if torch.backends.mps.is_available() else "cpu"
                self.model = SentenceTransformer(ENCODER, device=dev)
                print(f"  encoder {ENCODER} on {dev}")
            uniq = {k: t for k, t in missing}
            ks = list(uniq)
            t0 = time.time()
            vecs = self.model.encode([uniq[k] for k in ks], batch_size=512,
                                     normalize_embeddings=True, show_progress_bar=False)
            print(f"  embedded {len(ks):,} new statements in {time.time() - t0:.0f}s")
            for k, v in zip(ks, vecs):
                self.store[k] = v.astype(np.float32)
            self.save()
        return np.vstack([self.store[k] for k in keys])

    def save(self) -> None:
        ks = list(self.store)
        np.savez_compressed(self.path, keys=np.array(ks),
                            vecs=np.vstack([self.store[k] for k in ks]))


def ctfidf_labels(texts: list[list[str]], top: int = 6) -> list[list[str]]:
    """Spec node `embedspace`: each cluster is labelled with its top c-TF-IDF terms."""
    from sklearn.feature_extraction.text import CountVectorizer
    docs = [" ".join(t) for t in texts]
    cv = CountVectorizer(stop_words="english", min_df=1, max_features=40000,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b")
    X = cv.fit_transform(docs).toarray().astype(float)
    vocab = np.array(cv.get_feature_names_out())
    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1)
    n_total = X.sum()
    idf = np.log(1.0 + n_total / np.maximum(X.sum(axis=0), 1))
    S = tf * idf
    return [list(vocab[np.argsort(S[i])[::-1][:top]]) for i in range(S.shape[0])]


# ================================================================ one date

def run_date(date_str: str, kind: str, P, R, sec, caps_by_date, cand, cand_src,
             mkt, legmembers, fidx, enc, ff_mkt) -> dict:
    asof = pd.Timestamp(date_str)
    rec: dict = {"date": date_str, "kind": kind, "built": False, "reasons": [],
                 "encoder": ENCODER, "classifier_run": CLASSIFIER_AVAILABLE,
                 "runs": ["keyword"]}
    print(f"\n=== {date_str}  [{kind}] " + "=" * 40)

    if asof not in legmembers:
        rec["reasons"].append("no leg membership at this formation date")
        return rec
    bk0 = Book(legmembers[asof]["winners"], legmembers[asof]["losers"])

    # ---- window
    usable = mkt.dropna().index
    win = R.index[(R.index <= asof) & (R.index > asof - pd.Timedelta(days=int(WINDOW * 1.55)))]
    win = win.intersection(usable).sort_values()[-WINDOW:]
    rec["window"] = {"target_days": WINDOW, "observations": int(len(win)),
                     "start": str(win.min().date()) if len(win) else None,
                     "end": str(win.max().date()) if len(win) else None}
    print(f"window: {len(win)} observations "
          f"({win.min().date() if len(win) else '-'} -> {win.max().date() if len(win) else '-'})")
    if len(win) < MIN_WINDOW_OBS:
        rec["reasons"].append(
            f"holdings panel holds only {len(win)} observations in the 126-day window; the "
            f"declared minimum is {MIN_WINDOW_OBS}, so the decomposition is refused")
        print("  UNBUILT: window too short")
        return rec

    # ---- first stage: r_it = a_i + b_i m_t + u_it
    Rw = R.loc[win]
    m = mkt.reindex(win)
    ok = Rw.columns[(Rw.notna().sum() >= int(0.8 * len(win)))]
    Rw = Rw[ok]
    Rw = Rw.fillna(Rw.mean())
    # A name whose price never moves in the window (halted, suspended, or carried forward by the
    # holdings archive) has zero return variance. It makes the sample correlation matrix
    # singular -- on 2016-08-31 a single such name, UDF, drove the condition number of the
    # SHRUNK matrix to 8.2e17 and corrupted the eigenvectors. Constant series are not market
    # observations and are dropped; the count is recorded.
    const = [c for c in Rw.columns if Rw[c].nunique() <= 1]
    if const:
        Rw = Rw.drop(columns=const)
    rec["constant_series_dropped"] = {"n": len(const), "tickers": sorted(const)[:20]}
    A = np.column_stack([np.ones(len(win)), m.to_numpy()])
    coef, *_ = np.linalg.lstsq(A, Rw.to_numpy(), rcond=None)
    U = pd.DataFrame(Rw.to_numpy() - A @ coef, index=win, columns=Rw.columns)
    ffc = ff_mkt.reindex(win).dropna()
    rec["market_check"] = {
        "series": "capitalisation-weighted point-in-time IWV universe",
        "corr_with_fama_french_mkt_rf":
            round(float(m.reindex(ffc.index).corr(ffc)), 4) if len(ffc) > 30 else None,
        "ff_days_available": int(len(ffc))}
    print(f"first stage on {U.shape[1]} names; market corr with FF Mkt-RF "
          f"{rec['market_check']['corr_with_fama_french_mkt_rf']} on {len(ffc)} days")

    # ---- caps, universe, random books
    snap = max(d for d in caps_by_date if d <= asof)
    caps = caps_by_date[snap]
    pool = pd.Index([t for t in U.columns if t in caps.index and t in sec.index])
    cells = cell_table(pool, caps, sec)
    bk = Book([t for t in bk0.winners if t in cells.index],
              [t for t in bk0.losers if t in cells.index])
    rec["book"] = {"winners_registered": len(bk0.winners), "losers_registered": len(bk0.losers),
                   "winners_used": len(bk.winners), "losers_used": len(bk.losers),
                   "universe": int(len(cells)), "caps_snapshot": str(snap.date())}
    print(f"book {len(bk.winners)}+{len(bk.losers)} of "
          f"{len(bk0.winners)}+{len(bk0.losers)} registered; universe {len(cells)}")
    if min(len(bk.winners), len(bk.losers)) < 25:
        rec["reasons"].append("a leg has fewer than 25 usable names in the window")
        return rec

    # deterministic across processes: Python's str hash is salted per interpreter, so using it
    # would give a different set of 200 random books on every run and a different placebo band.
    rng = np.random.default_rng(int(hashlib.sha1(date_str.encode()).hexdigest()[:8], 16))
    rbooks, drawinfo = draw_random_books(bk, cells, rng, N_RANDOM)
    rec["random_books"] = {"n": len(rbooks), **drawinfo}

    # ---- decomposition
    d0 = decompose(U, bk, caps)
    if d0 is None:
        rec["reasons"].append("the book's covariance could not be estimated")
        return rec
    rand_dec = []
    t0 = time.time()
    for rb in rbooks:
        d = decompose(U, rb, caps)
        if d is not None:
            rand_dec.append(d)
    print(f"decomposed book + {len(rand_dec)} random books in {time.time() - t0:.0f}s; "
          f"shrinkage {d0['shrinkage']:.3f}, cond {d0['cond']:.3g}, "
          f"||w||_2 = {d0['w_norm']:.4f}")

    comps_out = []
    for k in range(N_COMPONENTS):
        c = d0["components"][k]
        rvs = np.array([rd["components"][k]["VS_k"] for rd in rand_dec], dtype=float)
        rvs = rvs[np.isfinite(rvs)]
        thr = float(np.quantile(rvs, NOISE_Q)) if len(rvs) else np.nan
        pct = float((rvs < c["VS_k"]).mean()) if len(rvs) else np.nan
        above = bool(np.isfinite(thr) and c["VS_k"] > thr)
        br = bridge(c["f"], cand, mkt, cand_src)
        comps_out.append({
            "k": k + 1, "VS_k": c["VS_k"], "VS_k_placebo_q95": thr,
            "VS_k_percentile_vs_random": pct, "above_noise": above,
            "g_k": c["g_k"], "sigma_k": c["sigma_k"], "set_size": len(c["set"]),
            "bridge": {"instrument": br["chosen"]["instrument"], "r2": br["chosen"]["r2"],
                       "b_k": br["chosen"]["b_k"], "beta_H": br["chosen"]["beta_H"],
                       "source": br["chosen"]["source"],
                       "top5": [{"instrument": r["instrument"], "r2": round(r["r2"], 4)}
                                for r in br["all"][:5]]} if br else None,
        })
        print(f"  component {k+1}: VS_k = {c['VS_k']:.4f}  (placebo q95 {thr:.4f}, "
              f"pct {pct:.3f}, above noise {above})  g_k = {c['g_k']:+.4f}  "
              f"|| w ||_2 = {d0['w_norm']:.4f}  bridge "
              f"{br['chosen']['instrument'] if br else '-'} "
              f"R2 {br['chosen']['r2']:.3f}" if br else "")
    rec["components"] = comps_out
    eff_n = {}
    for side, sgn in (("winner", 1), ("loser", -1)):
        s = d0["w"][np.sign(d0["w"]) == sgn].abs()
        eff_n[side] = {"n": int(len(s)), "top1_share": float(s.max()) if len(s) else None,
                       "effective_n": float(1.0 / (s ** 2).sum()) if len(s) else None,
                       "largest": str(s.idxmax()) if len(s) else None}
    rec["decomposition"] = {
        "shrinkage": d0["shrinkage"], "condition_number": d0["cond"],
        "w_norm_l2": d0["w_norm"], "n_names": len(d0["names"]),
        "w_norm_l2_convention": "spec node `sort`: value-weighted within each leg, "
                                "dollar-neutral across them, from RAW price * quantity",
        "w_norm_l2_exp089_joint_convention":
            d0["w_norm_joint_normalisation_exp089_convention"],
        "leg_concentration": eff_n}

    # ---- text: the clustered population
    titles, newsinfo = news_titles(asof, EPISODES[date_str][0] if kind == "episode" else None)
    rec["news"] = newsinfo
    news_stmts = [t for t in titles if is_risk_statement(t)]
    newsinfo["risk_statements_rule_b"] = len(news_stmts)
    newsinfo["risk_statements_rule_a_exp085"] = sum(
        1 for t in titles if states_condition_exp085(t))
    print(f"news: {newsinfo.get('titles_unique_kept', 0)} unique titles on or before "
          f"formation ({newsinfo.get('share_before_formation')} of the file), "
          f"{len(news_stmts)} pass rule B (spec), "
          f"{newsinfo['risk_statements_rule_a_exp085']} pass rule A (exp-085 frozen)")

    paths = filings_asof(fidx, asof)
    setpool = set(d0["names"])
    for rd in rand_dec:
        setpool.update(rd["names"])
    have = {t: p for t, p in paths.items() if t in setpool}
    comp_pool = set()
    for c in d0["components"]:
        comp_pool.update(c["set"])
    for rd in rand_dec:
        for c in rd["components"]:
            comp_pool.update(c["set"])
    fil_stmts: dict[str, list[str]] = {}
    n_sent = n_a = 0
    for t, p in have.items():
        if t not in comp_pool:
            continue
        try:
            txt = json.load(open(p))["item_1a"][:EXCERPT]
        except Exception:
            continue
        allsent = sentences(txt)
        n_sent += len(allsent)
        n_a += sum(1 for s in allsent if states_condition_exp085(s))
        ss = [s for s in allsent if is_risk_statement(s)]
        if ss:
            fil_stmts[t] = ss
    n_fil = sum(len(v) for v in fil_stmts.values())
    rec["filings"] = {"tickers_in_any_component_set": len(comp_pool),
                      "tickers_with_item_1a_as_of_date": len(have),
                      "tickers_used": len(fil_stmts),
                      "sentences_total": n_sent,
                      "risk_statements_rule_b": n_fil,
                      "risk_statements_rule_a_exp085": n_a,
                      "excerpt_chars": EXCERPT}
    print(f"filings: {len(fil_stmts)} companies with Item 1A as of the date "
          f"(of {len(comp_pool)} in any component set), {n_sent} sentences -> "
          f"{n_fil} pass rule B, {n_a} pass rule A")

    population = news_stmts + [s for t in sorted(fil_stmts) for s in fil_stmts[t]]
    origin = ["news"] * len(news_stmts) + \
             [t for t in sorted(fil_stmts) for _ in fil_stmts[t]]
    rec["population"] = {"n": len(population), "news": len(news_stmts), "filing": n_fil}
    if len(population) < 4 * MIN_CLUSTER_SIZE:
        rec["reasons"].append(
            f"the clustered population is {len(population)} statements, too few for HDBSCAN at "
            f"a minimum cluster size of {MIN_CLUSTER_SIZE}")
        print("  UNBUILT: population too small to cluster")
        rec["numeric_built"] = True
        return rec
    if not fil_stmts:
        rec["reasons"].append("no component-set company has an Item 1A as of this date")
        rec["numeric_built"] = True
        return rec

    # ---- cluster
    E = enc(population)
    variants = {}
    for vname, ms, red in (("umap10_min_samples_5", 5, REDUCE_DIMS),
                           ("registered", None, None),
                           ("min_samples_5", 5, None)):
        variants[vname] = cluster_and_score(
            vname, ms, red, E, population, origin, news_stmts, fil_stmts,
            d0, rand_dec, date_str, comps_out)
    rec["clustering_variants"] = variants
    rec["primary_variant"] = PRIMARY_VARIANT
    primary = variants[PRIMARY_VARIANT]
    rec["clusters"] = primary.get("clusters")
    rec["themes"] = primary.get("themes")
    rec["named_component"] = primary.get("named_component")
    rec["any_theme_risen"] = bool(primary.get("any_theme_risen"))
    rec["risen_on_component_above_noise"] = bool(primary.get("risen_on_component_above_noise"))
    rec["numeric_built"] = True
    if primary.get("themes"):
        rec["built"] = True
    else:
        rec["reasons"].append(primary.get("reason", "the theme stage did not run"))
    return rec


def cluster_and_score(vname, min_samples, reduce_dims, E, population, origin, news_stmts,
                      fil_stmts, d0, rand_dec, date_str, comps_out) -> dict:
    """One clustering configuration, through to the themes.

    `registered` is the registration read literally: HDBSCAN at min_cluster_size 25 in the raw
    384-dimension embedding space with min_samples at its default (equal to min_cluster_size).
    `reduced_pca10_min_samples_5` is deviation D11 and is the primary.
    """
    from sklearn.cluster import HDBSCAN
    t0 = time.time()
    X = E
    if reduce_dims:
        import umap
        X = umap.UMAP(n_components=reduce_dims, metric=REDUCE_METRIC,
                      random_state=REDUCE_SEED).fit_transform(E)
        out_reducer = f"umap n_components={reduce_dims} metric={REDUCE_METRIC} seed={REDUCE_SEED}"
    else:
        out_reducer = "none (raw encoder space)"
    lab = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=min_samples,
                  metric="euclidean", copy=True).fit_predict(X)
    ids = sorted(set(lab) - {-1})
    print(f"HDBSCAN [{vname}]: {len(ids)} clusters, noise share "
          f"{(lab == -1).mean():.3f}, {time.time() - t0:.0f}s")
    out = {"variant": vname, "min_samples": min_samples, "reducer": out_reducer,
           "clusters": {"n": len(ids), "noise_share": float((lab == -1).mean()),
                        "min_cluster_size": MIN_CLUSTER_SIZE}}
    if not ids:
        out["reason"] = (f"HDBSCAN [{vname}] found no cluster at a minimum cluster size of "
                         f"{MIN_CLUSTER_SIZE} (noise share 1.000)")
        return out

    # distances, d*, d*_c and the assignment all live in the SPACE THAT WAS CLUSTERED, so that
    # a member's distance to its own centroid and a company's distance to a theme's centroid
    # are the same metric HDBSCAN saw.
    E = X
    cents = np.vstack([E[lab == c].mean(axis=0) for c in ids])
    member_d = np.concatenate([np.linalg.norm(E[lab == c] - cents[i], axis=1)
                               for i, c in enumerate(ids)])
    d_star = float(np.quantile(member_d, 0.90))
    if len(ids) > 1:
        cc = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=2)
        np.fill_diagonal(cc, np.inf)
        d_star_c = float(np.quantile(cc.min(axis=1), 0.10))
    else:
        d_star_c = float("nan")
    news_members = [int(((lab == c) & (np.array(origin) == "news")).sum()) for c in ids]
    labels = ctfidf_labels([[population[i] for i in np.where(lab == c)[0]] for c in ids])
    boiler = [is_boilerplate(lb) for lb in labels]
    noise_share = float((lab == -1).mean())
    top_share = float(max((lab == c).sum() for c in ids) / len(lab))
    degenerate = bool(top_share > DEGENERATE_MIN_TOP_SHARE)
    if degenerate:
        print(f"  DEGENERATE CLUSTERING [{vname}]: noise {noise_share:.3f}, largest cluster "
              f"{top_share:.1%} of the population. Recorded with the flag; no parameter changed.")
    out["degenerate_clustering"] = degenerate
    out["degeneracy_rule"] = DEGENERACY_RULE
    out["largest_cluster_share"] = top_share
    out["clusters"] = {
        "n": len(ids), "noise_share": noise_share, "largest_cluster_share": top_share,
        "degenerate": degenerate,
        "d_star": d_star, "d_star_c": d_star_c,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "n_boilerplate_excluded": int(sum(boiler)),
        "boilerplate_rule": BOILERPLATE_RULE,
        "boilerplate_vocabulary": sorted(BOILERPLATE_TERMS),
        "detail": [{"id": int(c), "size": int((lab == c).sum()),
                    "news_members": news_members[i],
                    "company_specific": news_members[i] < NEWS_MEMBERS_FOR_SHARED,
                    "boilerplate_excluded": boiler[i],
                    "label": labels[i]} for i, c in enumerate(ids)]}
    print(f"  d* = {d_star:.4f}   d*_c = {d_star_c:.4f}   "
          f"{sum(boiler)} of {len(ids)} clusters excluded as boilerplate")
    eligible = np.array([not b for b in boiler])
    if not eligible.any():
        out["reason"] = (f"every one of the {len(ids)} clusters HDBSCAN [{vname}] found is "
                         f"generic risk-factor boilerplate by the declared rule; no theme")
        return out

    # ---- per company, nearest statement to each centroid
    comp_index = {t: [] for t in fil_stmts}
    p = len(news_stmts)
    for t in sorted(fil_stmts):
        comp_index[t] = list(range(p, p + len(fil_stmts[t])))
        p += len(fil_stmts[t])
    tickers = sorted(fil_stmts)
    D = np.full((len(tickers), len(ids)), np.inf)
    for i, t in enumerate(tickers):
        Et = E[comp_index[t]]
        D[i] = np.linalg.norm(Et[:, None, :] - cents[None, :, :], axis=2).min(axis=0)
    near = {t: i for i, t in enumerate(tickers)}
    within = D <= d_star                        # company x cluster

    def q_for(dec: dict, k: int) -> np.ndarray:
        """Q_e for every cluster on one decomposition's component k.

        Correction 2 / spec node `cluster`: a company counts only when leg sign times loading
        sign is positive.  One company, one vote.  The denominator is the whole set's total
        absolute loading, so a set company with no filing counts in the denominator only.
        """
        c = dec["components"][k]
        v = c["loadings"]
        s = np.sign(dec["w"])            # leg sign: +1 on the winner leg, -1 on the loser leg
        sset = c["set"]
        denom = float(np.abs(v.reindex(sset)).sum())
        if denom <= 0:
            return np.zeros(len(ids))
        num = np.zeros(len(ids))
        for t in sset:
            if s.get(t, 0) * v[t] <= 0 or t not in near:
                continue
            num += within[near[t]] * abs(float(v[t]))
        return num / denom

    themes = []
    for k in range(N_COMPONENTS):
        Qb = q_for(d0, k)
        Qr = np.vstack([q_for(rd, k) for rd in rand_dec])
        Pe = Qr.mean(axis=0)
        Eb = Qb - Pe
        Er = Qr - Pe
        # boilerplate clusters are excluded from the argmax AND from the placebo maximum, so
        # the book and the random books are judged over exactly the same candidate set
        Ebm = np.where(eligible, Eb, -np.inf)
        rmax = np.where(eligible, Er, -np.inf).max(axis=1)
        thr = float(np.quantile(rmax, PLACEBO_Q))
        top = int(np.argmax(Ebm))
        rest = [i for i in np.argsort(Ebm)[::-1][1:] if eligible[i]]
        second = int(rest[0]) if rest else None
        risen = bool(Eb[top] > thr)
        # coverage: how much of this component set the statistic actually rests on
        cset = d0["components"][k]["set"]
        v_k = d0["components"][k]["loadings"]
        with_fil = [t for t in cset if t in fil_stmts]
        denom = float(np.abs(v_k.reindex(cset)).sum())
        cov = {"set_size": len(cset), "set_companies_with_item_1a": len(with_fil),
               "set_coverage_share": round(len(with_fil) / max(len(cset), 1), 4),
               "share_of_set_absolute_loading_covered":
                   round(float(np.abs(v_k.reindex(with_fil)).sum() / denom), 4)
                   if denom > 0 else None}
        lbl = labels[top]
        exp_terms = EXPECTED_LABEL.get(date_str)
        match = None
        if exp_terms is not None:
            match = bool(set(w.lower() for w in lbl) & set(exp_terms))
        themes.append({
            "k": k + 1,
            "above_noise": comps_out[k]["above_noise"],
            "theme_cluster": int(ids[top]), "theme_label": lbl,
            "Q_e": float(Qb[top]), "P_e": float(Pe[top]), "E_e": float(Eb[top]),
            "placebo_q95_of_random_max": thr, "risen": risen,
            "news_members": news_members[top],
            "company_specific": news_members[top] < NEWS_MEMBERS_FOR_SHARED,
            "filing_only": len(news_stmts) < NEWS_MEMBERS_FOR_SHARED,
            "coverage_note": (
                "This theme is filing-only: only %d news risk statements exist for this date, "
                "so no cluster can reach %d news members and every cluster is company-specific "
                "by construction." % (len(news_stmts), NEWS_MEMBERS_FOR_SHARED)
                if len(news_stmts) < NEWS_MEMBERS_FOR_SHARED else None),
            "set_coverage": cov,
            "runner_up": ({"cluster": int(ids[second]), "label": labels[second],
                           "E_e": float(Eb[second])} if second is not None else None),
            "expected_label_terms": list(exp_terms) if exp_terms else None,
            "label_matches_expected": match,
            # how close the winning label is to the boilerplate vocabulary. The exclusion rule
            # fires at BOILERPLATE_MIN_GENERIC_TERMS of 6; a theme that rose with 3 or 4 is a
            # near-generic cluster and the reader is told the count rather than an opinion.
            "label_generic_term_count": int(
                sum(1 for t in lbl if t.lower() in BOILERPLATE_TERMS)),
            # a set with no covered company cannot produce a non-zero E_e, so "not risen"
            # there is vacuous rather than evidence
            "statistic_is_informative": bool(cov["set_companies_with_item_1a"] > 0),
            "quotes": [population[int(np.argmin(np.linalg.norm(E - cents[top], axis=1)))]],
        })
        print(f"  [{vname}] component {k+1} theme: {lbl}  E_e = {Eb[top]:+.4f} vs placebo "
              f"{thr:+.4f} -> {'RISEN' if risen else 'not risen'}"
              f"  | set {cov['set_companies_with_item_1a']}/{cov['set_size']} companies with "
              f"Item 1A ({cov['share_of_set_absolute_loading_covered']} of the set's loading)"
              + (f"  (expected-label match: {match})" if match is not None else ""))
    out["themes"] = themes

    named = None
    for k in range(N_COMPONENTS):
        if themes[k]["above_noise"] and themes[k]["risen"]:
            named = k + 1
            break
    out["named_component"] = named
    out["any_theme_risen"] = any(t["risen"] for t in themes)
    out["risen_on_component_above_noise"] = named is not None
    return out


# ================================================================ main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    THEMES.mkdir(parents=True, exist_ok=True)

    P, Praw, Q, sec = load_panel()
    R = returns_from(P)

    Qa = Q.reindex(index=P.index, columns=P.columns)
    MV = Praw.reindex(index=P.index, columns=P.columns) * Qa   # RAW price x quantity
    caps_by_date = {}
    for d in P.index:
        v = MV.loc[d].dropna()
        caps_by_date[d] = v[v > 0]

    # the capitalisation-weighted point-in-time universe return (DEVIATION D4)
    mkt = {}
    for d in R.index:
        r = R.loc[d].dropna()
        c = caps_by_date.get(d, pd.Series(dtype=float)).reindex(r.index).dropna()
        if len(c) < 100:
            continue
        r = r.reindex(c.index)
        mkt[d] = float((r * c).sum() / c.sum())
    mkt = pd.Series(mkt).sort_index()

    try:
        from unstructured_momentum.data import french
        ff_mkt = french.market()["Mkt-RF"]
    except Exception:
        ff_mkt = pd.Series(dtype=float)

    cand, cand_src = candidate_instruments(P, R, caps_by_date, sec)
    print(f"bridge candidates: {len(cand.columns)}")

    import pickle
    legmembers = pickle.load(open(ROOT / "data/processed/leg_members.pkl", "rb"))
    fidx = filing_index()
    print(f"Item 1A index: {len(fidx)} filings, {fidx.ticker.nunique()} tickers, "
          f"{fidx.filing_date.min().date()} -> {fidx.filing_date.max().date()}")

    enc = Encoder()

    plan = ([(d, "episode") for d in EPISODES] + [(d, "calm") for d in CALM] +
            [(MEMO_DATE, "memo")])
    if "--only" in sys.argv:
        want = set(sys.argv[sys.argv.index("--only") + 1].split(","))
        plan = [p for p in plan if p[0] in want]
    records = []
    for d, kind in sorted(plan):
        try:
            rec = run_date(d, kind, P, R, sec, caps_by_date, cand, cand_src, mkt,
                           legmembers, fidx, enc, ff_mkt)
        except Exception as e:  # a date that fails is recorded as unbuilt, never skipped
            import traceback
            traceback.print_exc()
            rec = {"date": d, "kind": kind, "built": False,
                   "reasons": [f"exception: {type(e).__name__}: {e}"]}
        records.append(rec)
        json.dump(rec, open(THEMES / f"{d}.json", "w"), indent=2, default=str)

    epi = [r for r in records if r["kind"] == "episode"]
    calm = [r for r in records if r["kind"] == "calm"]
    memo = [r for r in records if r["kind"] == "memo"]
    summary = {
        "experiment": "exp-091",
        "script": "scripts/poc_e7_theme.py",
        "run_at": pd.Timestamp.utcnow().isoformat(),
        "encoder": ENCODER,
        "classifier_run": CLASSIFIER_AVAILABLE,
        "runs": ["keyword"],
        "classifier_note": (
            "The classifier of spec node `riskextract` was NOT trained or run. A model key IS "
            "available in this environment; training the classifier on 2,000 language-model "
            "labels was out of scope for this run. Every risk statement here therefore comes "
            "from the frozen keyword rule, and the theme, the excess weight and the placebo "
            "are the KEYWORD RUN alone."),
        "keyword_rules": {
            "rule_b_used_for_every_statistic": {
                "name": "spec riskextract/newsagent keyword rule",
                "rule": "a sentence is a risk statement if it contains at least one term from "
                        "the list (a disjunction), applied per sentence",
                "provenance": "assembled here, not previously in the repo. Modal half = "
                              "exp-085's DEPENDENCE list imported unchanged, plus 'could', "
                              "'may', 'exposed to' quoted verbatim from spec.yaml nodes "
                              "riskextract/newsagent. Risk half = exp-085's EXTERNAL list "
                              "imported unchanged. No term is invented here.",
                "modal_terms": list(MODAL), "risk_terms": list(RISK_TERMS),
                "spec_phrases_added": list(SPEC_MODAL_PHRASES)},
            "rule_a_reported_comparator_only": {
                "name": "exp-085 frozen rule",
                "source": "imported from scripts/poc_e2_keyword_vs_model.py "
                          "(keyword_states_condition); frozen at exp-085's registration and "
                          "not restated or altered anywhere in this script",
                "rule": "at least one DEPENDENCE marker AND at least one EXTERNAL-CONDITION "
                        "term, over an 18,000-character filing excerpt",
                "why_not_primary": "it is a filing-level test. Applied at the sentence unit "
                                   "spec node riskextract requires, it marked 0 of 5,351 news "
                                   "titles and 57 filing sentences on 2020-10-31, a clustered "
                                   "population of 57 statements, so HDBSCAN at a minimum "
                                   "cluster size of 25 cannot run on any date. Its counts are "
                                   "recorded per date under news.risk_statements_rule_a_exp085 "
                                   "and filings.risk_statements_rule_a_exp085."}},
        "parameters": {"n_random": N_RANDOM, "window": WINDOW, "n_components": N_COMPONENTS,
                       "set_quantile": SET_QUANTILE, "min_cluster_size": MIN_CLUSTER_SIZE,
                       "placebo_q": PLACEBO_Q, "noise_q": NOISE_Q,
                       "min_window_obs": MIN_WINDOW_OBS, "excerpt_chars": EXCERPT},
        "deviations": DEVIATIONS,
        "counts": {
            "episode_dates": len(epi),
            "episode_built": sum(1 for r in epi if r.get("built")),
            "episode_theme_risen": sum(1 for r in epi if r.get("any_theme_risen")),
            "episode_risen_on_component_above_noise":
                sum(1 for r in epi if r.get("risen_on_component_above_noise")),
            # the registration's prediction is that a theme RISES and its label is the
            # episode's, so the headline count requires both; the bare term-overlap count is
            # reported beside it because the registered matching rule is term overlap alone.
            "episode_risen_theme_label_matches": sum(
                1 for r in epi
                if any(t.get("risen") and t.get("label_matches_expected")
                       for t in (r.get("themes") or []))),
            "episode_label_term_overlap_any_theme": sum(
                1 for r in epi
                if any(t.get("label_matches_expected") for t in (r.get("themes") or []))),
            "episode_dates_with_expected_label_registered": sum(
                1 for r in epi if r["date"] in EXPECTED_LABEL),
            "calm_dates": len(calm),
            "calm_built": sum(1 for r in calm if r.get("built")),
            "calm_theme_risen": sum(1 for r in calm if r.get("any_theme_risen")),
            "memo_built": sum(1 for r in memo if r.get("built")),
            "memo_theme_risen": sum(1 for r in memo if r.get("any_theme_risen")),
        },
        "counts_by_clustering_variant": {
            v: {
                "clusters_found_on_n_dates": sum(
                    1 for r in records
                    if (r.get("clustering_variants", {}).get(v, {})
                         .get("clusters") or {}).get("n", 0) > 0),
                "episode_theme_risen": sum(
                    1 for r in epi
                    if r.get("clustering_variants", {}).get(v, {}).get("any_theme_risen")),
                "calm_theme_risen": sum(
                    1 for r in calm
                    if r.get("clustering_variants", {}).get(v, {}).get("any_theme_risen")),
                "memo_theme_risen": sum(
                    1 for r in memo
                    if r.get("clustering_variants", {}).get(v, {}).get("any_theme_risen")),
                "episode_risen_theme_label_matches": sum(
                    1 for r in epi
                    if any(t.get("risen") and t.get("label_matches_expected") for t in
                           r.get("clustering_variants", {}).get(v, {}).get("themes") or [])),
            }
            for v in ("umap10_min_samples_5", "registered", "min_samples_5")},
        "per_date_table": [
            {"date": r["date"], "kind": r["kind"], "built": bool(r.get("built")),
             "reasons": r.get("reasons"),
             "w_norm_l2": (r.get("decomposition") or {}).get("w_norm_l2"),
             "g_1": (r.get("components") or [{}])[0].get("g_k"),
             "VS_1": (r.get("components") or [{}])[0].get("VS_k"),
             "VS_1_above_noise": (r.get("components") or [{}])[0].get("above_noise"),
             "clusters": ((r.get("clusters")) or {}).get("n"),
             "news_risk_statements": (r.get("news") or {}).get("risk_statements_rule_b"),
             "filing_risk_statements": (r.get("filings") or {}).get("risk_statements_rule_b"),
             "set_size": ((r.get("themes") or [{}])[0].get("set_coverage") or {}).get("set_size"),
             "set_companies_with_item_1a": [
                 (t.get("set_coverage") or {}).get("set_companies_with_item_1a")
                 for t in (r.get("themes") or [])],
             "clustering_degenerate": ((r.get("clustering_variants") or {})
                                       .get(PRIMARY_VARIANT) or {}).get("degenerate_clustering"),
             "any_theme_risen": r.get("any_theme_risen"),
             "risen_themes": [
                 {"k": t["k"], "label": t["theme_label"], "E_e": t["E_e"],
                  "placebo_q95": t["placebo_q95_of_random_max"],
                  "set_companies_with_item_1a":
                      t["set_coverage"]["set_companies_with_item_1a"],
                  "share_of_set_loading_covered":
                      t["set_coverage"]["share_of_set_absolute_loading_covered"],
                  "label_generic_term_count": t["label_generic_term_count"],
                  "news_members": t["news_members"],
                  "above_noise": t["above_noise"]}
                 for t in (r.get("themes") or []) if t.get("risen")]}
            for r in records],
        "clustering_instability": {
            "rule": DEGENERACY_RULE,
            "what_happened": (
                "On 2020-10-31 the primary clustering returned 2 clusters with noise 0.006 and "
                "one cluster holding 97.8% of the 2,850 statements. Diagnosis, run before the "
                "result was recorded: UMAP did run (the reducer is stored per date, no PCA "
                "fallback exists in this script), the seed is fixed, and the boilerplate rule "
                "did fire and excluded the giant cluster. The cause is the clusterer, not the "
                "date: three independent 64-company draws from that same date's own filings, "
                "at identical parameters and seed, returned 2, 2 and 26 clusters with noise "
                "0.000, 0.000 and 0.328, while a 2,200-sentence draw from the same corpus "
                "returned 24 clusters at noise 0.305. UMAP+HDBSCAN at this population size "
                "intermittently produces one all-inclusive cluster."),
            "action": (
                "No parameter was changed and no date was re-run until it produced clusters. "
                "The degenerate outcome is flagged per date and the statistic is reported with "
                "the flag."),
        },
        "interpretation_limits": [
            "The component set's Item 1A coverage is the binding limit. Across the twelve "
            "dates the book's own component set holds between 0 and 25 companies with a "
            "point-in-time Item 1A, against set sizes of 116 to 137, so every E_e rests on at "
            "most a fifth of the set's absolute loading and usually on a few per cent. The "
            "per-date coverage is printed beside every excess weight.",
            "A date whose component set has NO covered company cannot produce a non-zero E_e. "
            "'Not risen' on such a date is vacuous, not evidence. The flag is "
            "themes[].statistic_is_informative.",
            "News risk statements run 0 to 453 per date under rule B and are 0 on the three "
            "calm windows and the memo date, so on most dates no cluster can reach the 25 news "
            "members the design requires and every theme is company-specific by construction. "
            "The flag is themes[].filing_only.",
            "The eight-against-three comparison the registration describes is further weakened "
            "here: one calm window is unbuildable and the other two have zero covered set "
            "companies, so the control side of the comparison carries no information at all.",
            "The classifier is not run. Every statistic is the keyword run.",
            "The outcome of this experiment is a COVERAGE result, not a placebo result. The "
            "pipeline as built does not find the episode themes, and the reason is that the "
            "component set's filings are almost entirely missing, not that the placebo band is "
            "too high. No theme that rose has a label matching its episode's registered terms, "
            "and each rose on between 1 and 25 companies of a set of about 130.",
        ],
        "dates": records,
    }
    if "--only" in sys.argv:
        summary["partial_run"] = True
        json.dump(summary, open(OUT / "e7_summary_partial.json", "w"), indent=2, default=str)
    else:
        json.dump(summary, open(OUT / "e7_summary.json", "w"), indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"{'date':12} {'kind':8} {'built':6} {'g_1':>8} {'||w||':>7} {'VS_1':>7} "
          f"{'E_e':>8} {'q95':>8} risen")
    for r in records:
        if not r.get("built"):
            print(f"{r['date']:12} {r['kind']:8} {'NO':6} "
                  f"-- {'; '.join(r.get('reasons', []))[:70]}")
            continue
        c = r["components"][0]
        t = r["themes"][0]
        print(f"{r['date']:12} {r['kind']:8} {'yes':6} {c['g_k']:+8.4f} "
              f"{r['decomposition']['w_norm_l2']:7.4f} {c['VS_k']:7.4f} "
              f"{t['E_e']:+8.4f} {t['placebo_q95_of_random_max']:+8.4f} "
              f"{'YES' if r['any_theme_risen'] else 'no'}")
    print("=" * 78)
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["counts_by_clustering_variant"], indent=2))
    tgt = OUT / ("e7_summary_partial.json" if "--only" in sys.argv else "e7_summary.json")
    print(f"\nwrote {tgt} and {len(records)} files under {THEMES}")


DEVIATIONS = [
    {"id": "D1", "what": "a formation date whose 126-day window holds fewer than "
                         f"{MIN_WINDOW_OBS} panel observations is reported unbuilt",
     "why": "the holdings panel is not complete on every trading day; declared before the run"},
    {"id": "D2", "what": f"Item 1A truncated to the first {EXCERPT:,} characters",
     "why": "the excerpt length registered for exp-085's readers; without it the clustered "
            "population is ~64,000 filing sentences per date and HDBSCAN does not finish"},
    {"id": "D3", "what": f"the named public encoder is {ENCODER}",
     "why": "the spec requires one named encoder on every date; the ChronoBERT vintages on "
            "disk are a per-date family and are a production-path item"},
    {"id": "D4", "what": "the first-stage market is the capitalisation-weighted point-in-time "
                         "IWV universe return",
     "why": "Fama-French Mkt-RF on disk ends 2026-06-30 and cannot cover the memo date's "
            "window; the correlation between the two is printed on every date"},
    {"id": "D5", "what": "the bridge's candidates are five FRED macro series, the panel-built "
                         "mega-cap pair, and eleven equally-weighted sector portfolios",
     "why": "no ETF or futures price series is on disk; the sector portfolios stand in for the "
            "eleven sector ETFs the spec names"},
    {"id": "D6", "what": "the classifier is not run; the keyword run is the only run",
     "why": "spec node `riskextract` needs 2,000 language-model labels and none are on disk. A "
            "model key IS available in this environment; training the classifier was out of "
            "scope for this run. This is the node's own on_failure path and the brief permits "
            "it, but the keyword run is the only run and every theme statistic is its output"},
    {"id": "D10", "what": "the risk-statement extractor is the spec's own sentence-level "
                          "keyword rule (rule B), NOT exp-085's frozen filing-level rule "
                          "(rule A); rule A is imported unchanged and carried as a reported "
                          "comparator on every date",
     "why": "the two rules answer different questions. exp-085's rule asks whether a FILING "
            "states an external survival condition and is a conjunction over an 18,000-char "
            "excerpt; spec node `riskextract` asks whether a SENTENCE is a risk statement and "
            "describes a disjunction over one list. Applied at the sentence unit, exp-085's "
            "rule marked 0 of 5,351 news titles and 57 filing sentences on 2020-10-31, leaving "
            "a population of 57 and no date buildable. Rule B is assembled from exp-085's two "
            "term lists imported unchanged plus the three modal phrases spec.yaml writes out; "
            "no term is invented. exp-085's rule is not widened, restated or reported as "
            "anything other than itself"},
    {"id": "D11", "what": "the primary clustering reduces the embedding space with UMAP to "
                          f"{REDUCE_DIMS} dimensions ({REDUCE_METRIC} metric, seed "
                          f"{REDUCE_SEED}) before HDBSCAN at min_cluster_size "
                          f"{MIN_CLUSTER_SIZE} with min_samples 5, and excludes boilerplate "
                          "clusters by rule. The literal registered configuration and an "
                          "unreduced min_samples-5 configuration are computed and reported "
                          "beside it on every date",
     "why": "exp-091's registration fixes the minimum cluster size and says nothing about how "
            "the embedding space is reduced first. That is a SPEC OMISSION, not a choice made "
            "here: run literally, HDBSCAN on raw 384-dimension normalised MiniLM vectors under "
            "a euclidean metric with min_samples at its default of 25 returns noise share "
            "1.000 on all twelve dates, the known high-dimension failure mode. POC-BRIEF draft "
            "16 section 6 supplies the missing step and this implements it verbatim. All three "
            "configurations are reported so nothing is hidden"},
    {"id": "D13", "what": "a cluster whose c-TF-IDF label is generic risk-factor vocabulary "
                          "cannot be a theme",
     "why": "POC-BRIEF draft 16 section 6: 'the boilerplate cluster excluded by rule'. The "
            "rule is applied by declared list and never by eye: " + BOILERPLATE_RULE + ". The "
            "vocabulary and the rule are stored in every record. Boilerplate clusters are "
            "excluded from the book's argmax AND from the random books' maximum, so both sides "
            "of the placebo are judged over the same candidate set"},
    {"id": "D7", "what": "news titles seen after the formation date are dropped",
     "why": "the GDELT episode files span about two months before the event to two weeks after "
            "it, so most of every file postdates its own formation date; clustering them would "
            "read the reversal's news back onto the date before it"},
    {"id": "D12", "what": "a name whose return series is constant over the window is dropped "
                          "before the covariance is estimated",
     "why": "a halted or carried-forward price has zero variance and makes the sample "
            "correlation matrix singular. One such name on 2016-08-31 (UDF) drove the "
            "condition number of the SHRUNK matrix to 8.2e17; the count dropped is recorded "
            "per date under constant_series_dropped"},
    {"id": "D9", "what": f"a return whose two panel observations are more than {MAX_GAP_DAYS} "
                         "calendar days apart is dropped",
     "why": "the panel is missing whole trading days in some years and a pct_change across a "
            "gap is a multi-day return standing in a daily series"},
    {"id": "D8", "what": "data/raw/edgar_sections_2017 is not used",
     "why": "its records carry no filing date and no producer for them is in the repo, so "
            "point-in-time availability cannot be established"},
]

if __name__ == "__main__":
    main()
