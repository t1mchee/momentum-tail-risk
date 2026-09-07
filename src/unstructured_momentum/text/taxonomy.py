"""The driver taxonomy: eight drivers, seed terms expanded from the corpus, frozen per vintage.

Three properties are load-bearing and each was a ruling rather than a default.

VINTAGE. Snapshots are per-year and point-in-time: a snapshot for year Y is expanded using only
filings available before Y. That is the canonical artefact, and any claim whose CONTENT carries a
date must use it. The screen is granted a bounded exemption -- it may use the union vocabulary
for candidate selection, because a term that only exists in 2022 finds nothing in a 2016 filing
and because what the screen misses is measured by the false-negative design rather than assumed.
The exemption is stated here so it cannot be inherited silently.

IDENTITY. Every snapshot is content-hashed. Extraction records carry the taxonomy version and
hash the way they carry a prompt hash and a model id, and a record referencing a hash the
repository does not contain fails the build.

HONESTY. The eight-driver LIST is a design-level choice made with hindsight over the whole
period. It is frozen and stated as such. That is the same status as the decision to study
momentum at all, and pretending the list was derived point-in-time would be the more misleading
option.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

STORE = Path("data/taxonomy")

#: The eight drivers and their seeds. Three to five words each, chosen for what they NAME rather
#: than for what they retrieve, so the expansion does the work and the seed does not smuggle in a
#: hand-tuned vocabulary.
#: taxonomy-v2 reseeds rates_duration. v1's seeds were "interest", "rate", "yield", "duration",
#: and the first two are polysemous in filing prose -- "interest in the property", "rate of
#: growth" -- so the expansion scored on documents that had nothing to do with rates and returned
#: identical, false, removed and instituted. Polysemy is a UNIGRAM disease, so the fix is seeds
#: that are unambiguous standing alone, plus phrase seeds for concepts that only exist as a
#: bigram. Reseeded before any extraction record existed, so nothing was invalidated.
SEEDS: dict[str, tuple[str, ...]] = {
    "rates_duration":      ("libor", "sofr", "duration", "maturities", "swaps", "refinancing",
                            "borrowings", "amortization",
                            "interest rate", "basis points", "yield curve", "floating rate"),
    # v4 reseed, and the third driver to need one for the same reason. "cost" appears in 56
    # percent of filings and "pricing" names something every business discusses, so the
    # expansion returned cost savings, intense competition, lower cost and cost structure --
    # generic business prose rather than an inflation vocabulary. Three independent signals
    # agreed: 125 terms against 55 to 93 for the others, Jaccard stuck at 0.13 through v1, v2
    # and v3 alike as the only driver that never improved, and the only driver that would not
    # respond to thresholding -- still selecting 0.254 of the corpus where every other driver
    # had collapsed to 0.003 to 0.093.
    # Phrase seeds only, which is the cure that worked for rates_duration.
    "inflation_pricing":   ("pricing pressure", "cost inflation", "input costs",
                            "raw material costs", "price increases", "wage inflation",
                            "pass-through", "inflationary pressures"),
    # v3 reseed. "exchange" matches "Securities Exchange Act", which sits in the
    # disclosure-controls paragraph of every filing, so six of this driver's top fourteen terms
    # were SOX boilerplate: recorded processed, processed summarized, timely decisions.
    "fx_dollar":           ("foreign currency", "exchange rate", "currency translation",
                            "functional currency", "foreign exchange", "currency fluctuations",
                            "devaluation", "repatriation"),
    "energy_commodities":  ("oil", "commodity", "fuel", "energy"),
    # v3 reseed. "trade" matches "trade secrets", so four of this driver's top eight terms were
    # intellectual-property law rather than trade exposure.
    "china_trade":         ("china", "chinese", "tariff", "tariffs", "import duties",
                            "trade barriers", "trade restrictions", "customs duties"),
    "credit_refinancing":  ("credit", "refinance", "covenant", "indebtedness"),
    "supply_chain":        ("supply", "supplier", "shortage", "logistics"),
    "ai_compute":          ("artificial", "semiconductor", "compute", "datacenter"),
}

_WORD = re.compile(r"[a-z][a-z'-]{2,}")
#: Words that co-occur with everything and name nothing. Removed before scoring rather than
#: after, so they cannot crowd out a real term on count alone.
_STOP = frozenset("""the and for that with this from will been have has are was were our their
its they which such other than then them these those there here more most may might can could
would should shall must also into over under about between during within without upon each any
all not but our we us it he she his her you your who whom whose""".split())


@dataclass
class Snapshot:
    """One vintage of the taxonomy. Immutable once written."""

    version: str
    vintage: str
    drivers: dict[str, list[str]]
    n_documents: int
    seed_terms: dict[str, list[str]] = field(default_factory=dict)
    method: str = "king-lam-roberts expansion, expanding window"

    def content_hash(self) -> str:
        body = json.dumps({"version": self.version, "vintage": self.vintage,
                           "drivers": {k: sorted(v) for k, v in sorted(self.drivers.items())}},
                          sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        # Terms are stored in LIFT ORDER, not alphabetically. The order is the whole quality
        # signal: sorting it away made a good expansion look like noise on inspection, because
        # the first terms alphabetically are the tail of the ranking. Identity is still
        # order-independent -- content_hash sorts before hashing -- so a reordering cannot
        # invalidate a record while a content change still does.
        return {"version": self.version, "vintage": self.vintage, "method": self.method,
                "n_documents": self.n_documents, "seed_terms": self.seed_terms,
                "drivers": {k: list(v) for k, v in sorted(self.drivers.items())},
                "content_hash": self.content_hash()}

    def path(self) -> Path:
        return STORE / f"{self.version}__{self.vintage}.json"

    def write(self) -> Path:
        STORE.mkdir(parents=True, exist_ok=True)
        p = self.path()
        if p.exists():
            prior = json.loads(p.read_text())
            if prior.get("content_hash") != self.content_hash():
                raise ValueError(
                    f"{p} exists with a different content hash. A vintage is immutable; write a "
                    "new version rather than changing one that records may already reference.")
            return p
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p


def load(version: str, vintage: str) -> Snapshot:
    p = STORE / f"{version}__{vintage}.json"
    d = json.loads(p.read_text())
    s = Snapshot(version=d["version"], vintage=d["vintage"], drivers=d["drivers"],
                 n_documents=d["n_documents"], seed_terms=d.get("seed_terms", {}),
                 method=d.get("method", ""))
    if s.content_hash() != d["content_hash"]:
        raise ValueError(f"{p} content hash does not match its contents")
    return s


def known_hashes() -> set[str]:
    """Every taxonomy hash the repository contains. The validator checks records against this."""
    out = set()
    for p in sorted(STORE.glob("*.json")):
        try:
            out.add(json.loads(p.read_text())["content_hash"])
        except Exception:  # noqa: BLE001
            continue
    return out


# ------------------------------------------------------------------ expansion

def _tokenise(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 3}


def _terms(text: str, *, bigrams: bool = False) -> set[str]:
    """Unigrams, optionally with bigrams.

    Bigram space exists for one reason: some concepts have no unambiguous unigram. "interest
    rate" means something "interest" alone does not, and a driver whose seeds are phrases has to
    be scored where those phrases are visible.
    """
    toks = [w for w in _WORD.findall(text.lower()) if len(w) > 2]
    uni = {w for w in toks if w not in _STOP and len(w) > 3}
    if not bigrams:
        return uni
    bi = {f"{a} {b}" for a, b in zip(toks, toks[1:])
          if a not in _STOP and b not in _STOP}
    return uni | bi


def needs_bigrams(seeds: dict[str, tuple[str, ...]] | None = None) -> bool:
    """Whether any driver's seeds contain a phrase, in which case scoring runs in bigram space."""
    return any(" " in t for v in (seeds or SEEDS).values() for t in v)


def expand(docs: list[str], *, per_driver: int = 40, min_df_frac: float = 0.01,
           min_df_floor: int = 10, max_df_pctile: float = 95.0,
           seeds: dict[str, tuple[str, ...]] | None = None) -> dict:
    """Grow each driver's seed list into a term list, from the documents themselves.

    The mechanism is the one the political-risk literature validated: score a candidate by how
    much more often it appears in documents containing a seed than in documents that do not.
    Terms above a document-frequency ceiling are dropped first, because a word in two fifths of
    all filings describes the genre and would top every driver's list at once. The floor is a
    FRACTION of the sample rather than a fixed count, measured rather than guessed: on 1,500
    documents a floor of 5 admitted one-off terms that appear in a single filing, a floor near
    1.7 percent gave semiconductor, wafer, memory, lithography and foundry for the compute
    driver, and a floor near 5 percent degraded into generic words like solutions and leading.

    Returns terms per driver ORDERED BY LIFT, seeds first.
    """
    seeds = seeds or SEEDS
    bags = [_tokenise(d) for d in docs]
    n = len(bags)
    if n < 20:
        raise ValueError(f"expansion needs at least 20 documents, got {n}")
    min_df = max(min_df_floor, int(round(min_df_frac * n)))
    df: dict[str, int] = {}
    for b in bags:
        for w in b:
            df[w] = df.get(w, 0) + 1
    # The ceiling is a PERCENTILE of this vintage's own document-frequency distribution, not a
    # fixed fraction. Signature-block boilerplate -- undersigned at 0.191, thereunto at 0.185 --
    # earns high lift for some drivers through a document-LENGTH confound rather than a semantic
    # one: an over-captured section holds both a seed word and the signature block because it is
    # long. A hand-picked cutoff between those and libor at 0.137, which is one of the best rates
    # terms, would be fitted to two examples. The 95th percentile is a rule, it computes inside
    # the vintage so the point-in-time discipline is inherited rather than re-argued, and it
    # drops the boilerplate while keeping libor.
    #
    # It treats a symptom. The underlying confound is document length and the ceiling does not
    # measure it; what the ceiling costs in recall is priced by the false-negative design.
    # Measured among CANDIDATES, for the reason recorded in scripts/build_taxonomy_v2.py: a
    # percentile over the whole vocabulary sits below the floor once bigrams add a rare tail.
    counts = np.array([c for c in df.values() if c >= min_df], dtype=float)
    if counts.size < 20:
        raise ValueError(f"only {counts.size} terms clear a document-frequency floor of {min_df}")
    max_df = float(np.percentile(counts, max_df_pctile))
    if max_df <= min_df:
        raise ValueError(f"ceiling {max_df:.0f} at or below floor {min_df}; filter would be empty")
    vocab = {w for w, c in df.items() if min_df <= c <= max_df}

    out: dict[str, list[str]] = {}
    for drv, seed in seeds.items():
        hit = [i for i, b in enumerate(bags) if any(s in b for s in seed)]
        if len(hit) < min_df:
            out[drv] = list(seed)
            continue
        hs = set(hit)
        inside, outside = len(hit), n - len(hit)
        scores = []
        for w in vocab:
            a = sum(1 for i in hit if w in bags[i])
            if a < min_df:
                continue
            b_ = df[w] - a
            # Rate inside against rate outside, smoothed, so a rare term cannot win on one hit.
            lift = ((a + 1) / (inside + 2)) / ((b_ + 1) / (outside + 2))
            if lift > 1.0:
                scores.append((lift, w))
        scores.sort(reverse=True)
        terms = list(dict.fromkeys(list(seed) + [w for _, w in scores[:per_driver]]))
        out[drv] = terms
    return out


def build_snapshot(sections: pd.DataFrame, vintage_year: int, *, version: str = "taxonomy-v1",
                   text_cols: tuple[str, ...] = ("item_1a", "item_7a"),
                   sample: int = 4000, seed: int = 20260826) -> Snapshot:
    """One point-in-time snapshot: expanded ONLY on filings available before the vintage year.

    Point-in-time is the ruling and it is enforced here rather than trusted to a caller: the
    frame is cut by filing date before a single token is counted.
    """
    d = sections[pd.to_datetime(sections["filing_date"]).dt.year < vintage_year]
    docs: list[str] = []
    for c in text_cols:
        if c in d:
            s = d[c].fillna("")
            docs += s[s.str.len() > 2000].tolist()
    if len(docs) > sample:
        rng = np.random.default_rng(seed + vintage_year)
        docs = [docs[i] for i in rng.choice(len(docs), sample, replace=False)]
    drivers = expand(docs)
    return Snapshot(version=version, vintage=str(vintage_year), drivers=drivers,
                    n_documents=len(docs),
                    seed_terms={k: list(v) for k, v in SEEDS.items()})


def union_vocabulary(version: str = "taxonomy-v1") -> dict[str, list[str]]:
    """Every vintage's terms merged, for SCREENING ONLY.

    The bounded exemption: a screen may use this because a term that only exists in 2022 finds
    nothing in a 2016 filing, and because what the screen misses is measured by the
    false-negative design rather than assumed. Any claim whose content carries a date uses the
    vintage snapshot instead, and this function's name is meant to make that hard to forget.
    """
    merged: dict[str, set[str]] = {}
    for p in sorted(STORE.glob(f"{version}__*.json")):
        d = json.loads(p.read_text())
        for k, v in d["drivers"].items():
            merged.setdefault(k, set()).update(v)
    return {k: sorted(v) for k, v in merged.items()}


# ------------------------------------------------------------------ seed audit

def seed_coherence(docs: list[str], seeds: dict[str, tuple[str, ...]] | None = None,
                   *, bigrams: bool = True) -> pd.DataFrame:
    """Per SEED, how often it appears alongside another seed of its own driver.

    This targets the disease directly. A polysemous seed selects documents about something else:
    "exchange" fires on "Securities Exchange Act" in filings with no other currency language,
    and "trade" fires on "trade secrets" in filings with no other China language. A clean seed
    keeps company -- "libor" turns up beside maturities and swaps.

    An earlier attempt measured whether a driver's EXPANSION shared stems with its seeds. That
    was dropped rather than shipped: high orphaning means drift for one driver and a genuinely
    new vocabulary for another -- semiconductor legitimately recruits wafer and manufacturing
    yields -- so the number could not tell success from failure, and it ranked the healthiest
    driver in the set as the worst.
    """
    seeds = seeds or SEEDS
    bags = [_terms(d, bigrams=bigrams) for d in docs]
    rows = []
    for drv, ss in seeds.items():
        for s_ in ss:
            hits = [b for b in bags if s_ in b]
            if not hits:
                rows.append({"driver": drv, "seed": s_, "n_docs": 0,
                             "coherence": float("nan")}); continue
            others = set(ss) - {s_}
            with_other = sum(1 for b in hits if b & others)
            rows.append({"driver": drv, "seed": s_, "n_docs": len(hits),
                         "coherence": round(with_other / len(hits), 3)})
    out = pd.DataFrame(rows)
    out["doc_frac"] = (out["n_docs"] / max(len(bags), 1)).round(3)
    # TWO signals, because neither catches everything on its own and together they caught all
    # four seeds that were already known to be bad.
    #
    # SELECTIVITY. A seed matching most of the corpus cannot discriminate: "exchange" fired on
    # 78 percent of filings through "Securities Exchange Act", "interest" and "rate" on 72
    # percent each. Absolute and comparable across drivers.
    #
    # COHERENCE RANK. Whether a seed keeps company with its own driver's other seeds. "trade"
    # scored 0.593 against 0.85 to 0.99 for its siblings and was the one recruiting trade-secrets
    # language. Only comparable WITHIN a driver, because coherence rises with how many seeds a
    # driver has -- an eight-seed driver gives every seed more chances to find company than a
    # four-seed one -- so it is used as a rank and never as a threshold.
    med = out.groupby("driver")["coherence"].transform("median")
    out["too_common"] = out["doc_frac"] > 0.50
    out["incoherent_for_its_driver"] = out["coherence"] < med - 0.10
    out["suspect"] = out["too_common"] | out["incoherent_for_its_driver"]
    return out.sort_values(["driver", "coherence"]).reset_index(drop=True)


def polysemy_audit(snap: Snapshot) -> pd.DataFrame:
    """Cross-driver leak per driver: what share of an expansion another driver also carries.

    Kept because it is sound and it moved in the right direction when the seeds were fixed --
    v1 leaked 0.05 to 0.23 across drivers, v2 leaks 0.00 to 0.05. Seed orphaning was removed;
    see `seed_coherence` for why, and for the signal that replaced it.
    """
    seeds = {d: set(snap.seed_terms.get(d, [])) for d in snap.drivers}
    exp = {d: [t for t in ts if t not in seeds[d]] for d, ts in snap.drivers.items()}
    rows = []
    for d, terms in exp.items():
        others = set().union(*[set(v) for k, v in exp.items() if k != d]) if len(exp) > 1 else set()
        leak = sum(1 for t in terms if t in others) / max(len(terms), 1)
        rows.append({"driver": d, "n_expansion": len(terms),
                     "cross_driver_leak": round(leak, 3)})
    return pd.DataFrame(rows).sort_values("cross_driver_leak", ascending=False).reset_index(
        drop=True)
