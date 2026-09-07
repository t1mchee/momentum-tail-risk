"""Planted-effect certificates: what could this instrument have seen?

A null is a claim about the world only if the design could have detected the effect on trial.
This project recorded five that could not. The clearest: mean pairwise cosine is algebraically
the squared centroid norm, so a sub-bet of fraction f enters as f SQUARED, and a 40 percent
partial bet needed internal cohesion of 0.138 to move the statistic two standard deviations --
while a whole GICS sector inside that leg achieves 0.00 to 0.13. The positive control that
reassured us passed at z between 6.8 and 14.6 against SINGLE-SECTOR groups, a
hundred-percent-loading single-block alternative that was never the hypothesis. A control
validated against the wrong shape is worse than no control, because it converts an underpowered
design into a confident null.

Two design rules follow, and they are the reason this module exists rather than a one-off check.

**Plants are layered, and a certificate names its layer.** A vector-layer plant certifies a
STATISTIC: inject a shared direction into rows of an embedding matrix and ask whether the
statistic moves. A text-layer plant certifies an INSTRUMENT end to end: inject driver language
into documents and run the whole pipeline -- tokenise, embed, centre, score. An experiment whose
claim starts at text needs the second; the first is necessary and not sufficient, because every
step between the document and the number can destroy the signal, and this project has already
lost one to truncation and one to a wrong section.

**A certificate is a region, not a number.** Planting the single shape you happen to hypothesise
reproduces the original error one level up: if the real structure is smaller or more diffuse than
the plant, the check validates against the wrong shape too, and you are exactly as confident and
exactly as wrong. So detection is measured across a grid of (fraction, cohesion) and reported as
the boundary at which power reaches its target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

#: An instrument is any map from a matrix to a scalar. Larger means "more structure".
Instrument = Callable[[np.ndarray], float]


def universe_hash(pool, *, label: str = "") -> str:
    """A stable fingerprint of the population a certificate was issued against.

    Certificates describe an instrument operating on a POPULATION, not in the abstract. The MDE
    region and the false-alarm rate both come from the null's spread, and the null is drawn from
    the universe -- so a certificate issued against 1,323 tickers says nothing about the same
    instrument on 2,622. Without a fingerprint that mismatch is invisible: the certificate still
    reads as valid, still names a plausible effect size, and still gates a verdict.

    Hashes shape and content rather than identity, so re-running on the same data reproduces the
    hash and re-running on a doubled corpus does not.
    """
    import hashlib
    a = np.asarray(pool)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    try:
        h.update(np.ascontiguousarray(np.round(a.astype(np.float64), 6)).tobytes())
    except (TypeError, ValueError):
        # A universe is not always a matrix of numbers. The returns-layer certificate's
        # population is a list of tickers alongside a panel, and a hash that raises on the
        # first non-numeric universe is a gate that fails open the moment it is reused.
        h.update("|".join(map(str, a.ravel().tolist())).encode())
    h.update(label.encode())
    return h.hexdigest()[:16]


def null_hash(*parts, label: str = "") -> str:
    """A fingerprint of the NULL CONSTRUCTION a certificate was issued under.

    The companion to `universe_hash`, and it exists because the universe hash alone was not
    enough. A certificate already refused to gate a verdict whose POPULATION had changed. It said
    nothing about the null, so when a characteristic-cell construction was repaired -- one that
    could span a sector boundary, replaced by one that cannot -- the certificate was correctly
    re-issued and every sentence quoting it silently kept the broken null's number. Both figures
    were correct measurements of something, which is why nothing downstream could tell them
    apart.

    Hashes the defining CONTENT of the null rather than a description of it: the cell assignment
    itself, the permutation count, the level, the sampler's name. Change any of them and the
    hash moves, which is the whole point.
    """
    import hashlib

    h = hashlib.sha256()
    for part in parts:
        if part is None:
            h.update(b"None")
        elif isinstance(part, np.ndarray):
            a = np.asarray(part)
            h.update(str(a.shape).encode())
            try:
                h.update(np.ascontiguousarray(np.round(a.astype(np.float64), 9)).tobytes())
            except (TypeError, ValueError):
                h.update("|".join(map(str, a.ravel().tolist())).encode())
        else:
            h.update(str(part).encode())
        h.update(b"\x00")
    h.update(label.encode())
    return h.hexdigest()[:16]


@dataclass
class Certificate:
    """What an instrument can see, on WHICH population, in the form a registration must carry."""

    instrument: str
    plant_layer: str                     # "vector" or "text"
    effect_shape: str
    null_construction: str
    false_alarm_rate: float
    #: How many replicates the rate was estimated from. Without it the rate is a number with an
    #: unknown error bar, which is how a 0.13 reading and a 0.05 reading were both believed.
    false_alarm_n: int = 0
    mde: dict = field(default_factory=dict)
    curve: pd.DataFrame | None = None
    #: Fingerprint of the pool this was issued against. A verdict gated by a certificate whose
    #: hash does not match the data it ran on is refused by the validator.
    universe_hash: str = ""
    universe_n: int = 0
    #: Fingerprint of the NULL CONSTRUCTION. A certificate is quotable only when BOTH its
    #: population and its null are the ones in force; the universe hash alone let a repaired
    #: null go unnoticed for a day.
    null_hash: str = ""

    def statement(self) -> str:
        """The sentence that goes into result.power_statement."""
        stamp = (f" Universe {self.universe_n} rows, hash {self.universe_hash}."
                 if self.universe_hash else "")
        if self.null_hash:
            stamp += f" Null hash {self.null_hash}."
        if not self.mde:
            fa = f"{self.false_alarm_rate:.3f}"
            if self.false_alarm_n:
                se = (self.false_alarm_rate * (1 - self.false_alarm_rate)
                      / self.false_alarm_n) ** 0.5
                fa += f" (+/-{se:.3f}, {self.false_alarm_n} replicates)"
            return (f"{self.instrument} ({self.plant_layer} layer) detected no planted effect "
                    f"anywhere on the tested grid at the target power. False-alarm rate "
                    f"{fa}. Null: {self.null_construction}.{stamp}")
        b = ", ".join(f"f>={f:.2f} at cosine>={c:.3f}" for f, c in sorted(self.mde["boundary"]))
        fa = f"{self.false_alarm_rate:.3f}"
        if self.false_alarm_n:
            se = (self.false_alarm_rate * (1 - self.false_alarm_rate) / self.false_alarm_n) ** 0.5
            fa += f" (+/-{se:.3f}, {self.false_alarm_n} replicates)"
        return (f"{self.instrument} ({self.plant_layer} layer) resolves {self.effect_shape} at "
                f"{b}, at {self.mde['power']:.0%} power and alpha {self.mde['alpha']}. "
                f"False-alarm rate {fa}. "
                f"Null: {self.null_construction}.{stamp}")


# ------------------------------------------------------------------ vector-layer plants

def plant_partial_bet(X: np.ndarray, f: float, target_cos: float,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Inject one shared direction into a fraction f of rows, to a TARGET PAIRWISE COSINE.

    This is the shape that actually failed: a MINORITY of a book sharing a single latent
    exposure, the rest unrelated.

    The parameter is the resulting mean pairwise cosine AMONG PLANTED ROWS, not an opaque
    mixing weight, so a certificate speaks in the units the comparison uses -- "a whole GICS
    sector inside this leg achieves 0.00 to 0.13" is then directly readable against it.

    A first version mixed a unit direction into unnormalised Gaussian rows. In 768 dimensions
    those rows have norm about 27.7, so a weight of 0.3 on a unit vector was swamped by scale
    and the plant was roughly a hundred times weaker than intended. Rows are normalised before
    mixing here, and the mixing weight is solved from the target rather than passed in.
    """
    Y = X.astype(np.float64).copy()
    n = len(Y)
    k = int(round(f * n))
    if k < 2:
        return Y, np.array([], dtype=int)
    idx = rng.choice(n, k, replace=False)

    # Unit-scale the rows being planted so the mix is between comparable magnitudes.
    R = Y[idx]
    R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    d = rng.normal(size=Y.shape[1])
    d /= np.linalg.norm(d)

    # The mixing weight is solved NUMERICALLY, in the centred space the statistics use, not
    # analytically in raw space. Real embedding rows are strongly anisotropic -- any two filings
    # sit near 0.99 raw cosine -- so a weight derived from the raw-space identity
    # w^2/(w^2+(1-w)^2) over-delivered by an order of magnitude: a requested 0.05 produced 0.655
    # once centring removed the common direction and left the planted one as all that survived.
    # Centring depends on the whole basket, so there is no closed form; bisection is honest and
    # costs microseconds.
    t = float(np.clip(target_cos, 0.0, 0.999))
    scale = np.linalg.norm(Y[idx], axis=1, keepdims=True)

    def _mix(w: float) -> np.ndarray:
        M = w * d + (1.0 - w) * R
        M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12) * scale
        Z = Y.copy()
        Z[idx] = M
        return Z

    def _achieved(w: float) -> float:
        C = _normalise(_mix(w))[idx]
        G = C @ C.T
        iu = np.triu_indices(len(C), 1)
        return float(G[iu].mean())

    if t <= 0.0:
        return Y, idx
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _achieved(mid) < t:
            lo = mid
        else:
            hi = mid
    return _mix(0.5 * (lo + hi)), idx


def _normalise(X: np.ndarray) -> np.ndarray:
    Z = X - X.mean(0, keepdims=True)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


# ------------------------------------------------------------------ nulls and detection

#: Draws one basket's row-indices from a pool. The default is an unmatched random draw; an
#: experiment that matches its null on sector or size passes its own sampler, so the
#: certificate is issued against the SAME null the verdict will use. This matters: the failed
#: gate's sector-matched null had four times the spread of an unmatched one, and the effect
#: size a design can resolve moves with it.
BasketSampler = Callable[[np.random.Generator, int], np.ndarray]


def _random_basket(n_pool: int) -> BasketSampler:
    def draw(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(n_pool, min(size, n_pool), replace=False)
    return draw


def null_distribution(instrument: Instrument, pool: np.ndarray, size: int, *,
                      n_rep: int = 200, seed: int = 0,
                      sampler: BasketSampler | None = None) -> np.ndarray:
    """The instrument's distribution over random baskets of `size` drawn from `pool`.

    The null is BASKET SELECTION, not row scrambling, and the distinction is not cosmetic. A
    first version scrambled columns within rows; for a centred, unit-normalised matrix mean
    pairwise cosine is algebraically about -1/(k-1) whatever the data, so that null had zero
    variance and every z-score it produced was meaningless. The spread that matters comes from
    drawing DIFFERENT REAL FIRMS -- which is exactly what the failed gate's sector-matched
    baskets were doing, and where its 0.0109 came from.
    """
    rng = np.random.default_rng(seed)
    draw = sampler or _random_basket(len(pool))
    Pc = _normalise(pool)                     # centre ONCE on the pool, then index baskets out
    out = np.empty(n_rep)
    for i in range(n_rep):
        out[i] = instrument(Pc[draw(rng, size)])
    return out


#: Replicates for a false-alarm estimate. See the note in certify_comovement: at 200 the
#: standard error on a rate near 0.05 is 0.015, which is the difference between nominal and
#: three times nominal. Certificates now carry the replicate count so the reader can compute it.
FALSE_ALARM_REPS = 400


def false_alarm_rate(instrument: Instrument, pool: np.ndarray, size: int,
                     *, n_rep: int = FALSE_ALARM_REPS,
                     alpha: float = 0.05, seed: int = 0,
                     sampler: BasketSampler | None = None) -> float:
    """Planted-ABSENCE check: how often the instrument fires on structureless data.

    The mirror of a planted effect and equally necessary. A SUPPORTED verdict from an instrument
    that fires on noise is not a finding, and nothing in this project has previously measured it.
    """
    null = null_distribution(instrument, pool, size, n_rep=n_rep, seed=seed, sampler=sampler)
    thr = float(np.quantile(null, 1.0 - alpha))
    check = null_distribution(instrument, pool, size, n_rep=n_rep, seed=seed + 9973,
                              sampler=sampler)
    return float((check > thr).mean())


def detection_curve(instrument: Instrument, pool: np.ndarray, size: int, *,
                    fractions=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8),
                    cohesions=(0.02, 0.05, 0.1, 0.15, 0.25, 0.4),
                    n_rep: int = 50, alpha: float = 0.05, seed: int = 0,
                    sampler: BasketSampler | None = None) -> pd.DataFrame:
    """Detection rate over the (fraction, target-cosine) grid, one row per cell.

    Each replicate draws a fresh basket from the pool and plants into it, so the measured
    detection rate carries basket-selection variance rather than assuming one basket away.
    """
    draw = sampler or _random_basket(len(pool))
    null = null_distribution(instrument, pool, size, n_rep=max(200, n_rep * 4), seed=seed,
                             sampler=sampler)
    thr = float(np.quantile(null, 1.0 - alpha))
    rows = []
    for f in fractions:
        for c in cohesions:
            rng = np.random.default_rng(seed + int(f * 1000) + int(c * 100000))
            hits = 0
            for _ in range(n_rep):
                # Plant into the pool at the basket's rows, then centre the WHOLE pool and
                # index the basket out -- the order the failed gate used. Centring on the
                # basket instead is a different operation with a different answer: it forces
                # the basket to sum to zero, so a shared direction in a minority of rows pushes
                # the majority the other way and the cross-pairs cancel the effect. Planting
                # then made the statistic go DOWN. Order of operations is load-bearing here.
                sel = draw(rng, size)
                Ppl = pool.astype(np.float64).copy()
                sub, local = plant_partial_bet(pool[sel], f, c, rng)
                if len(local) < 2:
                    continue
                Ppl[sel] = sub
                hits += int(instrument(_normalise(Ppl)[sel]) > thr)
            rows.append({"fraction": f, "target_cos": c, "n_rep": n_rep,
                         "threshold": thr, "detection_rate": hits / n_rep})
    return pd.DataFrame(rows)


def mde_region(curve: pd.DataFrame, *, power: float = 0.80, alpha: float = 0.05) -> dict:
    """The (fraction, cohesion) boundary where detection first reaches the target power.

    Reported as a boundary rather than a point, because a design's power depends on BOTH how
    much of the book shares the bet and how tightly. A single number hides the trade.
    """
    boundary = []
    for f, grp in curve.groupby("fraction"):
        ok = grp[grp.detection_rate >= power].sort_values("target_cos")
        if len(ok):
            boundary.append((float(f), float(ok.iloc[0].target_cos)))
    return {"boundary": boundary, "power": power, "alpha": alpha,
            "detectable_anywhere": bool(boundary)}


def certify(instrument: Instrument, pool: np.ndarray, size: int, *, name: str,
            effect_shape: str = "partial cross-sector bet",
            plant_layer: str = "vector", power: float = 0.80, alpha: float = 0.05,
            n_rep: int = 50, seed: int = 0, sampler: BasketSampler | None = None,
            null_name: str = "unmatched random baskets", **grid) -> Certificate:
    """One call from instrument to the certificate a registration must carry."""
    curve = detection_curve(instrument, pool, size, n_rep=n_rep, alpha=alpha, seed=seed,
                            sampler=sampler, **grid)
    return Certificate(
        instrument=name, plant_layer=plant_layer, effect_shape=effect_shape,
        null_construction=(f"{null_name} of {size} from a pool of {len(pool)}, "
                           f"{max(200, n_rep*4)} draws, alpha {alpha}"),
        false_alarm_rate=false_alarm_rate(instrument, pool, size, alpha=alpha, seed=seed,
                                          sampler=sampler),
        false_alarm_n=FALSE_ALARM_REPS,
        mde=mde_region(curve, power=power, alpha=alpha), curve=curve,
        universe_hash=universe_hash(pool, label=name), universe_n=len(pool),
        null_hash=null_hash(null_name, size, len(pool), alpha, FALSE_ALARM_REPS,
                            getattr(sampler, "__name__", "random_basket"), label=name))


# ------------------------------------------------------------------ text-layer plants

#: Sentence frames that read like real risk-factor prose. The plant has to survive the whole
#: pipeline -- tokenisation, truncation, pooling, centring -- so it must sit INSIDE the document
#: in the register the section is written in, not be appended as a tag the encoder would treat
#: as an outlier.
_FRAMES = (
    "Our results of operations are sensitive to changes in {t}, and a material movement could "
    "adversely affect our financial condition.",
    "We have significant exposure to {t}, which we do not currently hedge.",
    "Adverse developments in {t} could reduce demand for our products and impair our operating "
    "margins.",
    "A substantial portion of our obligations is affected by {t}, and we may be unable to pass "
    "increased costs through to customers.",
    "Management monitors {t} closely because sustained changes would require us to revise our "
    "capital allocation plans.",
)


def plant_text_effect(texts: list[str], driver_terms: list[str], f: float, intensity: int,
                      rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """Inject driver language into a fraction f of documents, `intensity` sentences each.

    This certifies an INSTRUMENT rather than a statistic. A vector-layer plant asks whether a
    number moves when the geometry moves; it cannot see that a quarter of the corpus was the
    wrong section, or that truncation kept 7 percent of a document and dropped the part where
    the exposure was stated. Both of those decided a null in this project before any statistic
    ran, so any claim that starts at text needs a certificate that starts at text.

    Sentences are inserted at random positions in the document body rather than appended,
    because a block at the end is the first thing truncation removes and the plant would then
    measure the truncation rather than the instrument.
    """
    out = list(texts)
    n = len(out)
    k = int(round(f * n))
    if k < 2 or intensity < 1:
        return out, np.array([], dtype=int)
    idx = rng.choice(n, k, replace=False)
    for i in idx:
        body = out[i]
        parts = body.split(". ")
        for _ in range(intensity):
            term = str(rng.choice(driver_terms))
            frame = str(rng.choice(_FRAMES)).format(t=term)
            at = int(rng.integers(0, max(1, len(parts))))
            parts.insert(at, frame.rstrip("."))
        out[i] = ". ".join(parts)
    return out, idx


#: Invented terms with no pretraining representation, used to measure what chronological
#: consistency costs. A 2019-vintage encoder has never seen 2020's vocabulary and must detect a
#: novel driver through context words alone; this quantifies that handicap instead of leaving it
#: as an argument.
NOVEL_TERMS = ("zephyrite supply constraints", "the Calder index", "grivance-linked demand",
               "Tessaly Protocol compliance", "murex freight surcharges")

#: The null for a text-layer certificate: the SAME insertion procedure with vocabulary that is
#: ordinary filing language rather than a driver. This is what isolates the driver from the act
#: of inserting text at all -- adding sentences changes length, token mix and pooling regardless
#: of what they say, and a null that does not control for that measures the insertion.
NEUTRAL_TERMS = ("our headquarters lease", "employee training programs", "our trademark portfolio",
                 "routine facility maintenance", "our records retention policy")


def plant_novel_tokens(texts: list[str], f: float, intensity: int,
                       rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """A text plant using terms the encoder cannot have learned.

    The gap between this and `plant_text_effect` with real driver terms IS the vintage cost. If
    it is large, a leak-free encoder is structurally handicapped on genuinely new drivers, and
    an emergence result needs a contemporaneous-model arm reported as an upper bound. If it is
    small, context carries the signal and the chrono constraint is close to free.
    """
    return plant_text_effect(texts, list(NOVEL_TERMS), f, intensity, rng)


def text_detection_curve(embed: Callable[[list[str]], np.ndarray], instrument: Instrument,
                         texts: list[str], driver_terms: list[str], *,
                         fractions=(0.2, 0.4), intensities=(1, 3, 6),
                         n_rep: int = 10, alpha: float = 0.05, seed: int = 0,
                         novel: bool = False, basket: int | None = None) -> pd.DataFrame:
    """Detection over the (fraction, intensity) grid, running the FULL pipeline each replicate.

    `embed` is the caller's text-to-vector map, so the certificate covers whatever tokenisation,
    truncation and pooling that pipeline actually uses. Expensive by construction: every planted
    document is re-embedded. Only the planted documents are, which is the one economy taken.

    `basket` evaluates the instrument on a strict subset of the centred population. This is
    required for any statistic that is pinned by centring -- mean pairwise cosine is exactly
    -1/(k-1) when the centred set IS the measured set, so without a subset it reads a constant
    (trp-47). Statistics with genuine range under centring, such as the leading eigenvalue, do
    not need it, and leaving `basket` unset measures the whole population.
    """
    if basket is not None and basket >= len(texts):
        raise ValueError(f"basket {basket} must be strictly smaller than the {len(texts)}-doc "
                         "population, or centring pins any pairwise statistic")
    plant = plant_novel_tokens if novel else (
        lambda t, f, i, r: plant_text_effect(t, driver_terms, f, i, r))
    base = embed(texts)

    # The null is the same plant with NEUTRAL vocabulary, AT THE SAME FRACTION AND INTENSITY as
    # the cell it scores. Two earlier versions were wrong in the same way -- a comparison that
    # did not isolate what it claimed to. A bootstrap of the unplanted vectors duplicated a
    # quarter of the rows, and duplicate pairs have cosine 1.0, so its threshold sat four and a
    # half times above the statistic's value on real data. Replacing it with a single neutral
    # plant at MAXIMUM fraction and intensity was no better: inserting text at all moves the
    # statistic, by changing length and token mix, so every lighter cell was scored against a
    # twelvefold heavier intervention. Both produced a uniform zero, which reads as a blind
    # instrument and was a broken comparison.
    # Per-cell nulls cost one extra embedding pass per cell and are the only version that
    # isolates the driver vocabulary from the act of insertion.
    rows = []
    for f in fractions:
        for it in intensities:
            rng0 = np.random.default_rng(seed + 7717 + int(f * 1000) + it)
            null_vals = []
            for _ in range(max(8, n_rep)):
                neut, nidx = plant_text_effect(texts, list(NEUTRAL_TERMS), f, it, rng0)
                if len(nidx) < 2:
                    continue
                V = base.astype(np.float64).copy()
                V[nidx] = embed([neut[i] for i in nidx])
                Vc = _normalise(V)
                null_vals.append(instrument(Vc[nidx] if basket is None else Vc[
                    rng0.choice(len(Vc), basket, replace=False)]))
            if not null_vals:
                continue
            thr = float(np.quantile(null_vals, 1.0 - alpha))
            rng = np.random.default_rng(seed + int(f * 1000) + it)
            hits = 0
            for _ in range(n_rep):
                planted, idx = (plant(texts, f, it, rng) if novel
                                else plant_text_effect(texts, driver_terms, f, it, rng))
                if len(idx) < 2:
                    continue
                V = base.astype(np.float64).copy()
                V[idx] = embed([planted[i] for i in idx])
                Vc = _normalise(V)
                val = instrument(Vc if basket is None else Vc[
                    rng.choice(len(Vc), basket, replace=False)])
                hits += int(val > thr)
            rows.append({"fraction": f, "intensity": it, "n_rep": n_rep,
                         "threshold": thr, "null_draws": len(null_vals),
                         "novel_tokens": novel, "detection_rate": hits / n_rep})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ returns-layer plants
#
# A third plant layer, added 2026-08-25 for exp-044.
#
# The vector-layer certificates issued so far certify EMBEDDING-COHESION instruments -- mean
# pairwise cosine, leading eigenvalue, densest subset -- on a pool of vectors. The comparator
# gauntlet's verdict statistic is not one of those. It is co-movement lift: the mean within-group
# correlation of market-residualised returns, minus a characteristic-matched permutation null.
# Different instrument, different layer, different pool. The existing certificates say nothing
# about it, and using them here would be precisely the over-reading the scope line in exp-044's
# gates exists to forbid.
#
# The plant is therefore made in returns: a subset of names is given a genuine common factor at
# a calibrated correlation, and the certificate reports at what (fraction, correlation) a
# perfectly-informed grouping would be detected. That is an UPPER BOUND on every arm, because no
# representation can do better than naming the planted set exactly.


def _mean_pairwise_corr(C: np.ndarray, idx: np.ndarray) -> float:
    """Mean off-diagonal correlation within a set, from a precomputed matrix."""
    if len(idx) < 2:
        return float("nan")
    sub = C[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    v = sub[iu]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def mean_within(C: np.ndarray, labels: np.ndarray, *, min_size: int = 4) -> float:
    """Size-weighted mean of within-group correlation. The statistic `comovement` already uses.

    Taken from a PRECOMPUTED correlation matrix rather than recomputed per permutation. A
    permutation only relabels names; the correlations between them do not change, so
    recomputing them is a thousand redundant passes over the panel and the reason a
    certificate over this instrument looked unaffordable.
    """
    tot = w = 0.0
    for g in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == g)
        if len(idx) < min_size:
            continue
        m = _mean_pairwise_corr(C, idx)
        if m == m:
            tot += m * len(idx)
            w += len(idx)
    return float(tot / w) if w else float("nan")


def max_within(C: np.ndarray, labels: np.ndarray, *, min_size: int = 4) -> float:
    """The single most cohesive group. Sensitive to ONE real cluster among noise, where the
    size-weighted mean dilutes it across every group that carries nothing."""
    best = float("-inf")
    for g in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == g)
        if len(idx) < min_size:
            continue
        m = _mean_pairwise_corr(C, idx)
        if m == m:
            best = max(best, m)
    return best if best > float("-inf") else float("nan")


def eigen_within(C: np.ndarray, labels: np.ndarray, *, min_size: int = 4) -> float:
    """Size-weighted mean of each group's excess leading eigenvalue, scaled to [0, 1].

    (lambda_1 - 1)/(k - 1) equals the mean pairwise correlation for an exactly equicorrelated
    block and exceeds it when the group's co-movement runs along one direction rather than
    being spread evenly -- which is what a real common factor looks like.
    """
    tot = w = 0.0
    for g in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == g)
        if len(idx) < min_size:
            continue
        sub = C[np.ix_(idx, idx)]
        sub = np.nan_to_num(sub, nan=0.0)
        np.fill_diagonal(sub, 1.0)
        lam = float(np.linalg.eigvalsh(sub)[-1])
        tot += ((lam - 1.0) / (len(idx) - 1)) * len(idx)
        w += len(idx)
    return float(tot / w) if w else float("nan")


COMOVEMENT_STATISTICS = {"mean_within": mean_within, "max_within": max_within,
                         "eigen_within": eigen_within}


def plant_common_factor(R: np.ndarray, members: np.ndarray, target_rho: float,
                        rng: np.random.Generator, *, tol: float = 0.005,
                        max_iter: int = 40) -> np.ndarray:
    """Give `members` a common factor calibrated to a target mean pairwise correlation.

    Calibrated numerically rather than from the closed form lambda = sqrt(rho/(1-rho)), because
    that form assumes the members start uncorrelated. They do not: residual returns keep several
    points of shared variation after the market is projected out, so the closed form overshoots
    by an amount that varies with the draw. Bisection on the measured correlation costs
    milliseconds and removes the assumption.
    """
    T = R.shape[0]
    g = rng.standard_normal(T)
    sd = np.nanstd(R[:, members], axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nanmean(sd))

    def measured(lam: float) -> float:
        Rp = R.copy()
        Rp[:, members] = R[:, members] + lam * sd[None, :] * g[:, None]
        C = np.corrcoef(Rp[:, members], rowvar=False)
        iu = np.triu_indices(len(members), 1)
        v = C[iu]
        return float(np.nanmean(v))

    base = measured(0.0)
    if target_rho <= base:
        return R.copy()
    lo, hi = 0.0, 4.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if measured(mid) < target_rho:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    lam = 0.5 * (lo + hi)
    if abs(measured(lam) - target_rho) > tol * 4:
        raise ValueError(f"could not calibrate a common factor to rho={target_rho}")
    Rp = R.copy()
    Rp[:, members] = R[:, members] + lam * sd[None, :] * g[:, None]
    return Rp


def matched_shuffle(labels: np.ndarray, cells: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Shuffle group labels WITHIN characteristic cells, preserving every group's size.

    The cell is the whole point. A free shuffle is beaten by any partition that tracks sector at
    all, which is the confound three earlier text experiments here died on; shuffling inside
    sector-by-size cells makes the null already know the industry and the size tilt, so the lift
    that survives is lift beyond them.
    """
    out = labels.copy()
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        vals = labels[idx].copy()
        rng.shuffle(vals)
        out[idx] = vals
    return out


def comovement_test(C: np.ndarray, labels: np.ndarray, cells: np.ndarray, statistic,
                    *, n_perm: int = 1000, seed: int = 0, min_size: int = 4) -> dict:
    """Observed statistic against its characteristic-matched permutation null."""
    rng = np.random.default_rng(seed)
    obs = statistic(C, labels, min_size=min_size)
    draws = np.array([statistic(C, matched_shuffle(labels, cells, rng), min_size=min_size)
                      for _ in range(n_perm)])
    draws = draws[np.isfinite(draws)]
    if not len(draws) or not np.isfinite(obs):
        return {"observed": obs, "lift": float("nan"), "p": float("nan"), "null_mean": float("nan")}
    return {"observed": float(obs), "null_mean": float(draws.mean()),
            "null_sd": float(draws.std()), "lift": float(obs - draws.mean()),
            "p": float((1 + (draws >= obs).sum()) / (1 + len(draws)))}


def comovement_detection_curve(R: np.ndarray, cells: np.ndarray, statistic, *,
                               fractions=(0.10, 0.20, 0.30, 0.40),
                               rhos=(0.05, 0.10, 0.20, 0.30, 0.40),
                               n_groups: int = 11, n_rep: int = 40, n_perm: int = 200,
                               alpha: float = 0.05, seed: int = 0,
                               min_size: int = 4) -> pd.DataFrame:
    """Detection rate over a (fraction planted, planted correlation) grid.

    The labels handed to the test give the planted set its own group and split the remainder
    into groups of comparable size, so the arm being certified is a PERFECT one: a
    representation that named the planted set exactly. Every real arm is bounded above by this,
    which is what makes the number a floor on what the experiment can conclude rather than a
    description of any particular representation.
    """
    rng = np.random.default_rng(seed)
    N = R.shape[1]
    rows = []
    for f in fractions:
        k = max(min_size, int(round(f * N)))
        for rho in rhos:
            hits = 0
            for rep in range(n_rep):
                members = rng.choice(N, size=k, replace=False)
                try:
                    Rp = plant_common_factor(R, members, rho, rng)
                except ValueError:
                    continue
                C = np.corrcoef(Rp, rowvar=False)
                labels = np.empty(N, dtype=int)
                rest = np.setdiff1d(np.arange(N), members)
                labels[members] = 0
                # Remaining names split into groups of the planted set's size, so no group is
                # distinguished by size alone -- a size-matched null would otherwise be beaten
                # by the planted group simply for being the largest.
                rng.shuffle(rest)
                for i, nm in enumerate(rest):
                    labels[nm] = 1 + (i // max(1, k))
                res = comovement_test(C, labels, cells, statistic, n_perm=n_perm,
                                      seed=int(rng.integers(1 << 30)), min_size=min_size)
                hits += int(res["p"] <= alpha)
            # Column names match the vector-layer curve exactly, so mde_region reads either
            # without knowing which layer produced it. A near-miss here cost one run.
            rows.append({"fraction": f, "target_cos": rho, "n_rep": n_rep,
                         "detection_rate": hits / max(n_rep, 1)})
    return pd.DataFrame(rows)


def certify_comovement(R: np.ndarray, cells: np.ndarray, tickers: list[str], *, name: str,
                       power: float = 0.80, alpha: float = 0.05, n_rep: int = 40,
                       n_perm: int = 200, seed: int = 0, **grid) -> Certificate:
    """One call from a residual-return panel to the certificate a registration must carry."""
    stat = COMOVEMENT_STATISTICS[name]
    curve = comovement_detection_curve(R, cells, stat, n_rep=n_rep, n_perm=n_perm,
                                       alpha=alpha, seed=seed, **grid)
    # False alarms: the same test on UNPLANTED returns with a grouping that carries no
    # information. Anything above alpha here means the matched null is not matched enough.
    rng = np.random.default_rng(seed + 977)
    N = R.shape[1]
    C0 = np.corrcoef(R, rowvar=False)
    k = max(4, N // 11)
    # 100 replicates was the first choice and it is not enough to certify anything. A rate
    # near 0.05 estimated from 100 draws carries a standard error of sqrt(.05*.95/100) = 0.022,
    # so 0.05 and 0.13 are two standard errors apart and indistinguishable -- and two runs of
    # this very certificate, at different seeds, returned exactly that pair for mean_within.
    # A false-alarm number that cannot tell nominal from two-and-a-half times nominal is a
    # certification layer failing to certify itself. 400 halves the error to 0.011.
    n_fa = max(400, n_rep * 10)
    fa = 0
    for _ in range(n_fa):
        lab = np.arange(N) // k
        rng.shuffle(lab)
        fa += int(comovement_test(C0, lab, cells, stat, n_perm=n_perm,
                                  seed=int(rng.integers(1 << 30)))["p"] <= alpha)
    return Certificate(
        instrument=name, plant_layer="returns",
        effect_shape="a sub-group given a genuine common factor",
        null_construction=(f"group labels shuffled within sector-by-size cells, {n_perm} draws, "
                           f"alpha {alpha}, on a panel of {N} names"),
        false_alarm_rate=fa / n_fa, false_alarm_n=n_fa,
        mde=mde_region(curve, power=power, alpha=alpha), curve=curve,
        # Fingerprint the PANEL, with the names folded into the label: the null is drawn from
        # both, so a certificate is only valid for the pair.
        universe_hash=universe_hash(R, label=f"{name}|{'|'.join(sorted(tickers))}"),
        universe_n=N,
        # The CELL ASSIGNMENT is the null here. Repairing it -- so no cell spans a sector
        # boundary -- changed every false-alarm rate, and nothing recorded that it had.
        null_hash=null_hash(cells, n_perm, alpha, n_fa, "matched_shuffle", label=name))
