# The analogue retrieval system — handover

**Audience:** someone taking over the experiments with no prior knowledge of this project.
**This is not the memo.** It is the working document: what the system is, why each choice was
made, what has been tested and settled, what is known to be broken, and what to run next.

Established empirical findings across the whole project live in `project/claims.yaml` (62
claims) and are rendered into `LEDGER.md`. This file covers the analogue/regime subsystem
specifically, which is newer than most of that ledger and largely absent from it.

**Last updated 2026-08-29.** Several numbers in earlier documents are superseded by the
inference correction in §7.1 — when this file and an older one disagree, this one is right.

---

## 1. What the system is

A momentum PM wants to know how bad a reversal could get, why the book is exposed, and what
this situation resembles. The system answers the third question directly and uses it to serve
the first two.

**Object:** the US equity momentum factor (Ken French daily WML, winners-minus-losers decile
spread, 1927-2026).

**Target:** severity, not probability. Specifically the **worst cumulative drawdown of WML
over the next 63 trading days**. This choice is forced and should not be revisited casually:
there are roughly 8-17 independent reversal regimes in the modern era, which cannot support a
probability estimate, but severity is a conditional distribution answerable from thousands of
daily observations. The brief asks for "probability and/or severity"; severity is the honest
half.

**Horizon:** 63 trading days (one quarter).

**User and decision:** a discretionary PM sizing momentum exposure. The output is an input to
judgement, NOT a systematic signal. This framing is load-bearing and changes what counts as
success — see §6.

**Core mechanic:** given today's state, retrieve the most similar prior states, show what
happened after them, show what the book was betting on each, and show how those bets differed
from today's.

---

## 2. Architecture — three layers

```
  LAYER 1  RETRIEVAL      which prior dates resemble today?
           deep_archive.py -- 5 price features, 25,671 days, 1928-2026
           output: 8 named dates + their realised forward drawdowns

  LAYER 2  MECHANISM      what is the book actually betting on, then and now?
           poc_engine.py -- conferred-exposure decomposition, 2,619 days, 2013-2026
           output: beta/value/size/duration tilts + per-match deltas

  LAYER 3  NARRATIVE      what was the story, and does today's story rhyme?
           news_dejavu.py -- episode narrative signatures, 8 episodes, 2018-2025
           output: cross-episode rhymes + certificates
```

Layers 2 and 3 **annotate**; they never select. That is a measured decision, not a
preference — see §4.3 and §5.2.

---

## 3. Layer 1 — the retrieval engine

`src/unstructured_momentum/pipeline/deep_archive.py`

### 3.1 Data

Ken French daily momentum decile portfolios (`data/raw/french/mom_10_daily/`), 26,174 rows.
WML = `Hi PRIOR` - `Lo PRIOR`. Public, free, no employer data — a binding constraint of the
brief. Market factor from F-F daily research factors.

25,671 usable days after feature warm-up, 1928-07-11 to 2026-06-30.

### 3.2 The key — five price features

```python
COLS = ["vol21", "vol63", "mom252", "mkt252", "dd"]
```

trailing 21d and 63d WML volatility, trailing 252d WML return, trailing 252d market return,
and current drawdown from the running WML peak.

Each is standardised on a **trailing** 756-day window (min 252), clipped to ±5. Trailing, not
full-sample: a full-sample z-score leaks the future into the key.

**Why only five, and why price-only.** Van den Dool (1994): the library needed for a close
analogue grows exponentially in degrees of freedom. Measured on our own data, 8th-neighbour
distance as a share of a random pair:

| features | days | 8th neighbour |
|---|---|---|
| 3 | 25,671 | 17% |
| 4 | 25,671 | 29% |
| 5 (current) | 25,671 | 41% |
| 7 (+ holdings) | ~2,400 | 74% |
| 9 (+ holdings + options) | 1,078 | 96% |

At 9 dimensions the 8th "analogue" is a randomly chosen date wearing a similarity label. Any
holdings- or options-derived feature also truncates the archive by ~90% and drops bear-state
coverage from 24% to 10-13% — removing the very states the tool exists to describe.

### 3.3 Eligibility — stricter than standard

A neighbour must (a) precede the query AND (b) have its own 63-day outcome window **closed**
before the query date. Condition (b) is stricter than the usual as-of convention and closes a
leakage channel most of this literature does not name. Do not relax it.

### 3.4 Collapse / exclusion zone

`COLLAPSE = 63` days. Two neighbours within 63 days of each other are the same episode counted
twice. Follows the matrix-profile convention of m/4 on a 252-day lookback.

### 3.5 Hysteresis

`PERSIST_MARGIN = 0.10`. An incumbent neighbour keeps its slot unless a challenger is ≥10%
closer. Without it, day-to-day neighbour-set overlap is 50% and more than half the set turns
over on 41% of days — the *story* reshuffles overnight, which is useless to a human.
With it: overlap 88%, high-turnover days 6%.

Selected as the smallest margin reaching 80% median overlap — a **structural** criterion fixed
before the sweep. Skill could not have chosen it: on 600 consecutive days the skill column is
~10 independent observations.

### 3.6 The analogue floor

`FLOOR_FRAC = 0.50`. A neighbour beyond 50% of a random-pair distance is annotated
`is_analogue = False` rather than deleted. "No analogue exists" is a legitimate and often
honest answer; suppressing the row hides that.

### 3.7 Distance metric

Unweighted Euclidean. Mahalanobis whitening (Ledoit-Wolf shrunk) was tested: it finds *closer*
neighbours (13% vs 16% of a random pair) and scores *worse* against the time-matched null
(6.0% vs 7.8%). The audit's logic was sound — Euclidean is Mahalanobis with correlations forced
to identity, and our two volatility features are heavily correlated — but the implied
over-weighting of volatility turns out to be economically load-bearing for momentum reversals.
Kept available (`metric="mahalanobis"`), not the default.

**Known gap:** a *learned or weighted* metric has never been tried. See §8.

---

## 4. Layer 2 — the mechanism layer

`src/unstructured_momentum/pipeline/poc_engine.py`

### 4.1 What conferred exposure is

The momentum sort **manufactures** factor loadings rather than discovering them. Ranking on
past returns automatically concentrates into whatever the market has been rewarding, so the
book acquires beta, value, size and duration tilts nobody chose. Measured on the 2019-09 book:
1.46-point beta spread, 100th percentile against 200 cap-matched placebo books, present in
104 of 128 months (`clm-selection-confers-exposure`, `clm-defensive-is-modal`).

**Prior art — state this honestly.** Grundy & Martin (2001) established momentum's time-varying
factor exposures and Daniel-Moskowitz built the bear-state result on it. Conferred exposure is
**not new in kind**. What is ours is narrower: daily measurement on a live book with placebo
calibration, used as an explanation layer on retrieved analogues.

### 4.2 The features

`beta_spread`, `hml_spread`, `smb_spread`, `d10y_spread`, `tilt_pctile` — all
winners-minus-losers, so positive always means "the long leg carries more of it".
2,619 days, 2013-12-02 to 2026-02-09 (daily holdings start Dec-2013).

`tilt_pctile` uses an **expanding** window, so it is as-of. Its first ~year is unreliable
(an expanding percentile with almost no history pins to 1.0).

### 4.3 Why mechanism ANNOTATES and never SELECTS

Putting these five into the retrieval distance was built, measured and rejected:

| key | archive | 8th neighbour |
|---|---|---|
| 5 price features, 1928-2026 | 25,671 days | **41%** |
| 5 price features, 2014+ | 3,141 days | 70% |
| 14 features (mech + dispersion), 2015+ | 1,857 days | **96%** |

At 96% the engine names statistically arbitrary dates. **This is the one place where the skill
metric and the user need come apart**: skill is scored against a time-matched null that is also
arbitrary dates, so a key can score well while returning matches nobody should look at. For a
tool whose output is read as "this resembles Nov-2022", retrieval quality binds.

### 4.4 Two panels

Because deep retrieval mostly returns pre-2014 dates, typically only 1 of 8 matches carries a
bet decomposition. So the readout runs two panels answering different questions:

- **DEEP** (1928-2026): matches at 18-36% of a random pair. Tight. Mostly no bet.
- **MODERN** (2014+, same 5 price features): matches at 27-59%. Looser, labelled as such.
  Every match carries its bet.

The modern panel is deliberately **not** the 14-feature key.

### 4.5 Options — decided, not yet wired

`mfis30`, `mfis91`, `mfis_slope` (model-free implied skew on MTUM), 1,622 days,
2017-08-01 to **2024-01-03**.

**Not in the key.** Two reasons hold: the series is 969 days stale so it cannot be computed
today, which disqualifies it for a monitoring key; and it truncates the archive to 1,363 days
and pushes the 8th neighbour to 74-96%.

**A third reason previously given is WITHDRAWN.** "Options scored worst at -7.8% skill" came
from a ~120-query sweep, the same underpowered sweep that produced two other figures that later
died. A later run put options at +0.5%, CI [-0.0019, +0.0564] on 7 blocks. The skill evidence
never decided anything about options in either direction.

**Status: should annotate matches, not yet implemented.** Note the asymmetry — it can say what
the option market priced at each historical analogue but not what it prices now.

---

## 5. Layer 3 — the narrative layer

`scripts/news_dejavu.py`

### 5.1 What it does

Replicates Franklin, Silcock, Arora, Bryan & Dell, "News Déjà Vu" (NLP+CSS @ ACL 2024).
Retrieves not articles but **narrative signatures**: the mean title embedding for one
(episode, day, leg) cell of the book. The PM's question is "when did the story around this book
last look like this", not "which article resembles this article".

Corpus: GDELT episode titles, `data/corpus/gdelt/episode_titles/*.parquet`, 8 episodes
(2018-02 volmageddon through 2025-04 tariff), ~72 days each, ~86k titles, winners/losers legs.

### 5.2 Why text cannot be a retrieval key

8 episodes, ~576 non-contiguous days. There is no daily text series back to 1928 and building
one is a decade-scale acquisition we have not made. **This is a data fact, not a judgement.**
Text describes and evidences; it never picks the dates.

### 5.3 Encoders and the leakage design

- **ChronoBERT-20151231** — trained only on text timestamped before 2016, so it cannot know how
  any 2018-2025 episode resolved. Leakage-free *by construction*. This matters because the
  project measured that date-stripping fails outright: a frontier model placed 40/40 historical
  windows to the exact year with every date removed (`clm-date-blinding-fails`).
- **all-mpnet-base-v2** — the off-the-shelf contrastive encoder News Déjà Vu found wanting, and
  also the leakage-exposed arm (its training data postdates every episode).

**One encoder for all episodes.** An earlier run used a per-episode vintage, which made encoder
identity a function of date, so embedding distance partly measured *which checkpoint* rather
than *which narrative*. 6 of 8 nearest pairs shared a vintage. Use
`episode_embeddings_single.parquet`, never `episode_embeddings.parquet`.

### 5.4 Certificates run before any retrieval is read

An instrument reports nothing until shown able to see. Two tests, both against nulls:
**leg separation** (winners vs losers on the same day should be further apart than two
different days of the same leg — proves the encoder reads *what*, not just *when*) and
**episode coherence** vs a 500-draw permutation null.

Result (2026-08-29): both PASS.

| encoder | leg gap | coherence | perm p |
|---|---|---|---|
| ChronoBERT-2015 | +0.0264 | +0.1514 | 0.000 |
| all-mpnet (OTS) | +0.1015 | +0.1282 | 0.000 |

Embeddings are cross-sectionally centred before any distance — transformer spaces are strongly
anisotropic and this project measured a ~30x separation gain from removing the corpus mean.

### 5.5 Finding

**News Déjà Vu's negative result does not replicate here.** The off-the-shelf encoder is fine —
better on leg separation, slightly worse on episode coherence. Both agree on substance:
2021-01 squeeze ↔ 2020-11 vaccine dominates (66-76% of days), 2025-04 tariff ↔ 2024-08 unwind
(66-73%). **The vintage-locked, leakage-impossible encoder finds the same structure as the
leakage-exposed modern one — so the safe choice costs nothing.**

---

## 6. Scoring methodology

### 6.1 Threshold-weighted CRPS

Gneiting & Ranjan (2011). The 8 neighbours' realised outcomes form an empirical predictive
distribution; twCRPS scores it against the actual outcome with weight concentrated in the
lower tail (that is what we care about). Weight function `1 - Φ(z; median, sd)` of the
eligible-pool outcome distribution.

### 6.2 The time-matched null — the central control

The right question is not "does the analogue set beat climatology" but **"does *similarity* do
work, or only *recency*?"** So the null holds the temporal footprint fixed — same number of
dates, same gap structure between them, anchored at a random point in the eligible window — and
varies **only** the feature-based selection.

This is not standard in the analogue literature, which usually compares to climatology. The
KDD 2026 stationarity-aware RA-TSF paper names exactly this question — input similarity is a
dependable proxy for future similarity on some series and not others — and this is the
instrument that answers it for ours.

### 6.3 Block bootstrap — READ §7.1 FIRST

Paired difference in twCRPS between the analogue set and the null, averaged within blocks,
resampled over blocks.

**Block length must exceed the outcome horizon.** With H=63 trading days and a query stride of
`s`, use `ceil(63/s)` queries per block. Shorter blocks share outcome windows and the resample
treats near-identical observations as independent.

### 6.4 Not implemented — Ferro's fair CRPS

Ferro (2014): a finite ensemble of size M carries a bias of `(1/2M)·E|X₁-X₂|`, scaling with
ensemble spread. At k=8 this is material and inflates our scores in an unquantified direction.
`m_eff()` exists in `deep_archive.py` but the correction itself does not. **This is a real gap.**

---

## 7. Known bugs, traps and corrections

### 7.1 THE BLOCK BOOTSTRAP WAS UNDER-BLOCKED (found 2026-08-29)

Earlier work used blocks of 4 consecutive queries. At H=63 and stride 2-4, four queries span
8-16 days and share >90% of their outcome window with adjacent blocks. **Every interval
computed that way is too narrow, and some "excludes zero" verdicts in the ledger will not
survive re-computation.**

Point estimates are unaffected. **Re-running the ledger's inference with correct blocks is the
highest-priority outstanding task.**

### 7.2 Results withdrawn on 2026-08-29

| withdrawn claim | why | corrected |
|---|---|---|
| "conferred exposure improves retrieval, 9.8%" | span confound | 6.3% vs 6.3% span-matched — no effect |
| "dispersion doubles retrieval skill, 19.5%" | 22 blocks | ties on 391 blocks, full archive |
| "adding rich features to the key hurts" | wrong in the OTHER direction; compared across spans on ~120 queries | on a fixed span they help; on the full archive they tie |
| "options score worst, -7.8%" | ~120 queries | uninformative; decision rests on staleness + neighbour quality |

**Pattern to internalise:** every one of these was a large, encouraging number on a short
window. The repo rule — *an implausible number is an instrument bug until proven otherwise* —
applies to results you like, not only to results you don't.

### 7.3 Older traps that still bind

- **Vintage confounding** in episode embeddings (§5.3).
- **Syndication**: a headline appearing 50 times is one observation. Dedup by title before
  embedding. Two thirds of "significant" themes once came from copies of one press release.
- **Anisotropy**: raw transformer cosines are near-useless without cross-sectional centering.
- **HARKing**: a mechanism-family test was once withdrawn *after* seeing it fail. Correct
  handling is to report it and label the reinterpretation post-hoc, not delete it.
- **Testing the component, never the integration**: a signature patch silently failed to apply
  and was "verified" by calling the helper directly. 140 downstream calls were wrong.

### 7.4 The encoder investigation (registered, both predictions refuted)

`reports/encoder_test/REGISTRATION.md` — predictions frozen and committed before running.

| encoder | vendor | objective | skill | vs 5 features |
|---|---|---|---|---|
| 5 hand-picked features | — | — | 7.4% | benchmark |
| Chronos-t5-small | Amazon | autoregressive tokens | 12.1% | **BETTER** |
| Chronos-Bolt-small | Amazon | direct patch quantile | 13.3% | **BETTER** |
| MOMENT-1-small | CMU | masked reconstruction | 7.8% | tie |
| TimesFM-2.5-200M | Google | decoder forecasting | 4.6% | tie |

The split is by **vendor**, not objective, architecture or scale. Vendor is not a mechanism, so
the advantage is unexplained and the frozen decision rule kept the existing key.

The Chronos positive survived six ablations — era split (no memorisation signature: positive in
1931-59, neighbours *not* pulled toward the modern era), raw 512d path (tie), 64/32 bucket
reductions (worse), fixed random projection to 512d (tie), as-of PCA to 8/32 (worse). So
pretraining does contribute something real; it just does not travel outside one vendor.

---

## 7.5 The 2026-08-29 method sweep — eleven alternatives, one improvement

Every modern alternative in the surveyed literature was tested as a retrieval key against the
incumbent, on identical paired queries with correctly-sized blocks.

| alternative | family | result |
|---|---|---|
| Chronos-t5-small | TS foundation model | BETTER (12.1% vs 7.4%) |
| Chronos-Bolt-small | TS foundation model | BETTER (13.3%) |
| MOMENT-1-small | TS foundation model | tie |
| TimesFM-2.5 | TS foundation model | tie |
| contrastive encoder, pre-1980 fit | trained self-supervised | tie, trending worse |
| Wasserstein-1 (32 quantiles) | optimal transport | tie (6.9% vs 6.7%) |
| MMD via random Fourier features | kernel two-sample | tie; TIGHTEST neighbours measured (37%) |
| log-signature depth 2 | rough paths | **WORSE** |
| trajectory +/-5d | Delle Monache 2013 | tie |
| fitted weighted metric | Delle Monache 2013 | tie (train +8.8% -> test 7.4% vs 9.1%) |
| devolatilised shape (EWMA) | GARCH-residual analogue | 0.2% -- no skill at all |

**Only the two Chronos models beat the key, and that does not replicate across vendors (§7.4).**

### What DID work: ensemble width

Changing how retrieved outcomes are COMBINED, holding retrieval fixed:

| rule | skill | vs k=8 |
|---|---|---|
| k=8 flat (old default) | 6.8% | — |
| **k=16 flat** | **7.9%** | **[+0.00146, +0.00234] BETTER** |
| k=32 flat | 10.2% (1980+ run) | better, but k=16 wins on the full archive |
| k=32 kernel-weighted | identical to flat | RAFT-style weighting adds NOTHING |

Replicated on two independent runs. **Use ~16 neighbours.** The interior optimum (k=16 > k=32)
argues the effect is real rather than mechanical.

### The trap this nearly became

The first aggregator run reported k=32 at 16.6% against 9.1% -- a near-doubling. It was an
instrument bug: the time-matched null was built with 8 dates regardless of the rule, so a
32-member analogue ensemble was scored against an 8-member null. twCRPS carries a finite-ensemble
bias of `(1/2M)E|X1-X2|` (Ferro 2014) that shrinks with M, so the larger ensemble won
mechanically. Size-matching the null removed two thirds of the gain.

**Any experiment that varies ensemble size must size-match its null.** This also promotes §6.4
from bookkeeping to necessary: the finite-ensemble bias is large enough at k=8 to manufacture a
result.

### Volatility is most, but not all, of the signal

Three independent routes converged on volatility dominance: Mahalanobis lost by down-weighting
the correlated vol pair; the strongest modern-span block was four volatility features; and a
from-scratch contrastive encoder came out 78% linearly explained by trailing vol.

Stratified retrieval separates the layers — hold volatility fixed, then rank within the band:

| retrieval | skill |
|---|---|
| full 5-feature key | 6.8% |
| vol block only (2 features) | 5.5% |
| vol-matched -> rest (3 features) | 7.6% |
| vol-matched -> RANDOM | 5.7% |
| devolatilised shape | 0.2% |

Informed stage 2 beats chance within the same vol band ([+0.000083, +0.001854], 387 blocks), so
**non-volatility information does carry value**. But staged retrieval does NOT beat the full key
(tie, [-0.00010, +0.00074]) — the flat key already exploits it. Staging's value is
**interpretive, not statistical**: it decomposes a match into "which volatility regime" and
"what distinguished it within that regime", which is a better sentence for a PM than an
undifferentiated distance.

Caveat: the stage-2 interval's lower bound is +0.000083, and ~15 configurations were tested that
day. It would not survive multiplicity correction. Promising, not established. The k=16 result
is much sturdier.

### The recurring divergence — proximity is not relevance

Four independent occurrences of tighter neighbours scoring WORSE:

| method | neighbour distance | skill |
|---|---|---|
| Mahalanobis | 13% vs 16% (tighter) | 6.0% vs 7.8% (worse) |
| MMD via RFF | 37% vs 41% (tighter) | 3.9% vs 6.7% (worse) |
| fitted weighted metric | 28% vs 32% (tighter) | 7.4% vs 9.1% (worse) |
| 14-feature PoC key | 96% (far worse) | scored better on some runs |

**Proximity in feature space and relevance to the outcome are different objectives here, and
optimising either degrades the other.** This is why the PoC key was chosen on neighbour distance
rather than skill (§4.3), and it is not a point the surveyed analogue literature makes.

---

## 7.6 THE DECISIVE COMPARISON — the engine does not beat volatility (2026-08-30)

Every result above was scored against a TIME-MATCHED NULL: random dates with the same temporal
footprint. That is a floor, not a benchmark. The engine had never been compared to the
project's existing severity machinery. It now has.

Same dates, same outcome (worst 63d WML drawdown), same twCRPS. Lower is better.

| method | twCRPS | vs vol-scaled climatology |
|---|---|---|
| **vol-scaled climatology (full)** | **0.023400** | — |
| vol-scaled climatology, k=8 | 0.026860 | WORSE |
| analogue k=16 | 0.027127 | [-0.00518, -0.00247] **WORSE** |
| climatology (full) | 0.027578 | WORSE |
| analogue k=8 | 0.028860 | [-0.00695, -0.00414] **WORSE** |
| climatology k=16 | 0.029301 | WORSE |
| climatology k=8 | 0.031422 | WORSE |

Vol-scaled climatology = each past outcome divided by the trailing vol prevailing THEN, rescaled
to today's vol. One line of arithmetic, no archive, no key, no retrieval.

### Three findings

1. **The analogue engine does not beat trailing volatility.** Not at k=8, not at k=16. And not
   because of ensemble size: vol-scaled climatology with EIGHT RANDOM members still beats the
   analogue set at eight ([-0.00333, -0.00068]).

2. **Similarity does extract real information — it is simply dominated.** Against random
   climatology at matched size the analogue set wins at both k=8 ([+0.00125, +0.00396]) and
   k=16 ([+0.00115, +0.00330]). Retrieval works. Vol-scaling works better and is free.

3. **The k=8 -> k=16 "improvement" of §7.5 is a finite-ensemble effect, not retrieval.**
   Doubling k improved the analogue set by 0.00173 and RANDOM CLIMATOLOGY by 0.00212 — more.
   Any ensemble gains from more members (Ferro 2014); this told us nothing about analogues.
   **§7.5's one surviving improvement is withdrawn as a retrieval finding.**

### What this changes

The severity NUMBER comes from vol-scaling or GARCH. The conventional method wins, and the
brief explicitly asks where that is the case (`clm-nothing-beats-vol` said the same about six
other candidate signals; the analogue engine is now the seventh).

The analogue engine's job is what vol-scaling cannot do: name WHICH historical states these
are, show what the book was betting on each (§4), and evidence it with contemporaneous text
(§5). Descriptive and interpretive, not predictive.

### The methodological lesson

Eleven method experiments were ranked by a proper scoring rule for probabilistic FORECASTS,
for a tool explicitly framed as discretionary input rather than a forecaster — and the
incumbent it would have to displace was never in the comparison. The time-matched null answers
"does similarity beat recency", which is a real question; it does not answer "is this worth
building".

**Evaluation criteria, going forward — assign each decision to one and say which:**

  RETRIEVAL QUALITY      neighbour distance vs a random pair, churn, face validity.
                         Governs KEY SELECTION. (Already the basis for §4.3.)
  SEVERITY INFORMATIVENESS  twCRPS benchmarked against VOL-SCALED CLIMATOLOGY and GARCH,
                         never against the time-matched null alone.
                         Governs ENSEMBLE CONSTRUCTION -- where the engine currently loses.

Two reporting faults to fix: skill is quoted as a ratio of means (`1 - mean(nb)/mean(tm)`)
while the interval is on a difference of block means, so the CI does not bound the percentage;
and the null uses only 4 draws per query.

---

## 8. Experiment queue

Ordered. Items 1-2 are corrections to existing work and come first.

1. **Re-architect the readout around §7.6.** The severity number comes from vol-scaled
   climatology (or GARCH); the analogue panels supply WHICH states and WHAT the book was
   betting. Stop presenting the analogue ensemble as the severity estimate.
2. **Implement Ferro fair CRPS** (§6.4). Now clearly necessary: the finite-ensemble bias
   manufactured one headline result (§7.5) and inflated another (§7.6).
3. **Add GARCH to the benchmark set.** §7.6 used climatology and vol-scaling; the repo's
   GJR-GARCH skewed-t (`model/baselines.py`) is the third incumbent and is not yet scored on
   this outcome.
4. **Add the staged readout** (§7.5) — no skill gain, but it separates "which vol regime" from
   "what distinguished it", which reads better to a PM.
5. **Wire options as match annotation** (§4.5).
6. **Re-run ledger inference with correct blocks** (§7.1).

**NOT on the queue any more:** adopting k=16 (withdrawn, §7.6) and any further retrieval-key
search (§7.5 — eleven alternatives, none replicably better, and the whole family is dominated
by vol-scaling anyway).

Items 3-8 of the previous queue are **done** — see §7.5. Do not re-run them.

### What NOT to re-run

Settled, with adequate power, do not revisit without new data: conferred exposure in the key
(no effect span-matched); dispersion and crowding on the full archive (ties, 391 blocks); text
as a retrieval key (impossible, data); >5 hand-picked price features (ties); and the eleven
alternatives in §7.5 — foundation models, trained contrastive encoders, optimal transport,
kernel two-sample distances, rough-path signatures, trajectory matching, fitted weighted
metrics, and devolatilised return shape.

---

## 9. Reproducing the current state

```bash
# Layer 1 -- retrieval, and the two-panel readout with mechanism annotation
.venv/bin/python -m unstructured_momentum.pipeline.poc_engine 2026-02-09

# Layer 3 -- narrative certificates and cross-episode rhymes
.venv/bin/python scripts/news_dejavu.py
```

Key artefacts:

| path | what |
|---|---|
| `src/.../pipeline/deep_archive.py` | retrieval engine, 1928-2026 |
| `src/.../pipeline/poc_engine.py` | mechanism layer + two-panel readout |
| `scripts/news_dejavu.py` | narrative layer + certificates |
| `data/processed/episode_embeddings_single.parquet` | ChronoBERT signatures (USE THIS ONE) |
| `data/processed/dejavu_ots.parquet` | off-the-shelf encoder signatures |
| `data/processed/feature_panel.parquet` | 21-feature panel, 2013-2026 |
| `reports/encoder_test/REGISTRATION.md` | frozen predictions + outcome |

### House rules

- An experiment is created **before** it runs, with hypothesis, prediction and method frozen.
- An implausible number is an instrument bug until proven otherwise — including a *good* one.
- A null is a claim about the world only once the instrument is shown able to see the effect.
  A `refuted` verdict without a power statement is rejected.
- Report nulls. A well-evidenced negative is a deliverable.
