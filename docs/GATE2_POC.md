# Proof of concept — what the loser leg is collectively priced on

**Formation date 2020-10-31.** The book carried into the 9 November 2020 vaccine reversal, the
largest single-day momentum loss of the modern era. Everything below is read from filings
accepted **before** the formation date.

The claim under test is §1.7's: returns and holdings identify the loser leg's exposure and its
concentration, but they cannot say what condition three hundred distressed businesses *share*.
Filings can. This is the seat text is asked to earn.

## The pipeline

Point-in-time loser leg → 8-K filings gated on **regulator acceptance time** → schema-constrained
extraction of the external condition on which survival is stated to depend → verbatim quote
checked against source → local embedding → clustering → entropy. The winner leg runs identically
as the contrast.

Four disciplines are enforced in code, not asserted:

- **The schema has no numeric field**, so the model cannot return a probability or a severity.
- **A mandatory decline path.** "No stated condition" is a valid answer and is expected to be
  the common one.
- **A quote gate.** Every record carries a verbatim span; records whose quote cannot be found in
  the source are **dropped, not repaired**.
- **Acceptance-time gating**, so nothing read could postdate the formation date.

## Result

| | loser leg | winner leg |
|---|---|---|
| filings read | 112 | 206 |
| extraction errors | 0 | 0 |
| **states a survival condition** | **19 (17.0%)** | **11 (5.3%)** |
| declined | 92 (82.1%) | 194 (94.2%) |
| quote grounded | 19 of 20 (95.0%) | 11 of 12 (91.7%) |
| names with a condition | 9 | 10 |
| condition clusters | 4 | 7 |
| **normalised entropy** | **0.523** | **0.819** |
| **largest cluster** | **55.6% of names** | **20.0%** |

**The entropy figures do not survive their own robustness check, and the check is reported here
rather than buried.** Normalised entropy is compared at a fixed cosine threshold, which gives the
loser leg 4 clusters and the winner leg 7 — so the gap is partly the *cluster count*, not the
concentration. Forcing the same k on both legs collapses it:

| | threshold 0.55 | fixed k=3 | fixed k=4 | fixed k=5 |
|---|---|---|---|---|
| entropy gap (winner − loser) | **+0.296** | +0.004 | +0.048 | +0.004 |

**At equal k the difference is essentially zero.** The clustering step is a free parameter and it
was carrying the result. This is precisely the failure mode a reviewer named in advance: a
concentration claim is unattributable between a genuinely dispersed leg and a clusterer that split
one condition into three. **The entropy statistic is withdrawn as evidence.**

What survives the check, and it is the part that never needed the clusterer:

- **The rate of stating a survival condition: 19 of 112 against 11 of 206 — 17.0% versus 5.3%, a
  3.2× difference computed from counts alone.** No embedding, no threshold, no free parameter.
- **The conditions themselves, read directly.** Loser leg: cruise operations resuming, theatre
  attendance recovering, air travel returning, retail footfall, steelmaking demand — one condition
  in five different vocabularies. Winner leg: FDA protocol agreements, merger consents, a CMS
  payment rule, letters of credit — five conditions with nothing in common.
- **The UAL finding**, which involves no clustering at all.

So the loser leg is **3.2× more likely** to state a survival condition, and its conditions are
concentrated **by inspection and by rate, not by the entropy number I first reported**. Five of nine loser names sit in one cluster: the resumption
of normal demand after COVID-19 — cruise operations, theatre attendance, air travel, retail
footfall, steelmaking. Winner-leg conditions are idiosyncratic and name-specific: FDA protocol
agreements, merger consents, a CMS payment rule, letters of credit.

## The finding the design predicted

United Airlines, in a filing accepted **2020-09-09**, 52 days before the vaccine announcement:

> "expects demand to remain suppressed and plateau at levels of around 50%, relative to 2019
> levels, until a widely accepted treatment and/or vaccine for [COVID-19]"

and again on **2020-09-18**:

> "it is unlikely the aviation industry will see significant return of passengers until a vaccine
> is widely available to the public and international mar[kets reopen]"

The loser leg's shared condition was **named in public filings available at formation, and the
resolving event was named with it**. That is Route 1 of Figure 1 end to end: the short half
collects businesses priced on one external condition, and when that condition resolves they
reprice together within a session. No returns-based measure recovers this, because the condition
is not in the returns — it is in the documents.

**One detail sharpens it.** Novavax appears on the *winner* leg citing "future developments of
COVID-19 pandemic". The same variable, opposite sign: the losers need the pandemic to end, the
vaccine maker does not. The sort put both in the book and conferred opposite exposures to one
condition.

## What this is not

**n is 9 and 10 names.** This is a worked example on one date, not a statistic. **No null is
computed, because none computed at this sample would mean anything**, and the entropy figures
are reported as descriptive. Coverage is the binding limit and is stated rather than softened:
36 of 263 loser-leg names have any 8-K in the window, because the corpus covers 1,579 tickers
against a universe near 2,900 and is biased toward larger names — the loser leg is small and
distressed, so it is the half the corpus covers worst.

**The entropy time series across 2014–2022 that the design specifies is [GAP-03], and remains
so.** What would close it is corpus extension to the uncovered names, which is free from EDGAR
and priced in the production path. The limitation runs against the design's own claim, not for
it: better coverage can only add names to the loser leg's concentration test.

## Reproduce

```
python scripts/gate2_nov2020.py     # constituents, filings, coverage
python scripts/gate2_extract.py     # extraction, loser leg   (needs ANTHROPIC_API_KEY)
python scripts/gate2_extract.py winner
python scripts/gate2_entropy.py     # clustering and entropy  (local embedding, no model call)
```

---

## The analogue panel: retrieval, and a dissent that overrules it

Retrieval returns the nearest historical states **by construction**. It has no way to say the
nearest state is not a useful one. On this date that distinction is the whole question.

The five nearest states to 2020-10-31, with what followed each:

| rank | date | distance | forward 21-day WML |
|---|---|---|---|
| 1 | 2009-08-31 | 0.039 | −4.03% |
| 2 | 2009-05-31 | 0.070 | **+3.91%** |
| 3 | 2009-01-31 | 0.085 | **+11.24%** |
| 4 | 2008-10-31 | 0.088 | **+6.33%** |
| 5 | 2000-05-31 | 0.110 | **+16.09%** |

**Four of five were followed by gains. The realised outcome was −9.56% over ten days.** Read
naively, the analogue set pointed the wrong way.

A three-role panel runs over that record: a proponent argues the top match is apt, a dissent
argues where it fails, an adjudicator decides. **Verdict: not apt, failing on
`query_shared_condition`** — the query's loser leg is priced on post-pandemic demand recovery
resolving on a discrete event, while every retrieved state is a credit-crisis or dot-com
recovery. Same volatility regime, different mechanism. The panel overruled its own retrieval,
and the outcome says it was right to.

**Two containment rules, both enforced in code.**

- **Field citation.** Every claim must name a field present in the record; claims citing absent
  fields are dropped, not repaired. On this run **8 of 8 claims passed, so the gate did not
  fire** — an unfired gate demonstrates nothing, so it was tested directly with six planted
  claims, three real and three invented. It kept the three real fields and dropped
  `implied_correlation`, `vix_percentile` and `query_forward_return`.
- **No outcome leakage.** The agents see the analogues' forward returns, which were knowable at
  the query date, and never the query date's own. That rule is enforced by the same gate: the
  leaked field is rejected *by absence from the record*, which is what makes it a mechanism
  rather than an instruction.

**What this demonstrates about where AI adds value.** The retrieval is arithmetic and cannot
know its own limits. The judgement that a near-neighbour in state space is a distant neighbour
in mechanism is exactly the reading a model can make and a distance metric cannot — and it is
checkable, because every claim points at a number.

---

## The temporal control, and what it does to the claim

The 17.0% rate was uninterpretable on its own: it could be a permanent property of distressed
names rather than anything about November 2020. The winner-leg comparison controls for *leg*.
Nothing controlled for *date*. So the extraction was run on two ordinary formation months —
same construction, same leg, at least 120 days from any registered episode.

| | filings stating a condition | rate |
|---|---|---|
| **2020-10-31, target** | 19 of 112 | **17.0%** |
| 2019-04-30, control | 8 of 113 | 7.1% |
| 2021-10-31, control | 7 of 110 | 6.4% |
| **controls pooled** | 15 of 223 | **6.7%** |
| winner leg, target date | 11 of 206 | 5.3% |

**The two controls agree to within 0.7 points, which is what makes the baseline credible.** The
loser leg's ordinary rate (6.7%) sits close to the winner leg's (5.3%), so the leg contrast at
the target date is a property of *that month*, not a standing feature of distressed names.

**And the significance does not survive the correct unit.** By filing the ratio is 2.5× at
Fisher p = 0.0064. But filings cluster within filers — one company contributes five of the
nineteen — so the unit is the name, not the filing. By name it is **25.0% against 13.3%, a
ratio of 1.9× at p = 0.18.** At 36 names against 83 the difference is **not significant**, and
the by-filing p is inflated by that clustering.

**What this changes.** The headline moves from uninterpretable to *interpretable and
underpowered*, which is a real improvement and an honest ceiling. The elevation is roughly
two-fold, in the predicted direction, on a baseline established from two independent months
that agree — and the sample cannot establish it at conventional significance. That is the
finding, stated as it is.

The claim that does not depend on any of this is the instrument claim, and it is the one to
lead with: **the sector screen read ten sectors and diversified while the filings read one
bet.** That comparison needs no rate, no baseline and no p-value.

## A limit of the field gate, stated

The analogue panel's gate checks that a cited field **exists in the record**, not that the
claim is **supported by** it. In the live run the dissent cited `date` and `block_closeness`
while making claims about fiscal stimulus and circuit breakers — real fields, claims reaching
well past them into model world-knowledge. The gate passed them because the field was real.
This is the same weakness as a quote gate that checks presence rather than entailment, and it
is a bound on the containment rather than a defeat of it: the numbers on the page come from
deterministic code, and the panel emits a verdict, never a quantity.

