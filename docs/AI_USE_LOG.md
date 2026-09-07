# How AI was used in this package, and where it was not

The brief asks where AI adds value and where conventional methods are more appropriate. This
file answers that for the three artifacts in this package, one at a time, with the outcome of
each use rather than the intention behind it. A measured negative counts, and two of the three
answers below are negative.

Nothing here is a claim about capability in general. It is a record of what was run, on what,
and how the output was checked.

---

## The tail model: no model call at all

The severity line is today's trailing volatility multiplied by the 5% quantile of past
volatility-scaled returns. There is no language model in it, nothing is fitted, and there are no
parameters to tune beyond the horizon and the quantile, both fixed in advance.

This is the clearest finding in the package about where AI belongs. The quantity a PM needs
first — how bad is the ten-day tail, given how volatile the book is right now — is answered
better by an arithmetic estimator over three decades of returns than by anything with weights in
it. A fitted linear quantile regression was built and tested against this estimator and **tied**
it, so the simpler one ships.

**How it was checked.** Out-of-sample coverage on the reconstructed book, scored only on
formations whose ten-day outcome had already realised: the unconditional line **fails** its
coverage test and the scaled line **passes**. A shuffled-volatility control, which destroys the
conditioning while preserving every other feature of the data, recovers the result **0.0%** of
the time. The pinball-loss ratio is reported with a block-bootstrap interval that **includes
one**, and that is printed with the number rather than around it.

**Where AI was declined.** Three conditioning variables were tested — a bear-market state, a
calendar state built from scheduled catalysts, and a volatility tercile — and **none is used**,
because none improved the forecast. The calendar state appears on the page as a printed field so
a reader can see what is scheduled, and is explicitly labelled as not an input.

---

## Reading filings: a language model, with its error rate measured

One place in this package asks a language model to read a document. Given an 8-K, it answers
whether the filer states an **external survival condition** — something outside the company's
control that its recovery depends on — and, if so, returns the verbatim sentence.

**The model.** Claude, called with a constrained output schema, a mandatory decline path, and two
gates the output must pass before it counts: the quoted span must appear verbatim in the source
filing, and every populated field must cite one. Filings are gated on their **acceptance
timestamp**, so nothing is read before it was public.

**How the error rate was established, and why not by hand.** Precision and recall against human
labels was registered and then **blocked**: the author of a system cannot be its annotator
without the labels inheriting the system's assumptions, and no independent annotator was
available. Ninety labels are reserved for human annotation and no model has touched them.

Instead the ground truth was **constructed**. Real survival-condition sentences were transplanted
into filings the model had already declined — same sector, different company, different date —
alongside a control arm of real boilerplate sentences and an unplanted re-read of the same
filings. Against that:

| | model | frozen keyword rule |
|---|---|---|
| recall on planted conditions | 19.2% | 54.2% |
| quotes the planted span verbatim | 17.5% | n/a |
| specificity on planted boilerplate | **97.5%** | 62.5% |

*Rendered at build time from `reports/poc/e4_summary.json` — the registered arm, whose plants are
drawn from the same sector as the host filing and which carries the unplanted-reread control. A
superseded second design, with incongruous plants, sits beside it in
`e4_summary_incongruous.json`; its numbers are not quoted here and the two must not be mixed.*

**The honest reading.** The keyword rule finds 2.8 times as many planted conditions. The
model almost never fires on planted boilerplate, where the rule fires on more than a third of
it. My registered
prediction was that the model would win on recall; it did not, and the row stands. Recall here is
depressed by the transplant being out of context, which is a property of the test and is stated
rather than corrected for.

A separate head-to-head between the model and the frozen keyword rule, judged by an outcome
neither chose, came back **inconclusive**: neither reader separates episode dates from control
dates at this sample size.

---

## The theme pipeline: an encoder and a clusterer, reported as instrument-limited

The design's central claim is that the companies carrying a book's exposure state, in their
filings, one shared risk theme that randomly chosen companies do not. Testing it needs an
embedding model and a clusterer, and both were built and run on twelve dates.

**The models.** `sentence-transformers/all-MiniLM-L6-v2` for sentence embeddings; HDBSCAN over a
UMAP reduction for clustering; c-TF-IDF for cluster labels; 200 capitalisation-and-sector-matched
random books per date as the placebo. The classifier the design specifies over filing text was
**not trained**, so every risk statement here comes from a frozen keyword rule, and that is
recorded rather than glossed.

**The outcome is negative and the reason is coverage, not the method.** A theme rose above the
placebo on 2 of 8 episode dates against a registered prediction of at least five. More decisive:
the control side is **empty** — of three calm windows one does not build and the two that do have
essentially no company with point-in-time filing text, so "nothing rose on the calm dates" is
arithmetic rather than evidence. The comparison could not have gone the other way.

**And the clusterer is not stable on the one date this package shows.** Three independent
64-company draws from that date's own filings, at identical parameters and the same seed,
returned 2, 2 and 26 clusters. A declared degeneracy flag fires on that date and on no other.

So the page prints the filing-level reading instead, by a fallback rule fixed **before either
result was seen**, and the pipeline's null is printed beside it rather than omitted.

---

## The one agentic step, and why it carries no number

The page shows the five nearest historical dates by a deterministic distance rule, and beside
them the verdicts of a three-role panel — an advocate, a dissent and an adjudicator — arguing
over whether those dates are apt comparisons. A gate keeps a mechanism claim only when it comes
with a verbatim span from a condition record for that date, and **counts what it drops**.

On the date shown, retrieval alone would have reassured: four of the five nearest dates were
followed by gains. The adjudicator ruled the match **not apt** and the mechanism **impossible to
rule on**, because all five predate the condition corpus — so what those episodes were priced on
is unknown rather than different.

This step is **live-only**. It is the one part of the design that cannot be backtested, so its
contribution is shown and not measured, and the page says so on the page.

---

## In the process, not in the system

This package was written with AI assistance throughout, and the working method mattered more than
any single call. Two rules did the work.

**An experiment is registered before it runs**, with its hypothesis, prediction and method frozen
at that point, and a power statement saying what it could and could not have seen. `REGISTER.md`
holds those records for everything this package claims. Of the 13 that ran,
6 did not support
their hypothesis.

**An implausible number is an instrument bug until proven otherwise, including a flattering one.**
That presumption has been correct every time it was applied. 21 instrument faults found
during this build are recorded in `REGISTER.md`, each one a case where the code ran clean and the
number was wrong. Several changed a headline figure after it had already been written down:

- a market series divided by a hundred twice, which made a conditional model report exactly its
  own unconditional baseline — a result that looked like a finding and was an arithmetic identity;
- a published weight column that rounds to zero for 1,184 of 2,683 names, so dropping the zeros
  silently marked the book on 55 names instead of 235;
- a page that cited a calibration computed on its own future, and then described it wrongly;
- a numeral highlighter that corrupted the page it was meant to annotate while the checker
  reported PASS, because the checker read the source and the reader saw the output.

The last is why the page's numerals are verified by a build-time check rather than by review: the
page is composed from registered fields, and a numeral in its prose that matches no computed field
**fails the build**. That check exists because the page it now guards was once hand-typed. The
numbers were right at the time, and nothing would have noticed when one stopped being right.

---

## What was not used

No employer data, no proprietary or non-public data, and no research or commentary obtained
through a current or former employer. Every source is free and public, and each carries the
observation and availability timestamps that gate it.
