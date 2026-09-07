# The example page for 2020-10-30: nine items, registered before rendering

Committed **before** `scripts/poc_e9_page.py` exists. Every number the page prints must resolve to
a stored artifact named here; anything that does not is a hand-typed number and the numeral canary
fails the build on it.

Tim's decisions of 2026-09-04 govern three things and are carried verbatim:

- **The tail line is the parameter-free volatility-scaled quantile**, today's volatility times the
  5% quantile of past scaled returns, with **no conditioning state**.
- **The catalyst-window increase line is dropped.** The calendar state appears as page *fields* —
  catalyst dates and window catalyst days — never as a model input.
- **Weighting is value-weighted within each leg.** Where a number in the package is
  equal-weighted, the page says so beside it. The concentration reading is **not** printed as a
  discovered exposure.

## The nine items

| # | item | source artifact | reproduce |
|---|---|---|---|
| 1 | Horizon and event definition | `pipeline/contract.py` constants | frozen; not computed here |
| 2 | Severity: 10-day 5% VaR and ES, scaled line, against the unconditional | `reports/poc/e8_tail_final.json` | `poc_e8_tail_final.py` |
| 3 | Calibration of that line: coverage on the reconstructed book | `reports/poc/e5_book_tail_summary.json` | `poc_e5_book_tail.py` |
| 4 | What the book is: leg sizes, effective N and weight norm per leg, value-weighted | `reports/poc/e8_concentration_by_leg.csv` | `poc_e8_tail_final.py` |
| 5 | What the reader's existing screens say | `reports/gate2/nov2020_*.json` | `gate2_nov2020.py` |
| 6 | The theme, with verbatim quotes and the date each became public | see **the theme, two ways** below | `poc_e7_theme.py` or `gate2_extract.py` |
| 7 | Coverage and limits, including how much of the set the theme rests on | `reports/poc/e7_summary.json` or `nov2020_coverage.json` | as above |
| 8 | Catalyst dates and window catalyst days — **fields, not a model input** | `data/raw/fed/fomc_statement_dates.parquet`, 8-K item 2.02 | `poc_e5_book_tail.py` |
| 9 | What would change this reading | named by the filings, not by a model | — |

## The theme, two ways, decided before either is seen

Item 6 renders from **whichever of two results exists**, and the rule is fixed here so the choice
cannot be made after seeing which is more flattering.

- **If exp-091 fires on the 2020-11 episode with a vaccine-labelled cluster** — a theme above the
  95th percentile of the random books' maximum, whose label matches the expected label registered
  in exp-091 — the page prints the pipeline's theme, and the v1 8-K survival-condition result
  becomes confirming evidence beside it.
- **If it does not**, the page prints the v1 8-K result — vaccine timing; UAL, CCL and AMC with
  their accepted-at dates; 17.0% of loser-leg filings against 7.2% at matched controls — and
  **the pipeline's null is printed beside it**, with its coverage reason, not omitted.

Either way the page states which of the two produced the theme it shows.

## What is described-only and will not appear as a number

Positioning, the vulnerable-name lists, the analogue engine and the catalyst agent. The page names
them as designed-and-not-built rather than printing a placeholder.

## The check

Every numeral in the page's prose is matched against a registered field by
`report/numerals.py`, and the build fails on the first that matches neither a field nor a declared
definitional constant. Verbatim spans are excised before that scan, because a number inside a
filer's quote belongs to the filer. The provenance map is written beside the page so a reader can
resolve any number to its artifact and its command.

---

## Amendment, 2026-09-04: the analogue panel as a live-only item

Added after the nine items above were registered, on Tim's decision, and recorded here as a dated
amendment rather than by editing the registered text.

**A numbering note for the design session.** In the registration above, item 7 is *coverage and
limits*, which is load-bearing and stays. The analogue panel is therefore added as a **tenth**
item rather than replacing item 7. If the intent was to replace, say so and I will move it.

**What it prints.** The five nearest dates from the deterministic distance rule, and beside them
the panel's verdicts from the exp-082 run on the 2020-10-31 formation: the advocate's and
dissent's surviving claims, the field that decided each verdict, the spans cited, and **the count
of claims the gate dropped**.

**Sourcing.** `reports/analogue_rag/panel_2020-10-31.json`, rendered as stored. The panel is
**not** re-run — a fresh sample would be a different result on a page that is meant to reproduce.

**Two labels the item must carry, because both are limitations rather than findings.**

1. **Live-only, run on a historical date.** The panel is the design's one agentic step and cannot
   be backtested: it is included because a production system would use it, not because its
   contribution here is measured.
2. **The distance rule is the v1 engine's.** The caveat as relayed said its state space is "the
   published factor's five price features". That is not what the artifact shows and the page does
   not say it. `data/processed/state_vector_spec.json` defines **nine** blocks — tilt, comovement,
   dispersion, valuation, volregime, mechanism, shortside, options, textmacro — and the stored run's
   `block_closeness` shows the query at this date resolved on **three** of them: dispersion,
   volatility regime and mechanism. The page therefore says the distance used three blocks of the
   v1 engine's state space rather than the design's exposures, which is the accurate form of the
   same limitation. Flagged to the design session.

**The gate is the one exp-082 established** and is unchanged: a mechanism claim is kept only with
a verbatim span from a condition record for that date; a date with no condition record is ruled
*cannot be ruled on*; dropped claims are counted rather than silently discarded.

---

## Amendment, 2026-09-07: nine items, and the panel is not a tenth

Tim's decision, relayed by the drafting session: **the page is nine items, with the analogue
panel's verdicts rendering inside the analogues item rather than standing as a tenth.** This
supersedes the numbering note in the 2026-09-04 amendment above, which asked the question and
offered to move the item; the answer is that the intent was not a tenth item. Recorded as a dated
amendment rather than by editing either registered text, as before.

**The two documents number differently, and the decision is in the spec's numbering.** In the
draft-17 spec's page rule, item 7 is *the five analogues*, and that is the item the panel renders
inside. In the table above, item 7 is *coverage and limits*, and this registration never carried
an analogues item at all — the analogue engine was listed under *described-only and will not
appear as a number*. That absence is precisely why the 2026-09-04 amendment had to add the panel
as a tenth item, and it is why "the panel goes in item 7" cannot be transcribed into this
document unchanged without asserting something false about item 7 here.

**In this registration's numbering, therefore:** the analogues and the panel's verdicts are ONE
item, registered by that amendment and now counted within the nine rather than beyond them. Item
7 here remains coverage and limits. The item that moves is the analogue item, from tenth to
folded-in; nothing about what it prints, its sourcing or its two required labels changes.

**The page text does not change.** `reports/poc/page_2020-10-30.txt` already renders the five
dates and the panel's ruling as a single section, `WHEN DID THIS LAST HAPPEN`, carrying both
required labels — live-only and run on a historical date, and the distance rule as the v1
engine's three blocks. This amendment records a decision about how the page is *described*, not
about what it prints, and no artifact, numeral or provenance entry is affected.

**Provenance note.** This amendment lands AFTER the `poc-freeze` tag. The tag is not moved: no
result file, artifact or page byte changes here, so files cited at the tag remain the files at
the tag. A memo citing this registration for the item count must cite this commit rather than the
tag.
