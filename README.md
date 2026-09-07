# Reversal risk in US equity momentum

An AI-assisted system for identifying, explaining and monitoring tail risk in the US equity
momentum factor. This repository is the proof-of-concept package: the code, the data slice it
reads, the committed artifacts every page is composed from, and the register that records what
was claimed before each test ran. It is the proof of concept and only that: the working
repository it was built in holds retired phases that are deliberately not here.

The accompanying memo is the argument. This is the evidence, and every number in the memo has
a command here that regenerates it.

---

## The proof of concept: three artifacts

The proof of concept is deliberately three things, not a tour of everything in the register.
Each is a file you can open and a command that rebuilds it.

### 1. The tail model, with its controls and its nulls

A parameter-free volatility-scaled quantile: today's volatility times the 5% quantile of past
scaled returns. Nothing is fitted. Scored out of sample on the reconstructed book over
**81 formation months, 2015-09-30 to 2022-12-31**.

| | |
|---|---|
| Unconditional 5% VaR | **9 breaches of 81**, 11.11%, Kupiec **p = 0.028** — *fails* |
| Volatility-scaled | **6 of 81**, 7.41%, Kupiec **p = 0.352** — *passes* |
| Pinball vs unconditional | **0.890**, 90% block-bootstrap CI **[0.752, 1.061]** |

The interval includes one, and that is printed with the number rather than around it: scaling
earns its place on coverage, not on loss. Three conditioning variables were tested — a bear
state, a calendar state, and a volatility tercile — and **none is used**, because none improved
the forecast. A shuffled-volatility control recovers the result **0.0%** of the time, and a
fitted linear quantile regression ties the parameter-free estimator, which is why the simpler
one ships.

`scripts/poc_e1_tail.py`, `poc_e3_estimators.py`, `poc_e5_book_tail.py`, `poc_e8_tail_final.py`.

### 2. The theme on 2020-10-30, with the pipeline's outcome beside it

What the page prints is the **8-K survival-condition reading**: of the loser leg's filings,
those stating an external survival condition name one shared thing — the resumption of normal
demand after COVID-19 — with verbatim quotes and acceptance timestamps. That reading is the
**fallback**, and the rule choosing it was fixed before either result was seen.

The design's own theme pipeline is the thing that was supposed to supply it, and it is reported
beside the fallback rather than quietly replaced by it. Encoder
`sentence-transformers/all-MiniLM-L6-v2`; HDBSCAN over UMAP; 200 cell-matched random books per date as the placebo.
Its verdict is **instrument-limited**, and the reason is coverage, not the placebo:

- a theme rose above the placebo on **2 of 8** episode dates against a
  registered prediction of at least five;
- **3** of those dates carry an expected label written into the registration before
  the run, and **0** matched — but the dates where a theme rose are not among the
  dates that carry an expected label, so **no match was reachable**. The zero is a property of
  which dates were labelled, not evidence about the labels;
- the control side is **empty**: of 3 calm windows one does not build and the
  2 that do have essentially no component-set company with a point-in-time Item
  1A, so "nothing rose on the calm dates" is arithmetic rather than evidence;
- on the flagship date the clusterer is unstable — three 64-company draws from the same
  filings at identical parameters and seed returned 2, 2 and 26 clusters.

A negative result on the AI component, measured rather than asserted, is a deliverable here.

`scripts/poc_e7_theme.py`, `poc_e7_expected_labels.py`, `gate2_extract.py`.

### 3. The page

One PM-facing output for the 2020-10-30 formation: `reports/poc/page_2020-10-30.txt`, and
`page_2020-10-30.pdf` for the memo. Severity, whether the line is calibrated, what the book is,
what the reader's existing screens say, what the loser leg is priced on, the nearest historical
analogues with an adjudicator's verdict, the catalyst days in the window, and what is designed
and not built.

Every numeral on it is registered: **74 numerals against 55 computed
fields**, and a build that finds a hand-typed number stops. Where several registered fields
render the same string — both lines' breach rate at this date is 9.26% — the provenance says
**AMBIGUOUS** and lists the candidates instead of guessing; **9** occurrences do.

The page is written standing on 2020-10-30. What followed is printed under its own heading,
fenced as outside that information set: the book realised **-14.58%** over the next ten
sessions, of which 2020-11-09 alone was **-20.2%** — the loser leg **+16.0%**,
the short half taking the loss, which is the mechanism the page had named in advance. French's
published factor over the same window fell -9.56%. One formation is one observation and
it settles nothing; the coverage tests above are the evidence.

`scripts/poc_e9_page.py`, `poc_e10_realised.py`, `poc_page_pdf.py`.

### How AI was used, and where it was not

Every model call, its purpose and its outcome — including the calls that declined and the
component that came back instrument-limited — is in **[`docs/AI_USE_LOG.md`](docs/AI_USE_LOG.md)**.
The tail model contains no model call at all: it is a quantile of past scaled returns, and the
package says so rather than dressing it up.

---

## The result in nine numbers

| | |
|---|---|
| The severity model passes a sealed holdout on its registered design | **1 breach of 42**, rate 2.38%, coverage **p = 0.388** — on a period *wilder* than the design era, against a baseline that breached 2 |
| On the book a PM would hold, volatility scaling earns its place | unconditional **11.11%**, Kupiec **p = 0.028** (fails); scaled **7.41%**, **p = 0.352** (passes), over 81 formations |
| …and on the published factor the same test says the **opposite** | over 1047 formations the unconditional line passes (5.44%, p = 0.515) and the scaled line **fails** (7.16%, p = 0.0025). Both are reported; §1 argues why the book is the object of interest |
| The book's index-concentration reading is a **weighting artifact**, not a discovery | **z 18.9**, rank 1 of 8 equal-weighted; **z 1.5**, rank 7 of 8 value-weighted, which is what the design specifies. Same names, same date (2026-07-31), same 500 matched books. The same-rule null's typical monthly maximum is z 23.8 |
| The loser leg's shared condition is elevated before episodes, at design tier | **13.4%** against **7.1%** at matched controls, one-sided rank test **p = 0.0175** |
| …and that elevation does **not** reproduce on the holdout | **11.0%** against 10.4%, **p = 0.4686** |
| The three-route framework does not partition the record | **44.6%** of 193 decomposable episodes fit no route, against a registered band of 5–30% |
| The detectors do not recover the route before the event | **4 of 7**, p = 0.173; reported as instrument-limited, not as a refutation |
| The model seat measures its own reach | on 112 filings the decline path fires on **82.1%**; the quote gate grounds **95.0%** of what remains. Against constructed ground truth the model's advantage is **specificity** (97.5% against 62.5%), not recall (19.2% against 54.2%) |

**Five of those nine are failures, nulls or artifacts, and they are the reason to read the
rest.** Two of them contradict things this package would rather be true.

Every number in this table is rendered from a committed artifact at build time, not
transcribed — including the rank tests, which are recomputed from their panels on each build.
An earlier version of this README was hand-typed, and it went stale twice: once within a day of
a correction landing upstream, and once by quoting a precision figure from a retired experiment
whose labels the author had written himself.


---

## Run it

Two modes. Replay needs neither the network nor an API key, because every deterministic
artifact is committed.

```bash
uv sync
uv run python scripts/crash_inventory.py        # the census and the route partition
uv run python scripts/current_book.py           # what the sort is carrying now
uv run python scripts/gate2_page.py             # the PM page
```

The interactive read-out, with one live model seat:

```bash
export ANTHROPIC_API_KEY=...        # optional; every other panel works without it
uv run python -m uvicorn app.server:app --port 8000
```

Then open <http://localhost:8000>. Five panels: the book today, the crash inventory, the three
validation results, the example PM output, and a live seat that runs the extractor against a
filing you pick and shows the quote gate checking its answer against the source. The server
holds no analytics of its own — every number it serves was computed by a script with a reproduce
command, so the page cannot silently disagree with the engine.

`make replay` runs the deterministic chain end to end; `make live` adds the model stages;
`make test` runs the suite; `make register` re-derives `REGISTER.md` from `project/*.yaml`, so
the extract can be checked rather than taken on trust.

---

## What is here

```
src/unstructured_momentum/   the package: data loaders, factor construction, the null,
                             severity, the model seats and their gates
scripts/                     the 31 scripts the proof of concept runs, in README order
app/                         the read-out: a FastAPI server and one static page
REGISTER.md                  every experiment behind this package, with the hypothesis and
                             prediction frozen before it ran, and the faults found on the way
project/*.yaml               what REGISTER.md is generated from, so the extract is checkable
docs/                        the page's registration, the filing reading, the retrieval step,
                             the data inventory, and how AI was used
reports/                     every committed artifact the pages are composed from
data/                        the slice the proof of concept reads (see MANIFEST.json)
tests/                       the guards, including eight on data integrity
```

`MANIFEST.json` records exactly what was pruned from the working repository and why. The
working copy is 8.6 GB across twenty sources; this is 80 MB, and the prunes are row and column
filters, never transformations.

## The scripts, in the order the argument runs

| script | what it does | what it proves |
|---|---|---|
| `gate1_spine.py` | reproduces the factor against the French file, then the three baselines | the engine agrees with the published series before anything is claimed |
| `build_reference_panel.py` | the eight reference series, each estimated univariate | the exposure the sort confers, series by series |
| `gap01_family.py` | the family-maximum test across all eight | **saturated**: clears 108 of 108, and that is a finding about the null |
| `same_rule_null.py` | 12-1 against 6-1 and 24-13 sorts on the same dates | the null that discriminates, and the 43/57 split |
| `current_book.py` | the latest formation date, end to end | what the book is carrying today |
| `crash_inventory.py` | every episode the registered definitions admit, typed by rule | the partition claim, and its refutation |
| `sealed_run.py` | the one-shot holdout, both arms | the severity pass and the text non-reproduction |
| `gate2_nov2020.py` | constituents, filings, coverage — no model anywhere | the coverage gate, before any model is asked anything |
| `gate2_extract.py` | the one model seat that reads filings | the shared condition, with a quote gate and a decline path |
| `gate2_debate.py` | proponent, dissent, adjudicator over a fixed record | a panel that overruled its own retrieval, and was right |
| `gate2_page.py` | composes the PM page over fixed fields | the example output |

## How the model seats are constrained

Four risks, four rules, each of which has caught a real error in this project.

1. **They invent evidence.** Every extracted field carries a verbatim span, string-matched
   against the source before storage. An unmatched span drops the record and is counted. The
   extractor fabricated 7% of its supporting quotes before this gate was added.
2. **They drift on numbers.** Models produce records; deterministic code produces every
   statistic on the page; a tokeniser rejects any page where a numeral in the prose fails to
   match a computed field. Two model-written summaries inflated their own headline arithmetic,
   one by a factor of six.
3. **They recognise periods rather than measure them.** Given anonymised text with dates
   stripped, a model identified the exact year in 40 of 40 cases. A date-constrained reader runs
   beside the contemporary one and their agreement bounds how much of the text feature is
   recognition.
4. **They agree with each other.** Several instances of one base model share errors, so a panel
   of them produces the appearance of independent confirmation without the substance. Every seat
   is checked against a simpler deterministic twin printed beside it, and an idle seat is cut.

## The register is the point

**[`REGISTER.md`](REGISTER.md)** holds every experiment behind this package. Each was created
**before** it ran, with its hypothesis, prediction, method and both outcome sentences frozen at
that point, and a power statement saying what it could and could not have seen. A `refuted`
verdict is refused unless it carries that statement and a recovered planted effect.

That order is the whole claim. Of the 13 experiments that ran, 6
did not support their hypothesis, and those are only worth reading because the prediction each
one contradicts was written down first. The
selection is mechanical — an experiment is in the extract if its reproduce command names a script
that ships — and `make register` re-derives it, so nothing was kept for being flattering.

It also holds the instrument faults found during this build: every one a case where something
parsed cleanly and was wrong. Two of them sit in the *registration* rather than the code:

* **trp-84** — a sealed test deviated from its own registered sampling and reported a spurious
  failure. The instinct that caught it was disbelief at a failure; a flattering artifact of the
  same shape would have been much harder to catch.
* **trp-85** — a registered secondary statistic compared a label against the variable that
  defines it, so its AUC of 1.000 was arithmetic. Every code guard passed, because the fault was
  upstream of the code.

Both are in the register and neither is there by accident.

**An implausible number is an instrument bug until proven otherwise — including a flattering
one.** That presumption has been correct every time it was applied here.

## Limitations, stated once and plainly

* **Filing coverage is partial and skewed large.** 1,579 tickers against a universe near 2,900,
  and the micro-cap wreckage where the loser-leg mechanism concentrates is systematically absent.
  The measured contrasts therefore read as a floor on the true one, not a ceiling.
* **Crowding is inferred from returns, not measured.** Public data cannot see positioning before
  it moves prices. Two positioning nulls (margin debt, fails-to-deliver) are refuted with
  certified minimum detectable effects rather than quietly dropped.
* **No probability of a reversal is reported.** Three episodes in a century contain a crash day;
  any probability would carry an interval wider than itself. Severity, conditional on a state, is
  what the data supports.
* **The text arm's out-of-sample result is a null at n = 6 episodes.** It is reported as such,
  with its floor, and the design-tier result is kept and labelled design tier.

## Data

Free and public throughout: Ken French's daily factor and decile files, iShares IWV daily
holdings, FRED reference series, SEC EDGAR 8-K full text with acceptance timestamps, FINRA short
interest and Reg SHO volume, Cboe indices. No employer data and no proprietary code. Every input
carries the date it became public, and every feature is computed from that date and never from
the date the data describes.
