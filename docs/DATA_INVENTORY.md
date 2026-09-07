# Data inventory

Verified 2026-09-02 by querying the registry, the disk, BigQuery and GCS directly rather than
from the source registry alone. Where the registry and the store disagree, both are shown.

**Totals: 8.5 GB local disk · 59.2 GiB Google Cloud Storage · ~216 GB BigQuery (14 tables).**

---

## 1. Market and factor returns

| Source | Span | Grain | Size / rows | Notes |
|---|---|---|---|---|
| `src-french` Fama-French daily factors | 1926-07-01 → 2026-06-30 | daily | 43 MB | The target. WML plus market, size, value. Restates silently with no version numbers. |
| `src-fred` Federal Reserve economic data | 1990-01-01 → 2026-08-21 | daily | 2.3 MB | DGS10 and the macro library used for falsifier scoring. |
| `src-cboe-index` volatility & correlation indices | 1990-01-02 → 2026-08-19 | daily | 2.1 MB | Five series with different start dates: 1990, 2004, 2006, 2009, 2011. Series that stop upstream must not read as calm. |
| `src-cboe-chains` option chain snapshots | 2026-08-20 → 2026-08-20 | daily | 82 MB | **One day.** Capture began after the design tier closed; no history exists and none can be reconstructed. |
| `src-implied-moments` Vilkov/Rehman OSF panel | 2017-03-22 → 2023-12-29 | ticker-day | 84 KB | 1,705 rows, MTUM model-free implied moments. |

## 2. Holdings, positioning and flow

| Source | Span | Grain | Size / rows | Notes |
|---|---|---|---|---|
| `src-iwv` iShares Russell 3000 holdings | 2006-09-29 → 2013-03-31 month-end; 2013-04-01 → 2026-08-19 daily | month-end, then daily | 1.5 GB | 3,341 month-end rows then daily. **Two upstream gaps: 2014-12→2015-01 and 2017-01→2017-07.** Source of the [C] and [X] books. |
| `src-13f` SEC institutional holdings | 2001-01-01 → 2026-06-30 | quarterly per filer | 2.1 GB | 54 quarters as bulk parquet. Long-only, 45-day lag, manager level — cannot see gross or leverage. Observes 13 of 15 episodes. |
| `src-mtum` momentum ETF holdings | 2013-04-22 → 2026-08-19 | daily | in `ishares` | 3,218 rows. Shares outstanding differenced gives creation/redemption flow. |
| `src-finra` daily short volume | 2018-08-01 → 2026-08-20 | daily | 216 MB | 2,024 rows. ~Half of reported volume is short by construction, so only changes carry information. |
| `src-short-interest` FINRA consolidated | 2018-01-12 → 2026-08-15 | twice monthly per security | 214 MB | ~16,200 securities per settlement date, ~11,700 usable. Carries `settlementDate` and **no publication date**; dissemination is ~8 business days later. |
| `src-regsho` threshold securities lists | 2018-08-01 → 2026-08-19 | daily | 7.9 MB | 2,023 rows. **Tested and found unusable** for a large-cap book — 34–35 names market-wide. Kept as a reported negative. |
| `src-finra-margin` aggregate margin debt | 1997-01-31 → 2026-07-31 | month-end | 20 KB | 355 rows, **no gaps**. Acquired 2026-09-02. The only free leverage series reaching 2007; covers 15 of 15 episodes. Market-level, not factor-level. |
| `src-ftd` SEC fails-to-deliver | 2009-07-01 → 2022-12-30 | settlement-date per security | 408 MB raw → 252 MB parquet | **18.2M rows, 49,614 symbols, 3,305 settlement dates, 162 of 162 months complete.** Acquired 2026-09-02. |
| `src-nport` Form N-PORT | 2019-10-01 → 2022-12-31 registered | quarterly file of monthly holdings | 325 MB (1 of 13 quarters) | Only free source with **short positions, borrowings and gross leverage**. 2020q1 inspected: 11,259 usable funds, asset-weighted gross leverage 1.0438, $150.7bn explicit borrowings, $1,079bn quarterly redemptions. Registered funds only — no hedge funds. |
| `src-cot` CFTC Commitments of Traders | 2015 → present | weekly | 5.3 MB | Futures positioning by trader category. |

## 3. Text corpora

| Source | Span | Grain | Size / rows | Notes |
|---|---|---|---|---|
| `src-edgar-8k` 8-K bodies | 2018-01-02 → 2026-08-26 | filing-level | 638 MB | **52,527 indexed filings, 54,488 gzipped bodies, 1,339 tickers**, second-stamped `accepted_at`. The corpus behind the tripwire. |
| `src-edgar` company filings | 2001-01-01 → 2026-08-21 | filing-level | 378 MB | 2,519 rows plus 142 MB `edgar_panel` and ~26 MB of extracted sections. |
| `src-gdelt` global news index | registry says 2017-01-01 → 2026-08-21 | article-level | 140 MB local | **Registry disagrees with the store: BigQuery partitions begin 2015-02.** Bulk lives in BigQuery, not on disk. |
| CC-NEWS filtered (GCS) | **2020-10 → 2021-01 and 2024-12 → 2025-04** | WARC | **44.26 GiB, 4,067 files** | Two discontinuous blocks. **Roughly half sits inside the sealed era (2023-01 onward).** |
| `src-epu` uncertainty indices | 1900-01-31 → 2026-07-31 | monthly | 2.7 MB | Baker-Bloom-Davis EPU and Caldara-Iacoviello GPR. **The pre-1985 portion was constructed retrospectively**, so its term list was chosen by people who knew what followed. |
| `src-bis` BIS quarterly review | 1996-03-01 → 2026-06-30 | quarterly | in `edgar`/`official` | Tested as early warning and **refuted** — word frequencies track the editorial calendar. |
| `src-fomc` Fed meeting calendar | 1996-01-01 → 2026-12-31 | event-level | 776 KB | 255 events. Four different address schemes across four website eras. |
| `src-calls-curated` earnings-call transcripts | 2019-10-01 → 2022-12-31 | call-level | **0 rows** | Registered as a data rung, **not acquired**. |
| `src-external` Loughran-McDonald dictionary | — | word list | 8.5 MB | Finance sentiment lexicon. |

## 4. Derived panels (`data/processed`)

| Artifact | Size | What it is |
|---|---|---|
| `ftd.parquet` | 252 MB | Parsed fails-to-deliver, 18.2M rows |
| `panel_v2/` (149 files) | 90 MB | Attention extracts pulled down from BigQuery |
| `panel/` | 80 MB | First-generation attention panel |
| `episode_embeddings*.parquet` | 12.4 MB | Episode text embeddings, single-encoder and multi |
| `dejavu_ots.parquet` / `dejavu_chronobert` / `dejavu_all` | 4.7 MB | Retrieval-arm comparison outputs |
| `two_channel_state.parquet` | 2.7 MB | The two-arm state vector |
| `exp045/` | 1.8 MB | Cap-matched placebo book panel — source of the 104-of-128 figure |
| `leg_members.pkl` (+2 backups) | 1.5 MB | **151 month-ends, 2014-01-31 → 2026-07-31**, ~271 names per leg. The binding constraint on the FTD test. |
| `state_vector` / `state_series` / `feature_panel` | 0.7 MB | Retrieval keys and channel states |
| `finra_margin.parquet` | small | 355 months with `available_at` stamps |

## 5. Cloud

**BigQuery — dataset `momentum`, 14 tables, ~216 GB, partitioned monthly 2015-02 → 2026-08 (139 partitions).**

| Table | Rows | Size |
|---|---|---|
| `gkg_matched` | **122,266,705** | **211.55 GB** |
| `slug_first_seen` | 47,862,289 | 3.50 GB |
| `gkg_matched_companion` | 10,235,277 | — |
| `comention_daily` / `comention_weighted` | 9,824,570 each | — |
| `comention_weighted_v2` | 8,353,415 | 0.42 GB |
| `attention_daily` | 3,827,695 | — |
| `attention_daily_v2` | 3,305,563 | 0.15 GB |
| `book_membership` | 73,587 | — |
| `source_stats` | 22,163 | — |
| `book_names` | 8,164 | — |
| `spam_sources`, `matched_test2`, `matched_test_202011` | — | scratch |

**GCS — `gs://unstructured-momentum-data`, 59.21 GiB.** `ccnews_filtered/` 44.26 GiB (4,067 WARCs); `repo_raw_backup/` 14.55 GiB (13F crawl archive and raw mirrors, deleted from local disk after a verified count-matched upload of 170,818 files); `extracts/` 407.7 MiB (attention parquets).

## 6. Registered but NOT acquired

| Source | Span it would cover | Why it matters |
|---|---|---|
| `src-optionmetrics` | 1996 → 2026 daily | Whether crash protection on the book's own names richens before a reversal. Untestable now: chain capture is one day old. |
| `src-borrow` securities-lending cost | 2006 → 2026 daily | Whether loser-leg skew signals a squeeze or merely prices borrow — **mechanically linked and inseparable with free data**. |
| `src-news-archive` licensed full text | 1990 → 2026 | Would lift text coverage from ~5% of the archive. |
| `src-calls-archive` licensed transcripts | 2004 → 2026 | The second rung of the emergence experiment. |

## 7. Discrepancies found while compiling this

1. **`src-gdelt` is registered as starting 2017-01-01; BigQuery partitions start 2015-02.** Two extra years exist in the store than the registry claims.
2. **Roughly half the CC-NEWS corpus (2024-12 → 2025-04) is inside the sealed era** and must not be read for design work.
3. `src-ftd` and `src-nport` carry `rows: 0` in the registry — placeholders written at registration and not yet backfilled with measured counts.
