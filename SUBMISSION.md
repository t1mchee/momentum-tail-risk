# Submission index

Four deliverables were asked for. This file says where each one is and, where something is not
here, says so plainly rather than leaving a reader to discover it.

| # | Deliverable | Where | Status |
|---|---|---|---|
| 1 | **Code / PoC package + README** | this repository; start at [`README.md`](README.md) | complete |
| 2 | **Research memo (~6–10 pages)** | `memo/` | **not in this repository** — see below |
| 3 | **Example risk output** | [`reports/poc/page_2020-10-30.txt`](reports/poc/page_2020-10-30.txt), with [`page_2020-10-30.pdf`](reports/poc/page_2020-10-30.pdf) for print and its provenance file beside it; rendered clickable at `make dash` → *Example output* | complete |
| 4 | **15–20 minute presentation** | `presentation/` | **not in this repository** — see below |

## The two that are not here

They are drafted outside this repository and are sent alongside it. This repo is the evidence
they rest on: every figure either of them quotes has a command here that regenerates it, and the
`reports/` directory holds the artifact each one was computed from.

If you are reading this and `memo/` or `presentation/` is empty, they have not been added yet.
That is a statement about this repository's state, not about the work.

## The seven required elements

The memo is the argument and answers all seven in prose. This repository is where four of them
can be checked rather than taken on trust:

| Element | Where it is checkable here |
|---|---|
| 1 · Problem definition | the event definitions are frozen constants in [`src/unstructured_momentum/pipeline/contract.py`](src/unstructured_momentum/pipeline/contract.py); the horizon and quantile appear on the example output |
| 2 · Data design | [`docs/DATA_INVENTORY.md`](docs/DATA_INVENTORY.md), and `project/sources.yaml` — 26 sources with their publication lags and coverage limits |
| 3 · System design | [`docs/GATE2_POC.md`](docs/GATE2_POC.md) and [`docs/ANALOGUE_SYSTEM.md`](docs/ANALOGUE_SYSTEM.md); the deterministic/model split is visible in the read-out's stage labels, and [`docs/AI_USE_LOG.md`](docs/AI_USE_LOG.md) says where AI adds value and where it was declined |
| 4 · **Proof of concept** | **the whole repository**, and `make dash` for the interactive form |
| 5 · Validation | [`REGISTER.md`](REGISTER.md) — every experiment behind this package, each registered before it ran, four of them not supporting their hypothesis |
| 6 · Example output | [`reports/poc/page_2020-10-30.txt`](reports/poc/page_2020-10-30.txt) |
| 7 · Production path | the memo |

## Where the AI-use documentation is

[`docs/AI_USE_LOG.md`](docs/AI_USE_LOG.md) — what each model seat does, what checks it, and the
**measured** error rates rather than claimed ones. `project/traps.yaml` holds every instrument
fault found in the process, including the ones found in this deliverable while preparing it.

## Reading order, if you have twenty minutes

1. `README.md` — the result in nine numbers, three of which are failures.
2. `make dash` → **Example output**, and click a number. Then click **UAL**.
3. [`REGISTER.md`](REGISTER.md) — what was predicted before each test ran, including the 6 of 13 that ran and did not support their hypothesis, and the 21 instrument faults found on the way.
4. `project/traps.yaml`, last five entries — what was caught, and how.
