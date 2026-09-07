"""A local read-out for the momentum reversal system, with one live model seat.

Everything deterministic is served from the artifacts the scripts committed, so the page
comes up with no network and no key. One endpoint is live: `/api/extract` runs the same
schema-constrained extractor the proof of concept runs, on a filing the reader picks, with
the same quote gate, and shows the record it returns and whether the gate accepted it. That
is the seat the memo argues for, and it is worth watching rather than reading about.

    uv run python -m uvicorn app.server:app --reload --port 8000

The server holds no analytics. Every number it serves was computed upstream by a script with
a reproduce command, so the page cannot silently disagree with the engine.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPORTS = ROOT / "reports"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Momentum reversal read-out", docs_url="/api/docs")


@app.middleware("http")
async def _no_store(request, call_next):
    """Never let a browser cache the page.

    A stale index.html served from cache while the server has the new one is a demo failing for
    a reason that is invisible: the API returns the right data and the page renders the old
    template against it. Found exactly that way.
    """
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


def _load(rel: str):
    p = REPORTS / rel
    if not p.exists():
        raise HTTPException(404, f"missing artifact: reports/{rel}. Run its script first.")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------------------
# Deterministic reads
# --------------------------------------------------------------------------------------

SERIES_LABEL = {
    "mkt": "S&P 500",
    "d10y": "10-year yield, change",
    "wti": "front-month WTI",
    "usd": "dollar index",
    "ig": "investment-grade spread",
    "concentration": "index concentration",
    "pc1": "cross-asset PC1",
    "pc2": "cross-asset PC2",
}


def _rank_test(csv: str) -> float:
    """The registered one-sided Mann-Whitney on a stated-condition panel, from the artifact.

    exp-077 fixed the direction before the run: episode dates ABOVE control dates. Both panels
    carry a `group` column and a per-date `rate`, so the p-value the page quotes is recomputed
    here rather than pasted. Reproduces U=36, p=0.0175 on the design panel and U=19, p=0.4686
    on the sealed arm.
    """
    d = pd.read_csv(REPORTS / csv)
    g = d["group"].str.lower()
    ep = d.loc[~g.str.contains("control"), "rate"]
    ct = d.loc[g.str.contains("control"), "rate"]
    return float(mannwhitneyu(ep, ct, alternative="greater").pvalue)


def sealed_text_p() -> float:
    return _rank_test("gate2/sealed_text_arm.csv")


def design_tier_p() -> float:
    return _rank_test("gate2/exp077_panel.csv")



@app.get("/api/health")
def health() -> dict:
    """Whether the live seat can run. The page degrades rather than erroring without a key."""
    return {
        "live_seat": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "artifacts": {
            n: (REPORTS / n).exists()
            for n in ("current_book.json", "crash_inventory/summary.json",
                      "sealed_run.json", "same_rule_null.json",
                      "poc/page_2020-10-30.txt")
        },
    }


@app.get("/api/book")
def book() -> dict:
    """What the 12-1 sort is carrying at the latest formation date, against 500 random books."""
    b = _load("current_book.json")
    z = b["z"]
    ordered = sorted(z, key=lambda k: -z[k])
    # The level alone is not readable. A momentum book differs from a random size-matched book on
    # every macro axis at once -- the family test clears 108 of 108 -- so what a z of 15 means
    # only becomes visible against what this sort usually reads. The same-rule null supplies that.
    try:
        srn = _load("same_rule_null.json")
        median_max = float(srn["median_z"]["12-1"])
    except Exception:  # noqa: BLE001
        median_max = None
    top = z[ordered[0]]
    return {
        "context": {
            "max_z_today": round(top, 2),
            "median_max_z_12_1": round(median_max, 2) if median_max else None,
            "quieter_than_typical": (top < median_max) if median_max else None,
            "n_series_above_family_threshold": int(sum(v > 2.74 for v in z.values())),
            "note": (
                "All eight series clear the family threshold on most dates, which is why the "
                "eight-series test was retired as saturated. The level is therefore not the "
                "reading; the ORDERING is, and the level is only interpretable against what this "
                "sort usually carries."
            ),
        },
        "as_of": b["as_of"],
        "dominant": b["dominant"],
        "direction": b["direction"],
        "series": [
            {
                "key": k,
                "label": SERIES_LABEL.get(k, k),
                "z": round(z[k], 2),
                "spread": round(b["spread"][k], 4),
                "dominant": k == b["dominant"],
            }
            for k in ordered
        ],
        "winner_top10": b["winner_top10"],
        "winner_sectors": b["winner_top_sectors"],
        "loser_sectors": b["loser_top_sectors"],
    }


@app.get("/api/inventory")
def inventory() -> dict:
    """The 195 episodes and the route the registered rule assigned each."""
    s = _load("crash_inventory/summary.json")
    csv = REPORTS / "crash_inventory/inventory.csv"
    if not csv.exists():
        raise HTTPException(404, "run scripts/crash_inventory.py first")
    d = pd.read_csv(csv, parse_dates=["peak_at", "trough_at", "formation_at"])
    cols = ["definition", "trough_at", "depth", "mkt_during", "mkt_prior_2y", "bear_state",
            "loser_share", "winner_share", "retrace_20d", "crash_days", "route",
            "formation_at", "era"]
    rows = d[cols].copy()
    for c in ("trough_at", "formation_at"):
        rows[c] = rows[c].dt.strftime("%Y-%m-%d")
    return {
        "summary": {
            "n": s["n_episodes"],
            "routes": s["routes"],
            "fits_none_share": s["fits_none_share"],
            "fits_none_ci": s["fits_none_ci"],
            "registered_band": s["registered_band"],
            "in_band": s["in_band"],
            "residue": s["residue"],
            "planted": s["planted"],
        },
        "episodes": rows.where(pd.notna(rows), None).to_dict("records"),
    }


@app.get("/api/validation")
def validation() -> dict:
    """The three results a reader should weigh: the holdout, the same-rule null, the family."""
    sealed = _load("sealed_run.json")
    same = _load("same_rule_null.json")
    fam = _load("gap01_family.json")
    text = REPORTS / "gate2/sealed_text_arm.csv"
    arms = []
    if text.exists():
        t = pd.read_csv(text)
        for g, h in t.groupby("group"):
            arms.append({"group": g, "n": int(len(h)), "median_rate": float(h["rate"].median())})
    scoring = REPORTS / "crash_inventory/detector_scoring.json"
    det = json.loads(scoring.read_text()) if scoring.exists() else None
    # Read from the artifact rather than repeating it. These were hardcoded here and went stale
    # within a day of a correction landing upstream -- the same fault as the hand-typed page.
    reg = sealed["severity_registered_step"]
    dev = sealed["severity_daily_overlapping_deviation"]
    return {
        "detectors": det,
        "sealed_severity": {
            "registered_step": {**reg["conditional (full)"],
                                "verdict": "PASS" if sealed["passed"] else "FAIL",
                                "step_days": sealed["forecast_step_days"]},
            "baseline_same_dates": reg["unconditional"],
            "as_run_deviation": dev["conditional (full)"],
            "note": "The registered design is the 21-day step. The daily-overlapping run is "
                    "the logged deviation, trp-84. The unconditional baseline is printed on the "
                    "same dates because under trp-87 the conditional model was identical to it, "
                    "and that identity was the defect.",
            "episodes": sealed["sealed_episodes"],
        },
        # Read from the artifact, like the severity block above. These were literals -- the
        # exact fault the comment twenty lines up describes, left in place while the block
        # beside it was fixed.
        "sealed_text": {"arms": arms,
                        "p_one_sided": sealed_text_p(),
                        "design_tier_p": design_tier_p(),
                        "verdict": "DID NOT REPRODUCE"},
        "same_rule_null": same,
        "family_saturation": fam,
    }


@app.get("/api/page")
def page() -> JSONResponse:
    """The composed PM output, plus the provenance of every numeral on it.

    The page is composed over a field registry and gated by the numeral canary, so each number
    knows the artifact that produced it and the command that regenerates that artifact. That is
    what lets the reader click a number rather than take it on trust.
    """
    # The delivered example output. reports/gate2/EXAMPLE_RISK_OUTPUT.txt is the SUPERSEDED
    # page from the first design -- a fitted quantile regression, a -9.47% VaR against this
    # page's -19.29%, and a positioning section the design dropped. Serving it here meant the
    # dashboard the submission tells a reader to open disagreed with the submitted page by a
    # factor of two on the same date.
    p = REPORTS / "poc/page_2020-10-30.txt"
    if not p.exists():
        raise HTTPException(404, "run scripts/poc_e9_page.py first")
    prov = REPORTS / "poc/page_2020-10-30.provenance.json"
    return JSONResponse({"text": p.read_text(),
                         "provenance": json.loads(prov.read_text()) if prov.exists() else None})


@lru_cache(maxsize=1)
def _filing_table() -> pd.DataFrame:
    p = REPORTS / "gate2/nov2020_loser_filings.parquet"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    ex = REPORTS / "gate2/nov2020_extractions.parquet"
    if ex.exists():
        e = pd.read_parquet(ex)[["accession", "states_a_condition", "condition"]]
        d = d.merge(e, on="accession", how="left")
    return d


@app.get("/api/filings")
def filings() -> dict:
    """The loser leg's filings at the flagship formation date, for the live seat to read."""
    d = _filing_table()
    if not len(d):
        raise HTTPException(404, "run scripts/gate2_nov2020.py first")
    d = d.sort_values("accepted_at")
    out = []
    for _, r in d.iterrows():
        out.append({
            "accession": r["accession"],
            "ticker": r["ticker"],
            "accepted_at": str(r["accepted_at"])[:19],
            "already_read": bool(r.get("states_a_condition")) if "states_a_condition" in d else None,
            "stored_condition": (r.get("condition") or None) if "condition" in d else None,
        })
    return {"formation": "2020-10-31", "leg": "loser", "n": len(out), "filings": out}


@app.get("/api/filing/{accession}")
def filing(accession: str) -> dict:
    """The filing's own text, so a reader can check the quote against the source themselves."""
    from unstructured_momentum.data import eightk  # noqa: PLC0415
    try:
        body = eightk.load_body(accession)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"no stored body for {accession}: {exc}") from exc
    text = body.get("text") or body.get("body") or ""
    return {"accession": accession, "chars": len(text), "text": text[:60_000]}


# --------------------------------------------------------------------------------------
# The live seat
# --------------------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    accession: str


@app.post("/api/extract")
def extract(req: ExtractRequest) -> dict:
    """Run the extractor live, with the same schema and the same quote gate as the PoC.

    Returns the record AND the gate's verdict, because a record whose quote is not in the
    source is a fabrication and the page should show the check running, not just its result.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "no ANTHROPIC_API_KEY in the environment; the live seat is "
                                 "off and every other panel still works")
    import anthropic  # noqa: PLC0415

    from unstructured_momentum.data import eightk  # noqa: PLC0415
    from unstructured_momentum.llm.exposure import quote_is_grounded  # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "scripts"))
    from gate2_extract import MODEL, SYSTEM, SurvivalCondition  # noqa: PLC0415

    try:
        body = eightk.load_body(req.accession)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"no stored body for {req.accession}: {exc}") from exc
    # 18,000 characters is the excerpt the proof of concept reads. The live seat reads the
    # same window, so a record produced here is comparable to one in the committed parquet.
    excerpt = (body.get("text") or body.get("body") or "")[:18_000]
    if len(excerpt) < 200:
        raise HTTPException(422, "stored body is too short to read; the PoC drops these too")

    client = anthropic.Anthropic()
    r = client.messages.parse(
        model=MODEL, max_tokens=700, system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Filing items: {body.get('items', '')}\n\n{excerpt}"}],
        output_format=SurvivalCondition,
    )
    rec = r.parsed_output
    grounded = (bool(rec.quote) and quote_is_grounded(rec.quote, excerpt)) \
        if rec.states_a_condition else None
    return {
        "accession": req.accession,
        "ticker": body.get("ticker"),
        "model": MODEL,
        "record": rec.model_dump(),
        "gate": {
            "checked": bool(rec.states_a_condition),
            "grounded": grounded,
            "kept": (grounded is not False),
            "rule": "the quote must appear character-for-character in the source; an "
                    "unmatched span drops the record and is counted",
        },
        "chars_read": len(excerpt),
        "system": SYSTEM,
    }


# --------------------------------------------------------------------------------------
# The analogue engine, in four visible stages
# --------------------------------------------------------------------------------------

_ANALOGUE: dict = {}


def _rag():
    sys.path.insert(0, str(ROOT / "scripts"))
    import analogue_panel_rag as P  # noqa: PLC0415
    import analogue_rag as R  # noqa: PLC0415
    return R, P


@app.get("/api/analogue/record")
def analogue_record(date: str = "2020-10-31") -> dict:
    """Stages 1 and 2. Arithmetic and an encoder; no generation, no key needed.

    Cached per date in process because building it loads a transformer checkpoint and the
    retrieval engine, which takes seconds rather than milliseconds.
    """
    if date in _ANALOGUE:
        return _ANALOGUE[date]
    R, P = _rag()
    corpus = R.build_corpus()
    q = pd.Timestamp(date)
    rec = P.build_record(q, corpus)
    out = {"record": rec, "planted_control": P.planted_control(rec),
           "corpus": {"records": int(len(corpus)),
                      "dates": int(corpus["formation"].nunique()),
                      "first": str(corpus["formation"].min().date()),
                      "last": str(corpus["formation"].max().date())}}
    _ANALOGUE[date] = out
    return out


@app.get("/api/analogue/row09")
def analogue_row09() -> dict:
    """Register row 09: how far the date-constrained and contemporary readings agree."""
    p = REPORTS / "analogue_rag/row09_summary.json"
    c = REPORTS / "analogue_rag/row09_agreement.csv"
    if not p.exists():
        raise HTTPException(404, "run scripts/analogue_rag.py --all first")
    out = json.loads(p.read_text())
    if c.exists():
        out["per_query"] = pd.read_csv(c).where(lambda d: d.notna(), None).to_dict("records")
    # The ablation that says what row 09's disagreement actually is.
    ab = REPORTS / "analogue_rag/encoder_cutoff_ablation.json"
    abc = REPORTS / "analogue_rag/encoder_cutoff_ablation.csv"
    if ab.exists():
        out["cutoff_ablation"] = json.loads(ab.read_text())
        if abc.exists():
            out["cutoff_ablation"]["per_query"] = pd.read_csv(abc).to_dict("records")
    return out


class PanelRequest(BaseModel):
    date: str = "2020-10-31"


@app.post("/api/analogue/panel")
def analogue_panel(req: PanelRequest) -> dict:
    """Stages 3 and 4, live: proponent, dissent, adjudicator, both gates running."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "no ANTHROPIC_API_KEY, so the panel cannot run. Stages 1 and 2 "
                                 "are deterministic and are already on the page above.")
    _, P = _rag()
    rec = analogue_record(req.date)["record"]
    return P.run_panel(rec)


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
