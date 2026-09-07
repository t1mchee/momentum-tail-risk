"""Event tripwires: the LLM declares once, deterministic code watches forever.

A macro falsifier (llm/translate.py) is scored by lookup against a fixed series registry.
Many drivers also have an ISSUER-EVENT signature: if the named bet is wrong, some of the
companies carrying it will eventually say so in an 8-K — a guidance move under Item 7.01,
a contract termination under 1.02, an impairment under 2.06. This module turns that into a
watchable condition with the same division of labour the rest of the system uses:

* The model declares ONCE, at arming time: which item families, which anchor phrases, how
  many distinct filers, over what horizon (llm.translate.EventFalsifier). Those fields are
  the claim. The declaration carries a mandatory DECLINE — some drivers have no issuer-event
  signature, and a declined event falsifier is a correct answer that gets logged as such.
* Deterministic code does ALL the watching. The daily scan is index filters and lowercase
  substring matches. Each survivor goes one-at-a-time to a Sonnet call that judges relevance
  only — its schema has no number field — and a MECHANICAL QUOTE GATE discards any judgment
  whose quote is not an exact substring of the filing body. The trip decision is a
  distinct-ticker count against the declared K. No model output is ever counted, averaged,
  or scored by a model.

Status vocabulary mirrors translate.score(): stood / refuted / pending / unscoreable, where
"refuted" means the tripwire TRIPPED — the issuer record contradicted the driver. A "stood"
or "refuted" verdict carries the scan's coverage counts, because a null over a corpus that
held no member filings is a statement about the corpus, not the world (repo rule: a refuted
verdict is rejected without a power statement).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..data.eightk import MATERIAL_ITEMS, load_index
from ..llm.translate import ITEM_MEANINGS, EventFalsifier

JUDGE_MODEL = "claude-sonnet-5"
TRIPWIRES = Path("data/processed/tripwires.jsonl")
HITS = Path("data/processed/tripwire_hits.jsonl")
CACHE = Path("data/interim/tripwires")

# --------------------------------------------------------------------------- ledgers


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def _append_once(path: Path, rec: dict, key: str) -> bool:
    """Append-once by content key; nightly re-runs and replays are idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and any(r.get("key") == key for r in _read_jsonl(path)):
        return False
    rec = {**rec, "key": key}
    with path.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return True


# --------------------------------------------------------------------------- recipe

def normalise_recipe(ef: EventFalsifier | dict) -> tuple[dict, list[str]]:
    """Deterministic normalisation of a declared recipe, with every repair reported.

    Item codes outside the closed vocabulary are DROPPED, not guessed at — a falsifier
    naming an item family the corpus does not carry is unscoreable, which is the failure
    the closed vocabulary exists to avoid. Anchors are lowercased because matching is
    lowercase. The horizon is clamped to the same 5-63 trading-day band the macro
    falsifier uses. Nothing here invents content; it only removes or bounds."""
    d = ef if isinstance(ef, dict) else ef.model_dump()
    problems: list[str] = []
    codes = []
    for c in d.get("item_codes", []):
        c = str(c).strip()
        if c in MATERIAL_ITEMS:
            codes.append(c)
        else:
            problems.append(f"item code {c!r} not in vocabulary; dropped")
    anchors = []
    for a in d.get("anchor_terms", []):
        a = str(a).strip().lower()
        if a and a not in anchors:
            anchors.append(a)
    if len(anchors) > 8:
        problems.append(f"{len(anchors)} anchors declared; kept first 8")
        anchors = anchors[:8]
    if len(anchors) < 3 and anchors:
        problems.append(f"only {len(anchors)} anchors declared (3-8 asked for); kept as-is")
    k = max(1, int(d.get("min_distinct_filers", 1)))
    hz = min(63, max(5, int(d.get("horizon_days", 21))))
    if hz != int(d.get("horizon_days", hz)):
        problems.append(f"horizon {d.get('horizon_days')} clamped to {hz}")
    out = {"item_codes": sorted(set(codes)), "anchor_terms": anchors,
           "min_distinct_filers": k, "horizon_days": hz,
           "refutes": str(d.get("refutes", "")).strip()}
    return out, problems


def tripwire_id(asof: str, members: list[str], recipe: dict) -> str:
    payload = json.dumps([asof, sorted(members), recipe["item_codes"],
                          recipe["anchor_terms"], recipe["min_distinct_filers"],
                          recipe["horizon_days"]], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def horizon_end(asof: str, horizon_days: int) -> pd.Timestamp:
    """Trading-day horizon end, tz-aware to compare against SEC acceptance stamps."""
    end = pd.Timestamp(asof) + pd.tseries.offsets.BDay(int(horizon_days))
    return end.tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59)


# --------------------------------------------------------------------------- declaration

DECLARE_SYSTEM = """\
You are declaring an ISSUER-EVENT falsifier for a bet a momentum portfolio is carrying.

The bet has already been identified; it is given to you. Your one job is to state, ONCE,
the condition under which the companies carrying the bet would themselves show it wrong in
their 8-K filings. After this declaration, deterministic code does all the watching: item
codes are matched against the corpus index, anchor terms are matched as lowercase
substrings of filing bodies, and the tripwire trips only when your declared number of
DISTINCT filers have quote-gated refuting evidence inside your declared horizon. You will
never see the filings and nothing you output is a measurement.

Rules:
- item_codes must come from the given closed vocabulary. Choose the item families the
  refuting filing would legally arrive under, not every family that might mention it.
- anchor_terms: 3 to 8 lowercase phrases. They are a RECALL net, not the refutation
  itself: a filing is read at all only if an anchor appears as a substring of its body,
  and then a reader judges whether it actually refutes. So choose the short topic
  vocabulary any filing about this subject would contain ("guidance", "bookings", "data
  center", "capacity", "impairment") -- single words and two-word phrases -- NOT the
  refuting conclusion pre-worded ("withdrew guidance", "cancelled the data center"). A
  five-word anchor almost never appears verbatim, and a filing your net misses can never
  be judged; a too-broad anchor only costs the reader a look at a filing it declines.
- min_distinct_filers: one filer is an idiosyncratic story. The number you declare is how
  many distinct companies must file evidence a reader accepts as REFUTING; it is what
  makes a trip a claim about the shared bet, so 2 or 3 is usually right.
  horizon_days is 5 to 63 trading days.
- DECLINING IS A CORRECT ANSWER. Some drivers have no issuer-event signature — a
  rates/duration bet is refuted by market series, not by anything a company files. If no
  8-K item family could carry evidence for or against this driver, return null for
  event_falsifier and say why in decline_reason.
"""

DECLARE_USER = """\
As of {asof}.

DRIVER (the bet, as named at declaration time):
{driver}

THEME MEMBERS (the tickers carrying it):
{members}

8-K ITEM VOCABULARY:
{items}

Declare the event falsifier, or decline.
"""


class EventDeclaration(BaseModel):
    """Structured wrapper so the DECLINE option is part of the schema, not prompt hope."""

    event_falsifier: EventFalsifier | None = Field(
        default=None, description="The declared recipe, or null to DECLINE.")
    decline_reason: str = Field(
        default="", description="Required when event_falsifier is null.")


def declare(asof: str, driver: str, members: list[str], *,
            feedback: str = "", model: str = JUDGE_MODEL,
            use_cache: bool = True) -> EventDeclaration:
    """One declaration call, cached on its inputs like Translator.generate.

    ``feedback`` carries the DETERMINISTIC placebo-calibration facts back to the declarer
    for at most one revision round at arming time (see the replay script). The numbers in
    the feedback are measured by code, never by a model, and once a tripwire is armed the
    watch loop never revises it."""
    key = hashlib.sha256(json.dumps([asof, driver, sorted(members), model, feedback, "v2"],
                                    sort_keys=True).encode()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"declare_{key}.json"
    if use_cache and f.exists():
        return EventDeclaration(**json.loads(f.read_text()))
    import anthropic
    body = DECLARE_USER.format(
        asof=asof, driver=driver, members=", ".join(sorted(members)),
        items="\n".join(f"  {k}: {v}" for k, v in ITEM_MEANINGS.items()))
    if feedback:
        body += ("\nCALIBRATION FEEDBACK on your previous declaration (measured "
                 "deterministically on placebo baskets in calm windows; a workable "
                 "recipe's string layer clears roughly 1-in-4):\n" + feedback)
    cli = anthropic.Anthropic()
    out = None
    for _ in range(2):  # transient empty parsed outputs observed (as in daily_xray)
        resp = cli.messages.parse(
            model=model, max_tokens=2000, system=DECLARE_SYSTEM,
            messages=[{"role": "user", "content": body}],
            output_format=EventDeclaration)
        out = resp.parsed_output
        if out is not None:
            break
    if out is None:
        raise RuntimeError("declaration returned no parsed output after retry")
    f.write_text(out.model_dump_json(indent=2))
    return out


def arm(asof: str, driver: str, members: list[str], decl: EventDeclaration,
        *, placebo: dict | None = None, note: str = "",
        revision_of: dict | None = None) -> dict:
    """Log one armed tripwire (or a decline) to the tripwire ledger, append-once."""
    members = sorted(set(members))
    if decl.event_falsifier is None:
        rec = {"as_of": asof, "driver": driver, "members": members,
               "status_at_arming": "declined",
               "decline_reason": decl.decline_reason, "note": note}
        key = hashlib.sha256(json.dumps([asof, driver, members, "declined"],
                                        sort_keys=True).encode()).hexdigest()[:16]
        _append_once(TRIPWIRES, rec, key)
        return {**rec, "key": key}
    recipe, problems = normalise_recipe(decl.event_falsifier)
    tid = tripwire_id(asof, members, recipe)
    rec = {"as_of": asof, "driver": driver, "members": members, **recipe,
           "tripwire_id": tid, "status_at_arming": "armed",
           "normalisation": problems, "note": note,
           # Measured, never tuned: the placebo clearance rate is logged NEXT TO the armed
           # recipe so a reader can see how discriminating the declared K actually is.
           "placebo_clearance": placebo,
           # If this recipe is a one-round revision of an earlier declaration, the earlier
           # recipe and ITS measured rate stay in the record — the revision is audit
           # trail, not something that happened off the books.
           "revision_of": revision_of}
    _append_once(TRIPWIRES, rec, tid)
    return {**rec, "key": tid}


# --------------------------------------------------------------------------- daily scan

def scan(rec: dict, *, idx: pd.DataFrame | None = None,
         since: str | pd.Timestamp | None = None,
         until: str | pd.Timestamp | None = None) -> list[dict]:
    """The deterministic string layer: index filters and lowercase substring matches.

    (accepted after the as-of date, or ``since`` for an incremental run) x (ticker in the
    theme) x (items intersect the declared codes) x (any anchor appears in the lowercased
    body). Survivors — and only survivors — are worth a model's attention. Re-scanning is
    harmless: the hit ledger appends once per (tripwire, accession)."""
    idx = load_index() if idx is None else idx
    if idx.empty:
        return []
    lo = pd.Timestamp(since) if since is not None else pd.Timestamp(rec["as_of"])
    if lo.tzinfo is None:
        lo = lo.tz_localize("UTC")
    hi = horizon_end(rec["as_of"], rec["horizon_days"])
    if until is not None:
        u = pd.Timestamp(until)
        u = u.tz_localize("UTC") if u.tzinfo is None else u
        hi = min(hi, u)
    sub = idx[idx["ticker"].isin(set(rec["members"]))]
    acc = pd.to_datetime(sub["accepted_at"], utc=True)
    sub = sub[(acc > lo) & (acc <= hi)]
    codes = set(rec["item_codes"])
    sub = sub[sub["items"].fillna("").map(
        lambda s: bool(codes & {c.strip() for c in s.split(",")}))]
    sub = sub.drop_duplicates(subset="accession", keep="last")
    out = []
    for _, row in sub.iterrows():
        body = _body(row["accession"])
        if not body:
            continue
        low = body.lower()
        matched = [a for a in rec["anchor_terms"] if a in low]
        if matched:
            out.append({"ticker": row["ticker"], "accession": row["accession"],
                        "filing_date": str(pd.Timestamp(row["filing_date"]).date()),
                        "accepted_at": str(row["accepted_at"]),
                        "items": row["items"], "anchors_matched": matched,
                        "body": body})
    return out


def _body(accession: str) -> str:
    from ..data.eightk import TEXT
    f = TEXT / f"{accession.replace('-', '')}.json.gz"
    if not f.exists():
        return ""
    try:
        return json.loads(gzip.decompress(f.read_bytes())).get("body", "")
    except (OSError, ValueError):
        return ""


# --------------------------------------------------------------------------- judgment

class TripwireJudgment(BaseModel):
    """Relevance only. No number field exists here, so the model cannot emit one."""

    is_evidence: bool = Field(description=(
        "True only if this filing text substantively bears on the declared driver. False "
        "for boilerplate, routine administration, or an anchor phrase hit in passing."))
    bears_on: Literal["refutes", "consistent"] = Field(description=(
        "'refutes' if this is the kind of issuer event the declaration said would show the "
        "driver wrong; 'consistent' if it bears on the driver but supports or is neutral "
        "to it. Ignored when is_evidence is false."))
    quote: str = Field(description=(
        "VERBATIM contiguous excerpt (under 300 characters) copied EXACTLY from the given "
        "text, showing the evidence. Checked mechanically against the filing body; the "
        "judgment is discarded if it is not an exact substring. Empty when declining."))
    decline_reason: str = Field(description=(
        "When is_evidence is false: one sentence on why this is not evidence. Declining "
        "is the expected answer for most anchor matches."))


JUDGE_SYSTEM = """\
You are reading excerpts of ONE SEC 8-K filing to decide whether it is evidence bearing on
a declared bet. You are not scoring, counting, or forecasting anything — deterministic
code does that. Your only job is relevance, and DECLINING IS A CORRECT ANSWER: most
filings that reach you matched a keyword in passing and are not evidence.

If it is evidence, your quote must be copied verbatim from the text shown. It will be
checked character-for-character against the full filing body and your judgment is thrown
away if it does not match exactly, so do not paraphrase, normalise whitespace, or bridge
two passages.
"""

JUDGE_USER = """\
DECLARED BET (as of {asof}): {driver}

WHAT WOULD REFUTE IT (declared at arming time): {refutes}

FILING: {ticker}, 8-K items {items}, filed {filing_date}. Excerpts around the anchor
matches ({anchors}):

{excerpts}

Judge relevance, or decline.
"""


def _excerpts(body: str, anchors: list[str], *, width: int = 1500, max_windows: int = 3) -> str:
    """Windows around anchor matches: the model sees where the string layer fired, the
    quote gate still checks against the FULL body."""
    low = body.lower()
    spans: list[tuple[int, int]] = []
    for a in anchors:
        p = low.find(a)
        while p != -1 and len(spans) < 12:
            spans.append((max(0, p - width), min(len(body), p + len(a) + width)))
            p = low.find(a, p + 1)
    spans.sort()
    merged: list[list[int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    merged = merged[:max_windows]
    return "\n\n[...]\n\n".join(body[lo:hi] for lo, hi in merged)


def quote_gate(quote: str, body: str) -> bool:
    """The mechanical gate: an accepted quote must be an exact substring of the source
    body. This is what makes every ledger row independently checkable against EDGAR."""
    return bool(quote) and quote in body


def judge(rec: dict, surv: dict, *, model: str = JUDGE_MODEL,
          use_cache: bool = True) -> dict:
    """One survivor, one call, one gated ledger row (returned; run() appends it)."""
    key = hashlib.sha256(json.dumps(
        [rec.get("tripwire_id"), surv["accession"], model, "v1"],
        sort_keys=True).encode()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"judge_{key}.json"
    if use_cache and f.exists():
        j = TripwireJudgment(**json.loads(f.read_text()))
    else:
        import anthropic
        body = JUDGE_USER.format(
            asof=rec["as_of"], driver=rec["driver"], refutes=rec["refutes"],
            ticker=surv["ticker"], items=surv["items"],
            filing_date=surv["filing_date"],
            anchors=", ".join(surv["anchors_matched"]),
            excerpts=_excerpts(surv["body"], surv["anchors_matched"]))
        cli = anthropic.Anthropic()
        j = None
        for _ in range(2):  # transient empty parsed outputs observed (as in daily_xray)
            resp = cli.messages.parse(
                model=model, max_tokens=1200, system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": body}],
                output_format=TripwireJudgment)
            j = resp.parsed_output
            if j is not None:
                break
        if j is None:
            raise RuntimeError(
                f"judgment returned no parsed output after retry ({surv['accession']})")
        f.write_text(j.model_dump_json(indent=2))
    gate = "n/a"
    if j.is_evidence:
        gate = "passed" if quote_gate(j.quote, surv["body"]) else "failed"
    return {"tripwire_id": rec.get("tripwire_id"), "as_of": rec["as_of"],
            "driver": rec["driver"], "ticker": surv["ticker"],
            "accession": surv["accession"], "filing_date": surv["filing_date"],
            "accepted_at": surv["accepted_at"], "items": surv["items"],
            "anchors_matched": surv["anchors_matched"],
            "is_evidence": j.is_evidence, "bears_on": j.bears_on,
            "quote": j.quote, "decline_reason": j.decline_reason,
            "quote_gate": gate, "model": model,
            "human_adjudication": ""}


def run(rec: dict, *, idx: pd.DataFrame | None = None,
        since=None, until=None, model: str = JUDGE_MODEL) -> list[dict]:
    """Scan, judge each survivor once, append gated rows to the hit ledger.

    Every judged survivor is logged — accepted evidence, declines, and gate failures alike
    — because the ledger is the audit trail; only rows with is_evidence, bears_on
    'refutes' and a PASSED quote gate count toward tripping, and that counting happens in
    status(), deterministically."""
    rows = []
    for surv in scan(rec, idx=idx, since=since, until=until):
        row = judge(rec, surv, model=model)
        key = hashlib.sha256(
            f"{rec.get('tripwire_id')}|{surv['accession']}".encode()).hexdigest()[:16]
        _append_once(HITS, row, key)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- status

def refuting(rec: dict, hits: list[dict] | None = None) -> list[dict]:
    """Ledger rows that count: evidence, declared-refuting, and quote-gate passed."""
    hits = _read_jsonl(HITS) if hits is None else hits
    hi = horizon_end(rec["as_of"], rec["horizon_days"])
    out = []
    for h in hits:
        if h.get("tripwire_id") != rec.get("tripwire_id"):
            continue
        if not (h.get("is_evidence") and h.get("bears_on") == "refutes"
                and h.get("quote_gate") == "passed"):
            continue
        if pd.Timestamp(h["accepted_at"]) <= hi:
            out.append(h)
    return out


def status(rec: dict, *, hits: list[dict] | None = None,
           idx: pd.DataFrame | None = None,
           asof: str | pd.Timestamp | None = None) -> dict:
    """Mechanically mark one armed tripwire. Mirrors translate.score(): no model here.

    'refuted' = TRIPPED: >= min_distinct_filers distinct tickers with gated refuting
    evidence inside the horizon. A verdict is only issued with the coverage counts that
    show the instrument could have seen the effect."""
    if rec.get("status_at_arming") == "declined":
        return {**{k: rec[k] for k in ("as_of", "driver")}, "status": "declined",
                "reason": rec.get("decline_reason", "")}
    if not rec.get("item_codes") or not rec.get("anchor_terms"):
        return {"as_of": rec["as_of"], "driver": rec["driver"], "status": "unscoreable",
                "reason": "empty item codes or anchors after normalisation"}
    idx = load_index() if idx is None else idx
    hi = horizon_end(rec["as_of"], rec["horizon_days"])
    now = pd.Timestamp(asof).tz_localize("UTC") if asof is not None \
        else pd.Timestamp.now(tz="UTC")
    # Power statement: how much of the horizon the corpus could actually see.
    lo = pd.Timestamp(rec["as_of"]).tz_localize("UTC")
    sub = idx[idx["ticker"].isin(set(rec["members"]))] if not idx.empty else idx
    if not idx.empty:
        acc = pd.to_datetime(sub["accepted_at"], utc=True)
        inwin = sub[(acc > lo) & (acc <= min(hi, now))]
        codes = set(rec["item_codes"])
        item_rows = inwin[inwin["items"].fillna("").map(
            lambda s: bool(codes & {c.strip() for c in s.split(",")}))]
        cov = {"members": len(rec["members"]),
               "members_with_any_filing": int(inwin["ticker"].nunique()),
               "filings_in_window": int(len(inwin)),
               "filings_matching_items": int(len(item_rows))}
    else:
        cov = {"members": len(rec["members"]), "members_with_any_filing": 0,
               "filings_in_window": 0, "filings_matching_items": 0}
    ref = refuting(rec, hits)
    distinct = sorted({h["ticker"] for h in ref})
    base = {"as_of": rec["as_of"], "driver": rec["driver"],
            "tripwire_id": rec.get("tripwire_id"),
            "min_distinct_filers": rec["min_distinct_filers"],
            "horizon_days": rec["horizon_days"],
            "n_refuting_filers": len(distinct), "refuting_filers": distinct,
            "coverage": cov}
    if len(distinct) >= int(rec["min_distinct_filers"]):
        return {**base, "status": "refuted"}
    if cov["filings_in_window"] == 0 and cov["members_with_any_filing"] == 0 \
            and now >= hi:
        return {**base, "status": "unscoreable",
                "reason": "no member filed anything in the window; the corpus could not "
                          "have seen the effect"}
    if now < hi:
        return {**base, "status": "pending",
                "reason": f"horizon runs to {hi.date()}"}
    return {**base, "status": "stood"}


# --------------------------------------------------------------------------- K calibration

#: Registered break episodes (mirrors scripts/analog_detector.FAMILIES, plus the COVID
#: crash months, which that registry carries only implicitly). Calm windows for placebo
#: calibration must not straddle any of these — clearance measured through a real break
#: would answer a different question.
EPISODES = ["2018-02-05", "2019-09-09", "2020-02-24", "2020-03-16", "2020-11-09",
            "2021-01-27", "2022-11-10", "2024-08-05", "2025-01-27", "2025-04-09"]


def calm_windows(horizon_days: int, *, n: int = 20,
                 start: str = "2018-03-31", end: str = "2024-09-30") -> list[pd.Timestamp]:
    """Deterministic calm as-of dates: month-ends whose horizon window stays at least 21
    calendar days clear of every registered episode, thinned evenly to n."""
    eps = [pd.Timestamp(e) for e in EPISODES]
    ok = []
    for d in pd.date_range(start, end, freq="ME"):
        w_end = d + pd.tseries.offsets.BDay(int(horizon_days))
        if all(e < d - pd.Timedelta(days=21) or e > w_end + pd.Timedelta(days=21)
               for e in eps):
            ok.append(d)
    if len(ok) <= n:
        return ok
    step = len(ok) / n
    return [ok[int(i * step)] for i in range(n)]


def placebo_clearance(recipe: dict, basket_size: int, *,
                      idx: pd.DataFrame | None = None,
                      leg_members_path: str = "data/processed/leg_members.pkl",
                      n_windows: int = 20, baskets_per_window: int = 10,
                      seed: int = 0) -> dict:
    """How often would this recipe's STRING LAYER clear on random baskets in calm times?

    DESIGN-tier calibration, fully deterministic, and disclosed as such: for the declared
    (item_codes x anchors x K x horizon), draw random same-size baskets of book names at
    each of ~20 calm month-ends and count how often >= K distinct placebo tickers have a
    filing matching (items intersect) x (anchor substring) inside the horizon. No LLM
    touches a placebo: the judge and the refutes-direction filter can only REMOVE hits, so
    the measured rate is an UPPER BOUND on the placebo tripped-rate. A well-set recipe
    clears roughly 1-in-4 here — rarely enough to discriminate, often enough that the
    threshold is real rather than unreachable. The measured rate is LOGGED next to the
    armed tripwire (see arm()); K is never tuned automatically, because a recipe whose
    clearance is far from target is a fact about the declaration, to be reported, not a
    parameter to be silently repaired."""
    import pickle

    import numpy as np

    idx = load_index() if idx is None else idx
    if idx.empty:
        return {"error": "no corpus index"}
    lm = pickle.load(open(leg_members_path, "rb"))
    keys = sorted(lm.keys())
    corpus_tk = set(idx["ticker"].unique())
    codes = set(recipe["item_codes"])
    anchors = list(recipe["anchor_terms"])
    k = int(recipe["min_distinct_filers"])
    rng = np.random.default_rng(seed)
    acc_all = pd.to_datetime(idx["accepted_at"], utc=True)

    cleared, total, per_window = 0, 0, []
    anchor_matches = {a: 0 for a in anchors}  # diagnostic: which anchors ever fire at all
    for w in calm_windows(recipe["horizon_days"], n=n_windows):
        mkey = max((x for x in keys if x <= w), default=None)
        if mkey is None:
            continue
        names = sorted((set(lm[mkey]["winners"]) | set(lm[mkey]["losers"])) & corpus_tk)
        if len(names) < basket_size:
            continue
        lo = w.tz_localize("UTC")
        hi = horizon_end(str(w.date()), recipe["horizon_days"])
        inwin = idx[(acc_all > lo) & (acc_all <= hi)]
        inwin = inwin[inwin["items"].fillna("").map(
            lambda s: bool(codes & {c.strip() for c in s.split(",")}))]
        inwin = inwin.drop_duplicates(subset="accession", keep="last")
        # Anchor-match once per candidate filing in this window, then count per basket.
        matched_tickers: dict[str, set[str]] = {}
        for _, row in inwin[inwin["ticker"].isin(names)].iterrows():
            body = _body(row["accession"]).lower()
            if not body:
                continue
            hit_anchors = [a for a in anchors if a in body]
            for a in hit_anchors:
                anchor_matches[a] += 1
            if hit_anchors:
                matched_tickers.setdefault(row["ticker"], set()).add(row["accession"])
        hit_set = set(matched_tickers)
        wc = 0
        for _ in range(baskets_per_window):
            basket = set(rng.choice(names, size=basket_size, replace=False))
            if len(basket & hit_set) >= k:
                cleared += 1
                wc += 1
            total += 1
        per_window.append({"window": str(w.date()), "cleared": wc,
                           "of": baskets_per_window,
                           "string_hit_tickers": sorted(hit_set)[:12]})
    rate = round(cleared / total, 3) if total else None
    return {"clearance_rate": rate, "cleared": cleared, "baskets": total,
            "anchor_match_filings": anchor_matches,
            "target": "roughly 1-in-4 (0.25)", "tier": "DESIGN",
            "note": ("string-layer arm rate on placebo baskets in calm windows; upper "
                     "bound on placebo tripped-rate (LLM gate and refutes filter only "
                     "remove hits); K is logged against this, never auto-tuned"),
            "windows": per_window}


# --------------------------------------------------------------------------- X-ray feed

def watch_summary(*, asof: str | pd.Timestamp | None = None) -> dict:
    """Everything the daily X-ray's falsifier-watch line needs, computed mechanically."""
    from ..llm.translate import LOG as FALS, content_key
    macro = []
    seen: set[str] = set()
    for r in _read_jsonl(FALS):
        key = r.get("key") or content_key(r)
        if key not in seen:
            seen.add(key)
            macro.append(r)
    stamps = [r.get("vintages", {}).get("run_at") or r.get("as_of") for r in macro]
    armed, tripped, declined = [], 0, 0
    hits = _read_jsonl(HITS)
    idx = load_index()
    for rec in _read_jsonl(TRIPWIRES):
        if rec.get("status_at_arming") == "declined":
            declined += 1
            continue
        st = status(rec, hits=hits, idx=idx, asof=asof)
        armed.append({"driver": rec["driver"], "as_of": rec["as_of"],
                      "k": rec["min_distinct_filers"], "n_members": len(rec["members"]),
                      "horizon_days": rec["horizon_days"], "status": st["status"],
                      "n_refuting_filers": st.get("n_refuting_filers", 0),
                      "placebo_clearance": (rec.get("placebo_clearance") or {}).get(
                          "clearance_rate")})
        if st["status"] == "refuted":
            tripped += 1
    return {"macro_active": len(macro),
            "macro_last_update": max([s for s in stamps if s], default=None),
            "event_armed": armed, "event_tripped": tripped,
            "event_declined": declined,
            "ledger": str(HITS)}
