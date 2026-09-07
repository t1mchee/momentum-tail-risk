"""exp-082 arm A -- the analogue panel, made two-sided, with a gate that can now catch it.

Registered at commit 71be4e8ba. Four stages, and the dashboard shows all four:

  1. STATISTICAL MATCH   the analogue engine's nearest states, covariance-aware distance over
                         the state vector, embargoed. Arithmetic. No model.
  2. RETRIEVAL           the nearest episodes by WHAT THE LOSER LEG WAS PRICED ON, over the 419
                         grounded condition records, read by a date-constrained encoder whose
                         cutoff precedes the query and by a contemporary one. No generation.
  3. COMPARISON          proponent, dissent, adjudicator over the merged record. Generation,
                         gated twice.
  4. OUTPUT              the verdict, with the claims that survived and the claims that did not.

The gate is sharpened here and that is the point of the experiment. The old gate checked that a
cited field EXISTED SOMEWHERE in the record, which is why the dissent could cite `date` and
`block_closeness` while making claims about fiscal stimulus. The new one:

  * resolves the claim to the entity it is about -- the query, or one named analogue;
  * requires the field to be present AND non-null in THAT entity's own sub-record;
  * and for any claim about mechanism, requires a verbatim span that appears character for
    character in one of that entity's stored quotes.

An analogue with no condition record therefore has no field to cite and no span to quote, so a
mechanism claim about it falls by the rule the gate already applies. The corpus begins
2018-09-30, so the pre-2018 neighbours are exactly that case, and they are the control rather
than a gap to apologise for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from unstructured_momentum.analogue.retrieve import Engine  # noqa: E402
from unstructured_momentum.data import french  # noqa: E402
from unstructured_momentum.llm.exposure import quote_is_grounded  # noqa: E402

import analogue_rag as RAG  # noqa: E402

MODEL = "claude-sonnet-4-5-20250929"
OUT = Path("reports/analogue_rag")
#: Fields whose claims are about MECHANISM and therefore need a span, not just a field.
CONDITION_FIELDS = {"conditions", "condition", "shared_condition", "quote"}


class Claim(BaseModel):
    about: str = Field(
        description="Exactly 'query', or the date string of ONE analogue this claim is about.")
    field: str = Field(description="The exact field name inside that entity's record.")
    quote: str = Field(
        default="",
        description="Required when the field is 'conditions': a VERBATIM span copied from one "
                    "of that entity's stored quotes. Leave empty for numeric fields.")
    claim: str = Field(description="One sentence. What that field shows about the match.")


class Side(BaseModel):
    claims: list[Claim] = Field(description="Two to four claims, each about one entity.")


class Verdict(BaseModel):
    state_match_is_apt: bool = Field(
        description="Is the nearest STATE match a usable analogue for the query date?")
    mechanism_is_rulable: bool = Field(
        description="False when no condition record exists for the state match, so mechanism "
                    "cannot be ruled on from the record at all.")
    fails_on: str = Field(default="", description="The single dimension on which it fails.")
    reasoning: str = Field(description="Two sentences, referring only to cited fields.")


# --------------------------------------------------------------------------------------
# Stages 1 and 2: the record
# --------------------------------------------------------------------------------------


def condition_block(corpus: pd.DataFrame, formation: pd.Timestamp) -> dict | None:
    g = corpus[corpus["formation"] == formation]
    if not len(g):
        return None
    return {
        "n_records": int(len(g)),
        "n_names": int(g["ticker"].nunique()),
        "conditions": [{"ticker": r["ticker"], "condition": r["condition"], "quote": r["quote"]}
                       for _, r in g.iterrows()],
    }


def build_record(query: pd.Timestamp, corpus: pd.DataFrame, k: int = 5) -> dict:
    """Merge the state matches and the condition matches into one record the panel argues over."""
    eng = Engine("deep")
    w = french.momentum()
    state = []
    for n in eng.query(query, k=k):
        fwd = w[w.index > n.date]
        blk = condition_block(corpus, pd.Timestamp(n.date))
        state.append({
            "date": str(n.date.date()), "rank": n.rank, "distance": round(n.distance, 4),
            "block_closeness": {kk: round(v, 3) for kk, v in n.block_closeness.items()},
            "forward_21d_return": round(float((1 + fwd.iloc[:21]).prod() - 1), 4),
            "condition_record": blk,
            "condition_record_status": "present" if blk else "NO RECORD",
        })

    ranked = RAG.rank_one(corpus, query, "chrono")
    cond = []
    for r in ranked["ranking"][:k]:
        f = pd.Timestamp(r["formation"])
        fwd = w[w.index > f]
        cond.append({
            "date": r["formation"], "rank": r["rank"],
            "condition_similarity": r["similarity"],
            "forward_21d_return": round(float((1 + fwd.iloc[:21]).prod() - 1), 4),
            "condition_record": condition_block(corpus, f),
            "condition_record_status": "present",
        })

    qb = condition_block(corpus, query)
    return {
        "query": {
            "date": str(query.date()),
            "n_records": qb["n_records"] if qb else 0,
            "n_names": qb["n_names"] if qb else 0,
            "conditions": qb["conditions"] if qb else [],
        },
        "encoder": ranked.get("encoder_id"),
        "n_candidates_after_embargo": ranked.get("n_candidates"),
        "state_matches": state,
        "condition_matches": cond,
        "corpus_note": (
            "The condition corpus begins 2018-09-30, where the 8-K body archive begins. A state "
            "match earlier than that carries condition_record_status NO RECORD, and no claim "
            "about what it was priced on can be made from this record."
        ),
        "similarity_note": (
            "condition_similarity is a RANK, not a level. Vectors are centred on the embargoed "
            "candidate pool, so with a pool this small the mean similarity is forced negative by "
            "construction and the absolute value carries no information. Only the ordering does, "
            "and the ordering is what the two-encoder agreement test scores."
        ),
    }


# --------------------------------------------------------------------------------------
# Stage 3: the gate
# --------------------------------------------------------------------------------------


def _entities(rec: dict) -> dict:
    ents = {"query": rec["query"]}
    for a in rec["state_matches"] + rec["condition_matches"]:
        ents.setdefault(a["date"], a)
    return ents


def gate(claims: list[Claim], rec: dict) -> tuple[list, list]:
    """Field presence in the claim's OWN entity, plus a grounded span for mechanism claims."""
    ents = _entities(rec)
    kept, dropped = [], []
    for c in claims:
        who = c.about.strip()
        ent = ents.get(who)
        if ent is None:
            dropped.append((c, f"'{who}' is not an entity in the record"))
            continue
        key = c.field.split(".")[-1].strip("[]")
        blk = ent.get("condition_record") if who != "query" else ent
        present = key in ent and ent[key] is not None
        if not present and isinstance(blk, dict):
            present = key in blk and blk[key] is not None
        if not present:
            dropped.append((c, f"field '{key}' is absent or null for {who}"))
            continue
        if key in CONDITION_FIELDS:
            src = (blk or {}).get("conditions") or []
            hay = " \n ".join(f"{x['condition']} {x['quote']}" for x in src)
            if not c.quote or not quote_is_grounded(c.quote, hay):
                dropped.append((c, f"mechanism claim about {who} carries no grounded span"))
                continue
        kept.append(c)
    return kept, dropped


def planted_control(rec: dict) -> dict:
    """Arm A's positive control: six claims, three answerable and three not. Deterministic.

    An unfired gate demonstrates nothing, which this project has said before and is why the
    control exists. Three claims cite a condition field on an episode that HAS a record, with a
    span lifted verbatim from it; three cite the same field on episodes that do not.
    """
    covered = [a for a in rec["condition_matches"] if a["condition_record"]][:3]
    uncovered = [a for a in rec["state_matches"] if not a["condition_record"]][:3]
    claims, expect = [], []
    for a in covered:
        q = a["condition_record"]["conditions"][0]
        claims.append(Claim(about=a["date"], field="conditions", quote=q["quote"][:90],
                            claim=f"{a['date']} was priced on {q['condition']}."))
        expect.append(True)
    for a in uncovered:
        claims.append(Claim(about=a["date"], field="conditions",
                            quote="the resolution of the credit crisis",
                            claim=f"{a['date']} was priced on the credit crisis resolving."))
        expect.append(False)
    if not claims:
        return {"planted": 0, "note": "no covered/uncovered split available at this query date"}
    kept, dropped = gate(claims, rec)
    kept_set = {(c.about, c.claim) for c in kept}
    got = [(c.about, c.claim) in kept_set for c in claims]
    return {
        "planted": len(claims),
        "expected_kept": int(sum(expect)),
        "actually_kept": int(sum(got)),
        "correct": bool(got == expect),
        "detail": [{"about": c.about, "expected": e, "kept": g, "claim": c.claim[:70]}
                   for c, e, g in zip(claims, expect, got)],
        "drop_reasons": [r for _, r in dropped],
    }


# --------------------------------------------------------------------------------------


def ask(client, system: str, payload: dict, fmt):
    r = client.messages.parse(
        model=MODEL, max_tokens=1100, system=system,
        messages=[{"role": "user", "content": json.dumps(payload, indent=1)[:120_000]}],
        output_format=fmt)
    return r.parsed_output


BASE = (
    "You are given one query month and two lists of historical months: the nearest by STATE, "
    "returned by a numerical retrieval engine, and the nearest by the CONDITIONS the loser leg's "
    "filings stated, returned by a text encoder whose training cutoff precedes the query date.\n\n"
    "Rules you must follow exactly:\n"
    "1. Every claim names the entity it is about in `about`: either 'query' or one analogue's "
    "date string. A claim about a month you do not name is dropped.\n"
    "2. Every claim cites a `field` that exists inside THAT entity's own record.\n"
    "3. A claim about what a month was priced on must set `field` to 'conditions' AND carry a "
    "VERBATIM span in `quote`, copied from that entity's own stored quotes. If a month's "
    "condition_record_status is NO RECORD, you cannot make such a claim about it at all — say so "
    "instead, citing condition_record_status.\n"
    "4. Never mention any outcome for the query month. It is not in the record and was not "
    "knowable."
)


def run_panel(rec: dict) -> dict:
    import anthropic
    client = anthropic.Anthropic()

    pro = ask(client, BASE + " Argue that the nearest STATE match is a useful analogue.",
              rec, Side)
    con = ask(client, BASE + " Argue where the nearest STATE match FAILS. Be specific about the "
                             "dimension, and about what the record does and does not let you "
                             "say.", rec, Side)
    pk, pdrop = gate(pro.claims, rec)
    ck, cdrop = gate(con.claims, rec)
    adj = ask(client, BASE + " You are given both sides' SURVIVING claims. Decide whether the "
                             "nearest state match is a usable analogue, and whether mechanism "
                             "can be ruled on from the record at all.",
              {**rec, "proponent_claims": [c.model_dump() for c in pk],
               "dissent_claims": [c.model_dump() for c in ck]}, Verdict)
    return {
        "proponent": {"kept": [c.model_dump() for c in pk],
                      "dropped": [{**c.model_dump(), "reason": r} for c, r in pdrop]},
        "dissent": {"kept": [c.model_dump() for c in ck],
                    "dropped": [{**c.model_dump(), "reason": r} for c, r in cdrop]},
        "verdict": adj.model_dump(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2020-10-31")
    ap.add_argument("--control-only", action="store_true",
                    help="arm A's planted control; deterministic, no model call")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    corpus = RAG.build_corpus()
    q = pd.Timestamp(args.date)
    rec = build_record(q, corpus)

    print(f"query {rec['query']['date']}  "
          f"{rec['query']['n_records']} grounded conditions across {rec['query']['n_names']} names")
    print(f"encoder {rec['encoder']}   {rec['n_candidates_after_embargo']} candidates after embargo")
    print("\nSTAGE 1  nearest by STATE")
    for a in rec["state_matches"]:
        print(f"  #{a['rank']} {a['date']}  d={a['distance']}  fwd21d={a['forward_21d_return']:+.2%}"
              f"   conditions: {a['condition_record_status']}")
    print("\nSTAGE 2  nearest by WHAT THE LOSER LEG WAS PRICED ON")
    for a in rec["condition_matches"]:
        print(f"  #{a['rank']} {a['date']}  sim={a['condition_similarity']}  "
              f"fwd21d={a['forward_21d_return']:+.2%}   "
              f"{a['condition_record']['n_records']} records")

    ctl = planted_control(rec)
    print(f"\nARM A  planted control: {ctl.get('actually_kept')} kept of {ctl.get('planted')}, "
          f"expected {ctl.get('expected_kept')}  ->  "
          f"{'CORRECT' if ctl.get('correct') else 'INCORRECT'}")
    for d in ctl.get("detail", []):
        print(f"    {d['about']}  expected {'keep' if d['expected'] else 'drop'}, "
              f"got {'keep' if d['kept'] else 'drop'}")

    out = {"record": rec, "planted_control": ctl}
    if not args.control_only:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("no ANTHROPIC_API_KEY; rerun with --control-only for the gate check")
        panel = run_panel(rec)
        out.update(panel)
        print(f"\nSTAGE 3  proponent {len(panel['proponent']['kept'])} kept / "
              f"{len(panel['proponent']['dropped'])} dropped;  "
              f"dissent {len(panel['dissent']['kept'])} kept / "
              f"{len(panel['dissent']['dropped'])} dropped")
        for side in ("proponent", "dissent"):
            for c in panel[side]["kept"]:
                print(f"    [{side}] {c['about']}.{c['field']}: {c['claim'][:100]}")
            for c in panel[side]["dropped"]:
                print(f"    [{side} DROPPED] {c['about']}.{c['field']}: {c['reason']}")
        v = panel["verdict"]
        print(f"\nSTAGE 4  state_match_is_apt={v['state_match_is_apt']}  "
              f"mechanism_is_rulable={v['mechanism_is_rulable']}")
        if v["fails_on"]:
            print(f"  fails on: {v['fails_on']}")
        print(f"  {v['reasoning']}")

    (OUT / f"panel_{rec['query']['date']}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
