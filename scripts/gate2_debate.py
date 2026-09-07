"""Gate 2 -- the analogue panel: a proponent, a dissent, and an adjudicator.

Figure 3 specifies a dissent role -- "where the match fails, cited fields only" -- and this is
that role built. The panel exists because retrieval returns the nearest states by construction
and has no way to say the nearest state is not a useful one. On 2020-10-31 that distinction is
the whole question: the five nearest states are crisis-recovery months whose forward returns
were mostly POSITIVE, and the realised outcome was sharply negative.

Two containment rules, both enforced in code:

  * FIELD CITATION. Every claim must name a field that exists in the record handed to the
    agent. A claim citing a field that is not there is dropped, not repaired -- the same gate
    the extractor uses for quotes, applied to numbers.
  * NO OUTCOME LEAKAGE. The agents see the analogues' forward returns, which were knowable at
    the query date, and never the query date's own forward return, which was not.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.analogue.retrieve import Engine  # noqa: E402
from unstructured_momentum.data import french  # noqa: E402

MODEL = "claude-sonnet-4-5-20250929"
AS_OF = pd.Timestamp("2020-10-31")
OUT = Path("reports/gate2")


class Claim(BaseModel):
    field: str = Field(description="The exact field name from the record this claim rests on.")
    claim: str = Field(description="One sentence. What that field shows about the match.")


class Side(BaseModel):
    claims: list[Claim] = Field(description="Two to four claims, each citing one field.")


class Verdict(BaseModel):
    match_is_apt: bool = Field(
        description="True if the retrieved state is a usable analogue for the query date.")
    fails_on: str = Field(
        default="",
        description="If not apt, the single dimension on which the match fails. Empty if apt.")
    reasoning: str = Field(description="Two sentences, referring only to cited fields.")


def build_record() -> dict:
    e = Engine("deep")
    ns = e.query(AS_OF, k=5)
    w = french.momentum()
    analogues = []
    for n in ns:
        fwd = w[w.index > n.date]
        analogues.append({
            "date": str(n.date.date()), "rank": n.rank,
            "distance": round(n.distance, 4),
            "block_closeness": {k: round(v, 3) for k, v in n.block_closeness.items()},
            "forward_21d_return": round(float((1 + fwd.iloc[:21]).prod() - 1), 4),
        })
    ext = pd.read_parquet(OUT / "nov2020_extractions.parquet")
    g = ext[ext.states_a_condition.fillna(False) & ext.quote_grounded.fillna(False)]
    win = pd.read_parquet(OUT / "nov2020_extractions_winner.parquet")
    gw = win[win.states_a_condition.fillna(False) & win.quote_grounded.fillna(False)]

    # The extracted conditions are handed over VERBATIM rather than summarised. An earlier
    # version passed a field called query_shared_condition holding the string "post-pandemic
    # demand recovery" -- which I wrote, not the pipeline -- and the adjudicator's verdict then
    # cited that field by name. The conclusion was right and the reasoning rested on my summary,
    # which is close to circular and is exactly what a reviewer pulls on. Every field below is
    # now either a count or a string the extractor produced.
    sec = _sectors()
    conds = [{"ticker": t, "condition": c}
             for t, c in g.groupby("ticker")["condition"].first().items()]
    return {
        "query_date": str(AS_OF.date()),
        "query_loser_filings_stating_a_condition": int(len(g)),
        "query_loser_filings_read": 112,
        "query_winner_filings_stating_a_condition": int(len(gw)),
        "query_winner_filings_read": 206,
        "query_extracted_conditions": conds,
        "query_condition_sectors": int(pd.Series([sec.get(t) for t in g.ticker.unique()])
                                       .dropna().nunique()),
        "query_loser_leg_gics_sectors": int(_leg_sector_count()),
        "analogues": analogues,
    }


def _sectors() -> dict:
    import glob
    sec = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        d = pd.read_parquet(f, columns=["as_of", "ticker", "sector"])
        d = d[d.as_of <= AS_OF]
        if len(d):
            sec.update(d.drop_duplicates("ticker", keep="last")
                       .set_index("ticker")["sector"].to_dict())
    return sec


def _leg_sector_count() -> int:
    import pickle
    legs = pickle.load(open("data/processed/leg_members.pkl", "rb"))
    sec = _sectors()
    return pd.Series([sec.get(t) for t in legs[AS_OF]["losers"]]).dropna().nunique()


def ask(client, role: str, system: str, record: dict, fmt):
    r = client.messages.parse(
        model=MODEL, max_tokens=900, system=system,
        messages=[{"role": "user", "content": json.dumps(record, indent=1)}],
        output_format=fmt)
    return r.parsed_output


def gate(side: Side, record: dict) -> tuple[list, list]:
    """Drop any claim whose cited field is not in the record. Not repaired -- dropped."""
    flat = set(record) | {f"analogues[].{k}" for k in record["analogues"][0]}
    flat |= set(record["analogues"][0]) | set(record["analogues"][0]["block_closeness"])
    flat |= {"ticker", "condition"}
    kept, dropped = [], []
    for c in side.claims:
        key = c.field.split(".")[-1].strip("[]")
        (kept if (key in flat or c.field in flat) else dropped).append(c)
    return kept, dropped


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("no ANTHROPIC_API_KEY")
    import anthropic

    rec = build_record()
    client = anthropic.Anthropic()
    print(f"query {rec['query_date']}   {len(rec['analogues'])} retrieved states")
    for a in rec["analogues"]:
        print(f"  #{a['rank']} {a['date']}  d={a['distance']}  fwd21d={a['forward_21d_return']:+.2%}")

    base = ("You are given a record describing one query month and the five nearest historical "
            "months returned by a retrieval engine, with what followed each. You may cite ONLY "
            "field names that appear in this record. Never invent a field. Never mention any "
            "outcome for the query month itself -- it is not in the record and is not knowable.")

    pro = ask(client, "proponent",
              base + " Argue that the nearest retrieved state IS a useful analogue.", rec, Side)
    con = ask(client, "dissent",
              base + " Argue where the match FAILS. Be specific about the dimension.", rec, Side)

    pk, pd_ = gate(pro, rec)
    ck, cd = gate(con, rec)
    print(f"\nPROPONENT  {len(pk)} claims kept, {len(pd_)} dropped by the field gate")
    for c in pk: print(f"   [{c.field}] {c.claim}")
    print(f"\nDISSENT    {len(ck)} claims kept, {len(cd)} dropped by the field gate")
    for c in ck: print(f"   [{c.field}] {c.claim}")

    adj = ask(client, "adjudicator",
              base + " You are given both sides' surviving claims. Decide whether the nearest "
                     "retrieved state is a usable analogue, and if not, name the single "
                     "dimension on which it fails.",
              {**rec, "proponent_claims": [c.model_dump() for c in pk],
               "dissent_claims": [c.model_dump() for c in ck]}, Verdict)

    print(f"\nADJUDICATOR: match_is_apt={adj.match_is_apt}")
    if adj.fails_on: print(f"  fails on: {adj.fails_on}")
    print(f"  {adj.reasoning}")

    json.dump({"record": rec,
               "proponent": {"kept": [c.model_dump() for c in pk],
                             "dropped": [c.model_dump() for c in pd_]},
               "dissent": {"kept": [c.model_dump() for c in ck],
                           "dropped": [c.model_dump() for c in cd]},
               "verdict": adj.model_dump()},
              open(OUT / "nov2020_debate.json", "w"), indent=2)
    print(f"\nwrote {OUT/'nov2020_debate.json'}")


if __name__ == "__main__":
    main()
