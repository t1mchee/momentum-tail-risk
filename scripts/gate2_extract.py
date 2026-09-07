"""Gate 2 -- extract the external condition each loser-leg name's survival depends on.

The claim this serves: returns and holdings identify the loser leg's exposure and its
concentration, but they cannot say what condition three hundred distressed businesses SHARE.
Filings can. This is the seat text is asked to earn.

Discipline, all of it enforced here rather than asserted:
  * schema-constrained, so the model cannot return a number or a probability
  * extractive only -- every record carries a verbatim span, checked against the source
  * a mandatory decline path, so "no stated condition" is an answer rather than a guess
  * acceptance-time gating upstream, so nothing read here postdates the formation date
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.data import eightk  # noqa: E402
from unstructured_momentum.llm.exposure import quote_is_grounded  # noqa: E402

MODEL = "claude-sonnet-4-5-20250929"
OUT = Path("reports/gate2")


class SurvivalCondition(BaseModel):
    """One filing's reading. No numeric field exists, so no number can be returned."""

    states_a_condition: bool = Field(
        description="True only if the filing states an EXTERNAL condition -- outside the "
                    "company's control -- on which its recovery or survival depends.")
    condition: str = Field(
        default="",
        description="The external condition, as a short noun phrase in the filing's own "
                    "vocabulary. Empty when states_a_condition is false.")
    quote: str = Field(
        default="",
        description="A VERBATIM span copied from the filing that states the condition. Must "
                    "appear character-for-character in the source. Empty when false.")
    decline_reason: str = Field(
        default="",
        description="When states_a_condition is false, why: routine disclosure, purely "
                    "internal matter, boilerplate risk language, or no substantive content.")


SYSTEM = (
    "You read one 8-K filing and report the EXTERNAL condition on which the filer's recovery "
    "or survival is stated to depend -- a rate path, a regulatory decision, a commodity price, "
    "demand recovery, a clinical or trial readout, access to funding.\n\n"
    "Rules:\n"
    "1. Extract, never infer. If the filing does not state such a condition, set "
    "states_a_condition to false and say why. Declining is a correct answer and is expected "
    "for most routine filings.\n"
    "2. Every condition must carry a VERBATIM quote from the filing. Copy it exactly.\n"
    "3. Boilerplate forward-looking-statement risk factors are NOT a stated condition.\n"
    "4. Report no numbers, probabilities or judgements of severity."
)


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("no ANTHROPIC_API_KEY in the environment")
    import anthropic

    leg = sys.argv[1] if len(sys.argv) > 1 else "loser"
    # A CONTROL date: same construction, same leg, a month with no registered episode near it.
    # Without this the 17 percent rate at the target date is uninterpretable -- it could be a
    # permanent property of distressed names rather than anything about that month.
    ctrl = sys.argv[2] if len(sys.argv) > 2 else None
    if ctrl:
        import pickle
        from unstructured_momentum.data import eightk as _e
        d0 = pd.Timestamp(ctrl)
        legs = pickle.load(open("data/processed/leg_members.pkl", "rb"))
        tick = set(legs[d0]["losers" if leg == "loser" else "winners"])
        ix = _e.load_index()
        acc = pd.to_datetime(ix["accepted_at"], errors="coerce", utc=True)
        m = ((acc >= pd.Timestamp(d0 - pd.Timedelta(days=91), tz="UTC")) &
             (acc <= pd.Timestamp(d0, tz="UTC") + pd.Timedelta(hours=23)) &
             ix["ticker"].isin(tick))
        lf = ix[m].copy(); lf["accepted_at"] = acc[m]
        leg = f"{leg}_{ctrl}"
    elif leg == "winner":
        import pickle
        from unstructured_momentum.data import eightk as _e
        legs = pickle.load(open("data/processed/leg_members.pkl", "rb"))
        tick = set(legs[pd.Timestamp("2020-10-31")]["winners"])
        ix = _e.load_index()
        acc = pd.to_datetime(ix["accepted_at"], errors="coerce", utc=True)
        m = ((acc >= pd.Timestamp("2020-08-01", tz="UTC")) &
             (acc <= pd.Timestamp("2020-10-31 23:59", tz="UTC")) & ix["ticker"].isin(tick))
        lf = ix[m].copy(); lf["accepted_at"] = acc[m]
    else:
        lf = pd.read_parquet(OUT / "nov2020_loser_filings.parquet")
    print(f"{len(lf)} loser-leg filings across {lf['ticker'].nunique()} names")

    client = anthropic.Anthropic()
    rows = []
    for i, (_, f) in enumerate(lf.iterrows(), 1):
        try:
            body = eightk.load_body(f["accession"])
            text = body.get("text") or body.get("body") or ""
        except Exception as exc:  # noqa: BLE001
            rows.append({"ticker": f["ticker"], "accession": f["accession"],
                         "error": f"{type(exc).__name__}"})
            continue
        if len(text) < 200:
            rows.append({"ticker": f["ticker"], "accession": f["accession"],
                         "error": "body too short"})
            continue
        excerpt = text[:18000]
        try:
            r = client.messages.parse(
                model=MODEL, max_tokens=700, system=SYSTEM,
                messages=[{"role": "user",
                           "content": f"Filing items: {f.get('items','')}\n\n{excerpt}"}],
                output_format=SurvivalCondition,
            )
            rec = r.parsed_output
        except Exception as exc:  # noqa: BLE001
            rows.append({"ticker": f["ticker"], "accession": f["accession"],
                         "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
            continue

        grounded = bool(rec.quote) and quote_is_grounded(rec.quote, excerpt)
        rows.append({
            "ticker": f["ticker"], "accession": f["accession"],
            "accepted_at": str(f["accepted_at"]), "items": f.get("items", ""),
            "states_a_condition": rec.states_a_condition,
            "condition": rec.condition, "quote": rec.quote,
            "decline_reason": rec.decline_reason,
            "quote_grounded": grounded,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(lf)}", flush=True)

    D = pd.DataFrame(rows)
    D.to_parquet(OUT / f"nov2020_extractions_{leg}.parquet")
    ok = D[D.get("error").isna()] if "error" in D else D
    stated = ok[ok.states_a_condition.fillna(False)]
    print(f"\nfilings read      {len(ok)}   errors {len(D)-len(ok)}")
    print(f"states a condition {len(stated)}  ({len(stated)/max(len(ok),1):.1%})")
    print(f"DECLINED           {len(ok)-len(stated)}  -- the mandatory decline path")
    if len(stated):
        g = stated.quote_grounded.sum()
        print(f"quote grounded     {g} of {len(stated)}  ({g/len(stated):.1%}); "
              f"ungrounded records are DROPPED, not repaired")
        print(f"distinct names with a condition: {stated.ticker.nunique()}")
        print("\nsample conditions:")
        for _, r in stated[stated.quote_grounded].head(6).iterrows():
            print(f"  {r.ticker:6s} {r.condition[:72]}")
    print(f"\nwrote {OUT}/nov2020_extractions_{leg}.parquet")


if __name__ == "__main__":
    main()
