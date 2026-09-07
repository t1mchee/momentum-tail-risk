"""exp-088 -- ground truth that is constructed, not annotated.

Registered at commit b510f97, before this file existed.

Precision and recall for the extractor need labels produced independently of the model, and that
measurement is blocked on annotation resources this project does not have. So the truth is built
instead of bought.

A condition span the model already extracted from one filing is a real statement of an external
survival condition, in a real filer's words. Injected into a different filing that the model
DECLINED, on a different date and for a different company, it makes a document whose correct
answer is known exactly: a condition is present, and the span stating it is known character for
character. Run the other way with real forward-looking boilerplate -- which the extraction
instructions explicitly exclude -- it makes a document whose correct answer is that none is.

That measures three things a rate cannot. Whether the reader finds a condition that is definitely
there. Whether it points at the right sentence rather than merely raising a flag. And whether it
can tell a stated condition from the boilerplate that surrounds it in every filing ever written.

The keyword rule runs on identical documents, so the comparison that came back inconclusive
without truth is finally scored against some.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from unstructured_momentum.data import eightk  # noqa: E402
from unstructured_momentum.llm.exposure import quote_is_grounded  # noqa: E402

import poc_e2_keyword_vs_model as KW  # noqa: E402
from gate2_extract import MODEL, SYSTEM, SurvivalCondition  # noqa: E402

GATE2 = Path("reports/gate2")
OUT = Path("reports/poc")
EXCERPT = 18_000
N_PER_ARM = 120
OVERLAP_CHARS = 40          # a quote counts as localised if it overlaps the plant this much
CUTOFF_OLD = pd.Timestamp("2024-12-31")     # may be in the labelling model's training data
CUTOFF_NEW = pd.Timestamp("2025-09-30")     # certainly after its release
SEED = 20260904
SEC: dict = {}

#: Real forward-looking boilerplate, which the instructions name as NOT a stated condition.
#: Drawn from filing text rather than written here, and held fixed.
BOILERPLATE_PAT = re.compile(
    r"[^.]*forward-looking statements[^.]*\.", re.I)


def sectors() -> dict:
    """Ticker to GICS sector, from the holdings panel.

    Needed because the first version of this experiment drew plants at random and produced
    documents whose ground truth was invalid: a sentence about FDA approval of a nerve graft,
    injected into a cruise line's filing, is NOT that filer stating a condition on its own
    survival, and the extractor was right to decline it. Plants must be plausible for the host
    or the constructed truth is not truth.
    """
    sec = {}
    for f in sorted(glob.glob("data/raw/ishares/IWV/panel/IWV_*.parquet")):
        d = pd.read_parquet(f, columns=["ticker", "sector"])
        sec.update(d.drop_duplicates("ticker", keep="last")
                   .set_index("ticker")["sector"].to_dict())
    return sec


def corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Declined filings to host the plants, and grounded records to draw plants from."""
    frames = []
    for f in sorted(glob.glob(str(GATE2 / "nov2020_extractions_loser_*.parquet"))) + [
            str(GATE2 / "nov2020_extractions.parquet")]:
        p = Path(f)
        if not p.exists():
            continue
        stem = p.stem.split("_")[-1]
        date = "2020-10-31" if stem == "extractions" else stem
        d = pd.read_parquet(p)
        if "states_a_condition" not in d:
            continue
        if "error" in d.columns:
            d = d[d["error"].isna()]
        d = d.assign(formation=date)
        frames.append(d)
    all_ = pd.concat(frames, ignore_index=True)
    # states_a_condition is object dtype with True/False/NaN mixed, so `~` on it is a BITWISE
    # not that yields -1 rather than a boolean mask, and pandas then reads -1 as a column key.
    # Cast before negating.
    stated = all_["states_a_condition"].fillna(False).astype(bool)
    ground = all_["quote_grounded"].fillna(False).astype(bool)
    declined = all_[~stated]
    grounded = all_[stated & ground]
    grounded = grounded[grounded["quote"].str.len().between(80, 400)]
    return declined.reset_index(drop=True), grounded.reset_index(drop=True)


def inject(text: str, span: str) -> tuple[str, int]:
    """Insert the span at a sentence boundary near the 40th percentile of the excerpt."""
    cut = int(len(text) * 0.40)
    m = re.search(r"\.\s", text[cut:])
    at = cut + (m.end() if m else 0)
    out = text[:at] + " " + span.strip() + " " + text[at:]
    return out, at + 1


def localised(quote: str, span: str) -> bool:
    """Does the returned quote actually point at the planted sentence?"""
    if not quote:
        return False
    q, s = quote.strip().lower(), span.strip().lower()
    if q in s or s in q:
        return True
    # longest common run, cheaply
    for n in range(min(len(q), 200), OVERLAP_CHARS - 1, -10):
        for i in range(0, len(q) - n + 1, 10):
            if q[i:i + n] in s:
                return True
    return False


def read(client, text: str, items: str = "") -> SurvivalCondition | None:
    try:
        r = client.messages.parse(
            model=MODEL, max_tokens=700, system=SYSTEM,
            messages=[{"role": "user", "content": f"Filing items: {items}\n\n{text}"}],
            output_format=SurvivalCondition)
        return r.parsed_output
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("no ANTHROPIC_API_KEY")
    import anthropic

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    global SEC
    SEC = sectors()
    declined, grounded = corpus()
    print(f"{len(SEC):,} tickers with a sector label")
    print(f"{len(declined):,} declined filings to host plants; "
          f"{len(grounded):,} grounded records to draw from\n")

    # Collect real boilerplate sentences from declined filings, for the negative arm.
    boiler: list[str] = []
    for _, r in declined.sample(400, random_state=SEED).iterrows():
        try:
            b = eightk.load_body(r["accession"])
        except Exception:  # noqa: BLE001, PERF203
            continue
        t = (b.get("text") or b.get("body") or "")[:EXCERPT]
        for m in BOILERPLATE_PAT.finditer(t):
            s = m.group(0).strip()
            if 100 <= len(s) <= 400:
                boiler.append(s)
        if len(boiler) > 300:
            break
    boiler = list(dict.fromkeys(boiler))
    print(f"{len(boiler)} distinct real boilerplate sentences collected for the negative arm\n")

    client = anthropic.Anthropic()
    rows = []
    hosts = declined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # The control the first two runs lacked. Every host was DECLINED on the committed run, but
    # the reader is not deterministic, so the baseline for "does a plant make it flag" is not
    # zero -- it is whatever a fresh read of the SAME document does with nothing added. Without
    # this the recall number has no denominator, and it doubles as the run-to-run reliability
    # measurement the design defines and nothing had run.
    #
    # `hi` is reset PER ARM. It used to be initialised once above this loop, so the three arms
    # consumed disjoint slices of the shuffled host list -- measured overlap 3, 3 and 0 of 120
    # -- while this comment and the printed label both said "same hosts". The lift was therefore
    # a difference between independent samples rather than a paired one, and the control did
    # not control for the thing it names. Every arm now reads the same hosts in the same order.
    for arm in ("unplanted_reread", "planted_condition", "planted_boilerplate"):
        hi = 0
        done = 0
        while done < N_PER_ARM and hi < len(hosts):
            h = hosts.iloc[hi]; hi += 1
            try:
                body = eightk.load_body(h["accession"])
            except Exception:  # noqa: BLE001, PERF203
                continue
            text = (body.get("text") or body.get("body") or "")[:EXCERPT]
            if len(text) < 2000:
                continue
            if arm == "unplanted_reread":
                span, planted = "", text
            elif arm == "planted_condition":
                # SAME SECTOR, different company, different date: real filer language that is
                # plausible for this host. Without the sector match the plant is incongruous and
                # declining it is correct, which is what the first run measured by mistake.
                hs = SEC.get(h["ticker"])
                pool = grounded[(grounded.formation != h.formation)
                                & (grounded.ticker != h.ticker)
                                & (grounded.ticker.map(SEC) == hs)]
                if hs is None or not len(pool):
                    continue
                span = pool.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]["quote"]
            else:
                if not boiler:
                    continue
                span = rng.choice(boiler)
            if arm != "unplanted_reread":
                planted, at = inject(text, span)
                planted = planted[:EXCERPT]
                if span.strip()[:60].lower() not in planted.lower():
                    continue                  # the plant must be inside the window the reader sees
            rec = read(client, planted, body.get("items", ""))
            if rec is None:
                continue
            rows.append({
                "arm": arm, "ticker": h["ticker"], "accession": h["accession"],
                "formation": h["formation"], "span": span,
                "flagged": bool(rec.states_a_condition),
                "quote": rec.quote, "condition": rec.condition,
                "localised": localised(rec.quote, span) if rec.states_a_condition else False,
                "keyword_flagged": KW.keyword_states_condition(planted),
            })
            done += 1
            if done % 20 == 0:
                print(f"  {arm}: {done}/{N_PER_ARM}", flush=True)
    t = pd.DataFrame(rows)
    t["host_sector"] = t["ticker"].map(SEC)
    t.to_csv(OUT / "e4_planted.csv", index=False)

    ctl = t[t.arm == "unplanted_reread"]
    pos = t[t.arm == "planted_condition"]; neg = t[t.arm == "planted_boilerplate"]
    old = pos[pd.to_datetime(pos.formation) <= CUTOFF_OLD]
    new = pos[pd.to_datetime(pos.formation) >= CUTOFF_NEW]
    res = {
        "n_control": len(ctl), "n_positive": len(pos), "n_negative": len(neg),
        "control_unplanted_reread": {
            "flag_rate": float(ctl.flagged.mean()),
            "keyword_flag_rate": float(ctl.keyword_flagged.mean()),
            "note": ("every host was DECLINED on the committed run, so a fresh read that flags "
                     "is run-to-run variance; this is the baseline the plant lift is measured "
                     "against and the reliability number the design defines"),
        },
        "model": {
            "recall_on_planted_conditions": float(pos.flagged.mean()),
            "span_localisation": float(pos.localised.mean()),
            "specificity_on_planted_boilerplate": float(1 - neg.flagged.mean()),
        },
        "keyword": {
            "recall_on_planted_conditions": float(pos.keyword_flagged.mean()),
            "span_localisation": None,
            "specificity_on_planted_boilerplate": float(1 - neg.keyword_flagged.mean()),
        },
        "leakage_split": {
            "n_pre_cutoff": len(old), "n_post_release": len(new),
            "recall_pre_cutoff": float(old.flagged.mean()) if len(old) else None,
            "recall_post_release": float(new.flagged.mean()) if len(new) else None,
        },
    }
    (OUT / "e4_summary.json").write_text(json.dumps(res, indent=2))

    c = res["control_unplanted_reread"]
    print(f"\n  CONTROL, same hosts re-read with nothing added:")
    print(f"    model flags {c['flag_rate']:.1%} (committed run: 0% -- all hosts were declined)"
          f"   keyword {c['keyword_flag_rate']:.1%}")
    print(f"\n{'':34}{'model':>10}{'keyword rule':>15}")
    m, k = res["model"], res["keyword"]
    print(f"  {'recall on planted conditions':<32}{m['recall_on_planted_conditions']:>10.1%}"
          f"{k['recall_on_planted_conditions']:>15.1%}")
    print(f"  {'quotes the planted span':<32}{m['span_localisation']:>10.1%}"
          f"{'n/a':>15}")
    print(f"  {'specificity on boilerplate':<32}{m['specificity_on_planted_boilerplate']:>10.1%}"
          f"{k['specificity_on_planted_boilerplate']:>15.1%}")
    ls = res["leakage_split"]
    lift = res["model"]["recall_on_planted_conditions"] - c["flag_rate"]
    print(f"\n  plant LIFT over the control: {lift:+.1%}")
    print(f"\n  leakage: recall {ls['recall_pre_cutoff']} on {ls['n_pre_cutoff']} pre-cutoff hosts, "
          f"{ls['recall_post_release']} on {ls['n_post_release']} post-release")
    print(f"\nwrote {OUT}/e4_planted.csv and e4_summary.json")


if __name__ == "__main__":
    main()
