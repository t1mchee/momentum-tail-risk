"""exp-082 -- the two-sided analogue: past episodes get their own condition records.

Registered at commit 71be4e8ba, before this file existed.

The committed panel artifact shows the asymmetry plainly. The query side carries nine extracted
conditions with verbatim spans; every retrieved neighbour carries a date, a distance, a block
closeness and a forward return, and nothing about what it was priced ON. So when the adjudicator
ruled the 2009 match not apt because every retrieved state is a credit-crisis recovery, it
compared the query's RECORD against the model's MEMORY. The field gate could not catch that
because the field it cited was real.

This builds the other side. 419 grounded condition records across 28 formation dates already
exist, each one a span the extractor's quote gate found character-for-character in its source,
so the corpus is pre-verified rather than newly trusted.

Two encoders read it, and that is register row 09 rather than decoration:

  * a **chronologically consistent** checkpoint whose training cutoff precedes the QUERY date,
    so it cannot know what followed the conditions it is reading;
  * a **contemporary** sentence encoder that has seen everything.

Their rank agreement bounds how much of the ranking is reading and how much is recognition. This
project has already measured that stripping dates does not hide the date -- a model placed the
exact year in 40 of 40 anonymised cases -- so an unconstrained encoder is not a neutral
instrument and is not treated as one.

One subtlety about centring, because the encoder module warns about exactly this. Centring is a
per-cross-section operation and centring over pooled dates would carry future documents into a
past measure. Here the pool is the EMBARGOED candidate set at one query date: every candidate's
outcome window closed at least a year before the query, so every document in the pool is strictly
past and one checkpoint covers all of them. Centring over that pool is point-in-time.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unstructured_momentum.embed import chrono  # noqa: E402

GATE2 = Path("reports/gate2")
OUT = Path("reports/analogue_rag")
MPNET = "sentence-transformers/all-mpnet-base-v2"
EMBARGO_DAYS = 365          # the analogue engine's own embargo, applied to documents
MAX_LEN = 256               # a condition and its span; not a filing


# --------------------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------------------


def build_corpus() -> pd.DataFrame:
    """Every grounded condition record on disk, one row per filing, with its formation date.

    Grounded means the extractor's quote gate found the span in the source. Ungrounded records
    were dropped at extraction and are not resurrected here: a corpus assembled from records
    that failed their own gate would be a worse instrument than no corpus.
    """
    rows = []
    files = sorted(glob.glob(str(GATE2 / "nov2020_extractions_loser_*.parquet")))
    files.append(str(GATE2 / "nov2020_extractions.parquet"))   # the flagship, undated in its name
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        stem = p.stem.split("_")[-1]
        formation = "2020-10-31" if stem == "extractions" else stem
        d = pd.read_parquet(p)
        if "states_a_condition" not in d:
            continue
        ok = d[d.get("error").isna()] if "error" in d else d
        g = ok[ok["states_a_condition"].fillna(False) & ok["quote_grounded"].fillna(False)]
        for _, r in g.iterrows():
            rows.append({
                "formation": pd.Timestamp(formation),
                "ticker": r["ticker"],
                "accession": r["accession"],
                "condition": str(r["condition"]),
                "quote": str(r["quote"]),
            })
    c = pd.DataFrame(rows).drop_duplicates(subset=["formation", "accession"])
    # One text per record: the condition names the bet, the span is the evidence for it.
    c["text"] = c["condition"].str.strip() + ". " + c["quote"].str.strip()
    return c.sort_values(["formation", "ticker"]).reset_index(drop=True)


def candidates(corpus: pd.DataFrame, query: pd.Timestamp) -> list[pd.Timestamp]:
    """Formation dates eligible as analogues for this query, under the engine's own embargo."""
    cut = query - pd.Timedelta(days=EMBARGO_DAYS)
    return sorted({d for d in corpus["formation"].unique() if pd.Timestamp(d) <= cut})


# --------------------------------------------------------------------------------------
# The two encoders
# --------------------------------------------------------------------------------------


def embed_chrono(texts: list[str], query: pd.Timestamp) -> tuple[np.ndarray, str]:
    """The date-constrained reader. One checkpoint for the whole cross-section, by the query."""
    repo = chrono.checkpoint_for(query)
    E = chrono.ChronoEmbedder(max_length=MAX_LEN).embed(texts, query)
    return E, repo


_ST = {}


def embed_mpnet(texts: list[str]) -> tuple[np.ndarray, str]:
    """The contemporary reader. Has seen everything, which is the point of the contrast."""
    from sentence_transformers import SentenceTransformer
    if MPNET not in _ST:
        _ST[MPNET] = SentenceTransformer(MPNET)
    E = _ST[MPNET].encode(texts, batch_size=32, show_progress_bar=False,
                          convert_to_numpy=True).astype(np.float32)
    return E, MPNET


def episode_vectors(Ec: np.ndarray, formations: pd.Series) -> dict:
    """Mean of an episode's centred record vectors, re-normalised. Records stay addressable."""
    out = {}
    for f in formations.unique():
        m = (formations == f).to_numpy()
        v = Ec[m].mean(axis=0)
        n = np.linalg.norm(v)
        out[pd.Timestamp(f)] = v / (n if n else 1.0)
    return out


# --------------------------------------------------------------------------------------
# Ranking, and the agreement between the two readings
# --------------------------------------------------------------------------------------


def rank_one(corpus: pd.DataFrame, query: pd.Timestamp, encoder: str) -> dict:
    """Rank every eligible candidate episode by the closeness of what its loser leg was priced on.

    The pool embedded is the query's own records plus every candidate's, so the centring
    reference is the pool the comparison is made over.
    """
    cands = candidates(corpus, query)
    pool = corpus[corpus["formation"].isin([*cands, query])].reset_index(drop=True)
    if not len(pool) or query not in set(pool["formation"]):
        return {"encoder": encoder, "query": str(query.date()), "n_candidates": len(cands),
                "ranking": [], "note": "query date has no grounded condition record"}

    texts = pool["text"].tolist()
    if encoder == "chrono":
        E, tag = embed_chrono(texts, query)
    else:
        E, tag = embed_mpnet(texts)

    Ec = chrono.center(E)                       # centred on the embargoed pool: all strictly past
    vecs = episode_vectors(Ec, pool["formation"])
    q = vecs[query]
    sims = {d: float(q @ vecs[d]) for d in cands if d in vecs}
    order = sorted(sims, key=lambda d: -sims[d])
    return {
        "encoder": encoder,
        "encoder_id": tag,
        "query": str(query.date()),
        "n_candidates": len(cands),
        "n_records_in_pool": int(len(pool)),
        "ranking": [{"formation": str(d.date()), "similarity": round(sims[d], 4),
                     "rank": i + 1, "n_records": int((pool["formation"] == d).sum())}
                    for i, d in enumerate(order)],
    }


def agreement(a: dict, b: dict) -> dict:
    """Register row 09: Spearman between the two readings, read two-sided against 0.80."""
    from scipy.stats import spearmanr
    ra = {r["formation"]: r["rank"] for r in a["ranking"]}
    rb = {r["formation"]: r["rank"] for r in b["ranking"]}
    common = sorted(set(ra) & set(rb))
    if len(common) < 4:
        return {"n": len(common), "spearman": None, "verdict": "too few candidates to rank"}
    rho, p = spearmanr([ra[d] for d in common], [rb[d] for d in common])
    top1 = a["ranking"][0]["formation"] == b["ranking"][0]["formation"] if a["ranking"] else None
    if rho < 0.80:
        v = ("below the registered 0.80: the contemporary encoder ranks on something the "
             "constrained one cannot have, so the constrained encoder's ranking is the one "
             "that reaches the page and the gap is printed")
    elif rho > 0.98:
        v = ("above 0.98: the constrained reader adds nothing the cheap one does not, and "
             "under the twin rule the seat is idle and retires")
    else:
        v = ("inside the band: the agreement bounds how much of the ranking is recognition, "
             "both are reported, and the constrained encoder's ranking is the one used")
    return {"n": len(common), "spearman": round(float(rho), 4), "p": float(p),
            "same_top_match": bool(top1), "registered_threshold": 0.80, "verdict": v}


# --------------------------------------------------------------------------------------


def run(query: pd.Timestamp, corpus: pd.DataFrame) -> dict:
    a = rank_one(corpus, query, "chrono")
    b = rank_one(corpus, query, "mpnet")
    return {"query": str(query.date()),
            "checkpoint": a.get("encoder_id"),
            "constrained": a, "contemporary": b,
            "row09_agreement": agreement(a, b) if a["ranking"] and b["ranking"] else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2020-10-31")
    ap.add_argument("--all", action="store_true", help="every formation date with a record")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    corpus.to_parquet(OUT / "condition_corpus.parquet")
    print(f"corpus: {len(corpus)} grounded records across "
          f"{corpus['formation'].nunique()} formation dates, "
          f"{corpus['formation'].min().date()} to {corpus['formation'].max().date()}")

    dates = sorted(corpus["formation"].unique()) if args.all else [pd.Timestamp(args.date)]
    results, rows = [], []
    for d in dates:
        d = pd.Timestamp(d)
        if len(candidates(corpus, d)) < 4:
            print(f"  {d.date()}: {len(candidates(corpus, d))} candidates after embargo, skipped")
            continue
        try:
            r = run(d, corpus)
        except Exception as exc:  # noqa: BLE001
            print(f"  {d.date()}: FAILED {type(exc).__name__}: {str(exc)[:90]}")
            continue
        results.append(r)
        ag = r["row09_agreement"] or {}
        rows.append({"query": r["query"], "checkpoint": r["checkpoint"],
                     "n_candidates": ag.get("n"), "spearman": ag.get("spearman"),
                     "same_top_match": ag.get("same_top_match")})
        print(f"  {d.date()}  ckpt={r['checkpoint'].split('-')[-1]}  "
              f"n={ag.get('n')}  rho={ag.get('spearman')}  top-match agrees={ag.get('same_top_match')}")

    if rows:
        t = pd.DataFrame(rows)
        t.to_csv(OUT / "row09_agreement.csv", index=False)
        rho = t["spearman"].dropna()
        summary = {
            "n_query_dates": int(len(t)),
            "median_spearman": float(rho.median()) if len(rho) else None,
            "min": float(rho.min()) if len(rho) else None,
            "max": float(rho.max()) if len(rho) else None,
            "share_at_or_above_0.80": float((rho >= 0.80).mean()) if len(rho) else None,
            "share_above_0.98": float((rho > 0.98).mean()) if len(rho) else None,
            "top_match_agreement": float(t["same_top_match"].mean()),
            "registered_threshold": 0.80,
        }
        (OUT / "row09_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n" + json.dumps(summary, indent=2))
    (OUT / "rankings.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
