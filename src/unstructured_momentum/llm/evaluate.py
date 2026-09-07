"""Measure whether the LLM extractor actually beats the keyword baseline.

The brief asks whether AI *materially improves* the process. That is a claim about a
comparison, so this module holds the comparison rather than leaving it as an assertion.

Ground truth
------------
Labels live in a CSV under ``data/labels/`` with the headline, source and label visible on
every row, so a reviewer can audit or correct any judgment without reading code. The
labelling rubric is deliberately narrow and is stated in `RUBRIC` below.

**Provenance caveat, stated plainly:** the seed labels were produced by the same
assistant that wrote the extractor prompt. That is a real circularity, and the mitigation
is that every label is human-readable and correctable in place. Any result computed from
unreviewed labels should be reported as provisional. Spot-checking a stratified sample is
a few minutes of work and converts this from provisional to defensible.

Metrics
-------
Precision, recall and F1 on the *relevance* judgment, plus the share of retained evidence
that is templated. Precision matters most here: a crowding feature built from a corpus
that is 85% filler is measuring a publishing schedule, and recall on genuinely-absent
signal is not meaningful.
"""

from __future__ import annotations

import pandas as pd

from ..config import DATA

RUBRIC = """\
A headline is RELEVANT only if it substantively concerns positioning, crowding,
deleveraging, or leadership rotation in US EQUITY FACTORS -- momentum, value, size,
quality, low-vol -- or in the levered market-neutral complex that trades them.

NOT relevant:
  - templated / auto-generated shareholder-composition or screener content
  - macro positioning that is not about equity factors (bond flows, gold, rates,
    recession sentiment, FX) even when it uses the word "crowded"
  - single-company news, including single-stock short interest, with no factor angle
  - generic market wrap-ups mentioning a keyword in passing
"""

LABEL_DIR = DATA / "labels"


def label_path(name: str = "sept2019_crowding"):
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    return LABEL_DIR / f"{name}.csv"


def make_template(articles: pd.DataFrame, name: str = "sept2019_crowding") -> pd.DataFrame:
    """Write a labelling template with one row per article, for human review."""
    cols = [c for c in ["seendate", "domain", "title", "url", "source_tier"] if c in articles]
    df = articles[cols].copy()
    df["is_relevant_label"] = pd.NA
    df["reviewed_by_human"] = False
    df["note"] = ""
    df.to_csv(label_path(name), index=False)
    return df


def load_labels(name: str = "sept2019_crowding") -> pd.DataFrame:
    p = label_path(name)
    if not p.exists():
        raise FileNotFoundError(f"No label file at {p}. Run make_template() first.")
    df = pd.read_csv(p)
    df["is_relevant_label"] = df["is_relevant_label"].astype("boolean")
    return df.dropna(subset=["is_relevant_label"])


def score(predictions: pd.DataFrame, labels: pd.DataFrame, *, key: str = "url") -> pd.Series:
    """Precision / recall / F1 of a predicted relevance flag against labels."""
    merged = predictions.merge(labels[[key, "is_relevant_label"]], on=key, how="inner")
    if not len(merged):
        return pd.Series(dtype=float)

    pred = merged["is_relevant"].astype(bool)
    truth = merged["is_relevant_label"].astype(bool)

    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else float("nan")
    )
    return pd.Series(
        {
            "n_scored": len(merged),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_true_relevant": int(truth.sum()),
            "base_rate": float(truth.mean()),
            "human_reviewed_share": float(
                labels.get("reviewed_by_human", pd.Series(dtype=bool)).fillna(False).mean()
            )
            if "reviewed_by_human" in labels
            else 0.0,
        }
    )


def compare(
    baseline_preds: pd.DataFrame,
    llm_preds: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Score both extractors side by side."""
    rows = {}
    if len(baseline_preds):
        rows["keyword_baseline"] = score(baseline_preds, labels)
    if len(llm_preds):
        rows["claude"] = score(llm_preds, labels)
    return pd.DataFrame(rows).T


def verdict(comparison: pd.DataFrame) -> str:
    """State the comparison honestly, including when the corpus has no signal to find."""
    if "keyword_baseline" not in comparison.index:
        return "No baseline scored."

    base = comparison.loc["keyword_baseline"]
    if base["n_true_relevant"] == 0:
        return (
            f"Of {int(base['n_scored'])} labelled articles, ZERO are genuine equity-factor "
            f"positioning evidence. The keyword baseline marks "
            f"{int(base['tp'] + base['fp'])} as relevant -- all false positives. "
            "Precision and recall are undefined because the corpus contains no signal to "
            "find. The correct conclusion is about the corpus, not the extractor: this "
            "text source carries no factor-crowding evidence in this window, and a better "
            "classifier cannot manufacture what is absent."
        )
    if "claude" not in comparison.index:
        return (
            f"Baseline precision {base['precision']:.2f} on {int(base['n_scored'])} "
            "labelled articles. LLM extractor not yet run (needs credentials)."
        )

    llm = comparison.loc["claude"]
    delta = llm["precision"] - base["precision"]
    direction = "improves on" if delta > 0 else "does not beat"
    return (
        f"On {int(base['n_scored'])} labelled articles (base rate "
        f"{base['base_rate']:.1%}): baseline precision {base['precision']:.2f}, "
        f"Claude precision {llm['precision']:.2f} (recall {llm['recall']:.2f}). "
        f"The LLM {direction} the keyword baseline by {abs(delta):.2f} on precision."
    )
