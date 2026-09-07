"""Prediction-powered inference: honest intervals on quantities built from model labels.

The problem this solves is specific. A model labels twenty thousand documents; a human labels
ninety. Using the model labels alone gives a tight interval around a possibly biased number.
Using only the gold labels gives an unbiased number with an interval too wide to act on. PPI
takes the model's estimate over everything and CORRECTS it by the measured model-minus-gold gap
on the labelled subset, carrying that correction's own uncertainty into the interval:

    theta_ppi = theta_model(all) - (theta_model(labelled) - theta_gold(labelled))

The correction term is what makes it honest. If the model is unbiased on the gold set the
correction is zero and the interval is nearly the model-only one; if the model is badly biased
the correction is large and the interval widens to say so.

Two disciplines this project has to keep
----------------------------------------
**Gold sets are task-matched.** The ninety hand-labelled records here answer whether an article
is ABOUT a holding. They rectify aboutness quantities and nothing else. An exposure aggregate
needs exposure labels; borrowing a gold set across tasks silently assumes the model's bias is
the same on both, which is the assumption PPI exists to avoid making.

**A model-labelled set is not a gold set.** `data/labels/sept2019_crowding.csv` carries 246 rows
of which zero are human-reviewed, and its own module records that the seed labels came from the
same assistant that wrote the extractor prompt. It is banned as gold here in code, not in prose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BANNED_GOLD = ("sept2019_crowding",)


def _z(alpha: float) -> float:
    """Two-sided normal quantile. Table lookup for the usual levels, erfinv otherwise.

    An earlier version drew a million standard normals and took a percentile, which is both
    slower and less accurate than the closed form it was approximating.
    """
    table = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}
    if alpha in table:
        return table[alpha]
    from math import erfinv, sqrt
    return sqrt(2.0) * erfinv(1.0 - alpha)


class NotGold(ValueError):
    """Raised when a label set that was never human-reviewed is offered as ground truth."""


def assert_is_gold(df: pd.DataFrame, *, name: str, reviewed_col: str = "reviewed_by_human") -> None:
    """A gold set must be human-reviewed, and the check is code rather than a convention."""
    if any(b in name for b in BANNED_GOLD):
        raise NotGold(
            f"{name!r} is on the banned list: its labels were produced by the same assistant "
            "that wrote the extractor prompt, so scoring against it is circular.")
    if reviewed_col in df.columns:
        n = int(pd.Series(df[reviewed_col]).fillna(False).astype(bool).sum())
        if n == 0:
            raise NotGold(f"{name!r} has no human-reviewed rows; it is not a gold set.")


def rectified_mean(model_all: np.ndarray, model_labelled: np.ndarray,
                   gold_labelled: np.ndarray, *, alpha: float = 0.05) -> dict:
    """PPI point estimate and interval for a mean.

    The interval combines the model estimate's variance over the whole corpus with the
    correction term's variance over the labelled subset. It is wider than the model-only
    interval by exactly the amount the correction is uncertain, which is the point.
    """
    model_all = np.asarray(model_all, dtype=float)
    model_labelled = np.asarray(model_labelled, dtype=float)
    gold_labelled = np.asarray(gold_labelled, dtype=float)
    if len(model_labelled) != len(gold_labelled):
        raise ValueError("model and gold labels must be paired on the same rows")
    n, N = len(gold_labelled), len(model_all)
    if n < 10:
        raise ValueError(f"only {n} gold labels; the correction term would be noise")

    rect = model_labelled - gold_labelled
    theta = float(model_all.mean() - rect.mean())
    var = model_all.var(ddof=1) / N + rect.var(ddof=1) / n
    se = float(np.sqrt(var))
    z = _z(alpha)
    return {
        "theta_ppi": theta,
        "se": se,
        "lo": theta - z * se,
        "hi": theta + z * se,
        "theta_model_only": float(model_all.mean()),
        "theta_gold_only": float(gold_labelled.mean()),
        "correction": float(rect.mean()),
        "n_gold": n,
        "n_model": N,
        #: How much of the model-only estimate was bias. Large means the model labels alone
        #: would have been misleading, which is the case worth reporting loudly.
        "bias_share": (float(abs(rect.mean()) / abs(model_all.mean()))
                       if model_all.mean() else float("nan")),
    }


def rectified_proportion(model_all: np.ndarray, model_labelled: np.ndarray,
                         gold_labelled: np.ndarray, *, alpha: float = 0.05) -> dict:
    """The same estimator for a 0/1 label, which is what a classifier produces."""
    return rectified_mean(np.asarray(model_all, dtype=float),
                          np.asarray(model_labelled, dtype=float),
                          np.asarray(gold_labelled, dtype=float), alpha=alpha)


def width_gain(res: dict, alpha: float = 0.05) -> dict:
    """What PPI bought against using the gold labels alone.

    Reported because the honest alternative to a corrected model estimate is not the raw model
    estimate -- it is throwing the model away and using only the humans. If PPI is not narrower
    than that, it has bought nothing.
    """
    n = res["n_gold"]
    se_gold = float(np.sqrt(max(res["theta_gold_only"] * (1 - res["theta_gold_only"]), 1e-9) / n))
    return {"se_ppi": res["se"], "se_gold_only": se_gold,
            "narrower_than_gold_only": res["se"] < se_gold,
            "ratio": res["se"] / se_gold if se_gold else float("nan")}


def load_gold(path: str | Path, *, name: str | None = None) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    assert_is_gold(df, name=name or p.stem)
    return df
