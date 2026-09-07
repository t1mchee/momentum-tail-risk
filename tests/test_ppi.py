"""Tests for prediction-powered inference.

The estimator is simple; the discipline around it is what these guard. A gold set that was
never reviewed by a human is not ground truth, and borrowing one task's gold for another task
silently assumes the model's bias transfers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unstructured_momentum.validation import ppi


def _synthetic(n=20000, n_gold=200, rate=0.30, err=0.10, seed=0):
    rng = np.random.default_rng(seed)
    truth = (rng.random(n) < rate)
    model = np.where(rng.random(n) < err, ~truth, truth).astype(float)
    idx = rng.choice(n, n_gold, replace=False)
    return model, model[idx], truth[idx].astype(float), rate


def test_ppi_interval_covers_the_truth():
    cover = 0
    for seed in range(30):
        m, ml, g, rate = _synthetic(seed=seed)
        r = ppi.rectified_mean(m, ml, g)
        cover += int(r["lo"] <= rate <= r["hi"])
    assert cover >= 25, f"coverage {cover}/30 is far below the nominal 95%"


def test_ppi_corrects_a_biased_model():
    """The model-only estimate should be further from truth than the corrected one."""
    off_model, off_ppi = [], []
    for seed in range(20):
        m, ml, g, rate = _synthetic(err=0.20, seed=seed)
        r = ppi.rectified_mean(m, ml, g)
        off_model.append(abs(r["theta_model_only"] - rate))
        off_ppi.append(abs(r["theta_ppi"] - rate))
    assert np.mean(off_ppi) < np.mean(off_model), (
        f"correction did not help: ppi {np.mean(off_ppi):.4f} vs model {np.mean(off_model):.4f}")


def test_too_few_gold_labels_refuses():
    m, ml, g, _ = _synthetic(n_gold=200)
    with pytest.raises(ValueError, match="would be noise"):
        ppi.rectified_mean(m, ml[:5], g[:5])


def test_unreviewed_label_set_is_refused_as_gold():
    df = pd.DataFrame({"reviewed_by_human": [False] * 246})
    with pytest.raises(ppi.NotGold):
        ppi.assert_is_gold(df, name="some_unreviewed_set")


def test_banned_set_is_refused_by_name():
    """Banned in code, not in prose: its labels came from the prompt's own author."""
    df = pd.DataFrame({"reviewed_by_human": [True] * 10})
    with pytest.raises(ppi.NotGold, match="circular"):
        ppi.assert_is_gold(df, name="sept2019_crowding")


def test_width_gain_reports_against_gold_only():
    """The honest alternative is discarding the model, not trusting it."""
    m, ml, g, _ = _synthetic()
    w = ppi.width_gain(ppi.rectified_mean(m, ml, g))
    assert {"se_ppi", "se_gold_only", "narrower_than_gold_only", "ratio"} <= set(w)
