"""Vintage-matched embeddings of company text, leak-free by construction.

Why chronologically consistent checkpoints
------------------------------------------
This project measured that removing dates does not hide the date: a model placed every
historical window correctly with dates stripped. Blinding is therefore not a defence, and
every embedding computed with a frontier model carries knowledge of what happened after the
date it is supposed to describe. A checkpoint trained only on text timestamped before its
cutoff cannot carry that knowledge. The 2018 year-end checkpoint embedding a 2019 filing is
point-in-time in a way no prompt discipline can achieve.

Checkpoints run annually from 1999 to 2024, so a date picks the checkpoint whose cutoff most
recently precedes it. The checkpoint id belongs in the version record beside the model id and
the prompt hash; changing it breaks the series.

Centering is not optional and this is the whole subtlety
--------------------------------------------------------
Raw mean-pooled BERT embeddings are anisotropic: they occupy a narrow cone where everything
resembles everything. Measured on eight deliberately unrelated business descriptions, raw
cosine separated same-industry from different-industry pairs by 0.013, with every pair
between 0.943 and 0.982. After subtracting the cross-sectional mean the same separation is
0.412. Thirty times the signal, from one subtraction, and the raw version would not have
looked broken -- dispersion measures computed on it would have moved a little and meant
nothing.

Whitening on top of centering destroyed the separation entirely at small sample, which is
what estimating a 768-dimensional covariance from a handful of documents does. Centering
alone is the operation that earns its place.

The centering reference is a point-in-time decision. The mean is taken over the cross-section
AS OF THE SAME DATE, never over pooled history, because a mean computed across all dates
carries future documents into a past measure. That is the same fault as dating a holdings
panel by its period instead of its filing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data/interim/embeddings")

#: Annual cutoffs published for the chronologically consistent encoder.
CHECKPOINTS = tuple(range(1999, 2025))

#: Encoder rather than the generative variants: this tier produces vectors, and the naming
#: step that needs generation lives in a different tier with different controls.
REPO = "manelalab/chrono-bert-v1-{year}1231"


def checkpoint_for(measured_at: str | pd.Timestamp) -> str:
    """The latest checkpoint whose cutoff precedes the MEASUREMENT date.

    The argument is the date the measurement is made, NEVER the date a document was filed.
    That distinction is load-bearing and was got wrong once: `embed_corpus.py` mapped each
    filing's own date to a checkpoint, which would have put documents from different years into
    different embedding spaces inside a single cross-section. Vectors from different checkpoints
    are not comparable -- their cosines mean nothing -- so mixing them silently invalidates every
    statistic computed downstream, with no error and no obvious symptom.

    A measurement at any date in February 2020 therefore uses the 2019 year-end checkpoint for
    EVERY document entering it, whether that document was filed in 2016 or last week. The cost
    is that a document is re-embedded once per vintage it serves; the alternative is numbers
    that cannot mean anything.
    """
    y = pd.Timestamp(measured_at).year - 1
    if y < CHECKPOINTS[0]:
        raise ValueError(f"no checkpoint precedes {measured_at}; earliest cutoff is "
                         f"{CHECKPOINTS[0]}")
    return REPO.format(year=min(y, CHECKPOINTS[-1]))


class MixedVintage(ValueError):
    """Raised when one cross-section would be embedded under more than one checkpoint."""


def assert_single_vintage(measured_at, *, name: str = "cross-section") -> str:
    """Every document in one measurement must share one checkpoint. Returns it.

    Accepts a scalar or an iterable of dates. An iterable spanning a year boundary raises,
    because that is the mixed-space error stated above and it is silent otherwise.
    """
    if isinstance(measured_at, (str, pd.Timestamp)):
        return checkpoint_for(measured_at)
    repos = {checkpoint_for(d) for d in measured_at}
    if len(repos) > 1:
        raise MixedVintage(
            f"{name} spans {len(repos)} checkpoints ({sorted(repos)}). Vectors from different "
            "checkpoints are not comparable; embed each measurement date separately.")
    if not repos:
        raise ValueError(f"{name} has no dates")
    return repos.pop()


@dataclass
class ChronoEmbedder:
    device: str = "mps"
    max_length: int = 2048
    batch_size: int = 8

    def _load(self, repo: str):
        from transformers import AutoModel, AutoTokenizer

        if getattr(self, "_repo", None) != repo:
            self._tok = AutoTokenizer.from_pretrained(repo)
            self._mdl = AutoModel.from_pretrained(repo).to(self.device).eval()
            self._repo = repo
        return self._tok, self._mdl

    def embed(self, texts: list[str], measured_at: str | pd.Timestamp, *,
              use_cache: bool = True) -> np.ndarray:
        """Mean-pooled vectors from the checkpoint matching the MEASUREMENT date.

        Not the filing date. See `checkpoint_for`. Vectors are not centred here -- centring is
        a per-cross-section operation and belongs where the population is known.
        """
        import torch

        repo = checkpoint_for(measured_at)
        # max_length belongs in the key. Without it, re-running the same texts at a longer
        # context returns the vectors computed at the SHORTER one, instantly and silently --
        # which read as a free 8x context increase when it was the cache answering. Truncation
        # length changes the vector as surely as the checkpoint does.
        key = hashlib.sha256(
            ("|".join(texts) + repo + f"|len={self.max_length}").encode()).hexdigest()[:20]
        f = CACHE / f"{key}.npy"
        if use_cache and f.exists():
            return np.load(f)

        tok, mdl = self._load(repo)
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            with torch.no_grad():
                b = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_length).to(self.device)
                h = mdl(**b).last_hidden_state
                mask = b["attention_mask"].unsqueeze(-1).float()
                out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
        E = np.vstack(out).astype(np.float32)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, E)
        return E


def center(E: np.ndarray) -> np.ndarray:
    """Subtract the cross-sectional mean and unit-normalise.

    Call this on the vectors of ONE date's cross-section. Centering across pooled dates would
    put future documents into a past measure.
    """
    X = E - E.mean(axis=0, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


# ---------------------------------------------------------------- E1 measures

def dispersion(Ec: np.ndarray) -> float:
    """Mean pairwise cosine within a set of centered vectors: high means one theme."""
    if len(Ec) < 2:
        return float("nan")
    C = Ec @ Ec.T
    iu = np.triu_indices(len(Ec), 1)
    return float(C[iu].mean())


def centroid(Ec: np.ndarray) -> np.ndarray:
    c = Ec.mean(axis=0)
    n = np.linalg.norm(c)
    return c / (n if n else 1)


def leg_distance(Ew: np.ndarray, El: np.ndarray) -> float:
    """Cosine between the two leg centroids. Near -1 means one two-sided bet."""
    return float(centroid(Ew) @ centroid(El))


def centroid_drift(prev: np.ndarray, cur: np.ndarray) -> float:
    """One minus cosine between consecutive centroids: how far the book's theme moved."""
    return float(1.0 - prev @ cur)
