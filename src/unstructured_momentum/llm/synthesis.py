"""Semantic aggregation over a book's news, with a placebo and a hindsight probe.

Why this exists
---------------
Every text measure in this project so far operates on co-occurrence counts in a fixed
theme vocabulary. That representation cannot express the thing the September 2019 book
actually was: a utility, a property trust and a consumer staple are, in theme space, three
unrelated companies, and in economic terms they are one bet on falling interest rates.
Theme concentration was never a fair test of the hypothesis.

A model can hold that abstraction. So the question becomes: shown only the news about each
holding, with no dates and no ticker, does it identify the shared driver?

Two safeguards are part of the method, not additions to it
---------------------------------------------------------
**The placebo.** A model asked whether a set of companies shares a driver is being invited
to say yes. Three matched calm windows measure that tendency. If the model reports a
confident common thread everywhere, the method is refuted whatever it says about 2019.

**The hindsight probe.** Every prompt also asks the model to name the year. Every model
reachable today was trained on text covering 2019, so if it recognises the period it may be
scoring what it knows happened rather than what the headlines say. A successful guess voids
the result in either direction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Stripped from every headline before the model sees it. Years, month names and any
#: four-digit token that could date the window.
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_MONTH = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
                    r"(uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\b", re.I)
_DATEISH = re.compile(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b|\b\d{6,8}\b")

#: Fixed before the run: what counts as the model having named the rate story.
RATE_WORDS = ("rate", "interest", "bond", "yield", "treasury", "duration", "defensive",
              "bond proxy", "monetary", "fed", "central bank", "safe haven")


def deidentify(headline: str) -> str:
    s = _YEAR.sub(" ", headline)
    s = _DATEISH.sub(" ", s)
    s = _MONTH.sub(" ", s)
    return " ".join(s.split())


def company_digests(corpus: pd.DataFrame, members, *, max_per: int = 6) -> dict[str, list[str]]:
    """Headlines per holding, de-identified, capped so no company dominates the prompt."""
    out = {}
    sub = corpus[corpus["ticker"].isin(set(members))]
    for t, g in sub.groupby("ticker"):
        hs = [deidentify(h) for h in g["headline"].head(max_per) if len(deidentify(h)) > 20]
        if hs:
            out[t] = hs
    return out


PROMPT = """You are shown news coverage of the companies held on one side of an equity portfolio.
Company identities are replaced with letters. All dates have been removed.

Your task has three parts.

1. Are these companies exposed to a COMMON underlying economic driver, or are they an
   assortment of unrelated businesses? Judge from the coverage only.
2. If there is a common driver, name it in under twelve words. If there is not, say "none".
3. Score the concentration from 0 to 10, where 0 means every company has its own unrelated
   story and 10 means every company is the same bet wearing a different name.

Then, separately: guess the calendar year this coverage comes from, and state how confident
you are on a 0 to 10 scale. If you cannot tell, say 0.

Return strict JSON and nothing else:
{"driver": "<text or none>", "concentration": <0-10>, "year_guess": <int or 0>,
 "year_confidence": <0-10>, "reasoning": "<one sentence>"}

COVERAGE:
"""


def build_prompt(digests: dict[str, list[str]], rng) -> str:
    keys = list(digests)
    rng.shuffle(keys)
    lines = []
    for i, t in enumerate(keys):
        label = chr(65 + i % 26) + (str(i // 26) if i >= 26 else "")
        for h in digests[t]:
            lines.append(f"  [{label}] {h}")
    rng.shuffle(lines)
    return PROMPT + "\n".join(lines)


@dataclass
class Reading:
    window: str
    draw: int
    driver: str
    concentration: float
    year_guess: int
    year_confidence: float
    names_rate: bool


def scores_to_frame(readings: list[Reading]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in readings])


def summarise(df: pd.DataFrame, episode: str = "episode_2019") -> dict:
    ep = df[df.window == episode]
    calm = df[df.window != episode]
    return {
        "episode_concentration": float(ep.concentration.mean()),
        "calm_concentration": float(calm.concentration.mean()),
        "gap": float(ep.concentration.mean() - calm.concentration.mean()),
        "episode_names_rate": float(ep.names_rate.mean()),
        "calm_names_rate": float(calm.names_rate.mean()),
        "year_guess_accuracy": float((df.year_guess == df.true_year).mean())
        if "true_year" in df else float("nan"),
        "mean_year_confidence": float(df.year_confidence.mean()),
    }
