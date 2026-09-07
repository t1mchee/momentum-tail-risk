"""Concentration in what is *written about* the book's holdings.

The idea, and why it is not another sentiment feature
-----------------------------------------------------
The most useful thing this project found is that in September 2019 the book was one bet on
falling interest rates wearing ten sector labels, and every portfolio-level average called
it diversified while the individual holdings did not. The failure was aggregation, not
measurement.

If that is a general property of fragile momentum books rather than one episode's accident,
it should be visible in what is written about the holdings. A crowded winning side is
crowded *because the companies are being bought for the same reason*, and that reason gets
written down. So the measure is **concentration in what is said**, not how much is said.

Counting articles measures attention volume, which the news cycle dominates and which
trading volume already spans. Concentration is a different quantity and it has the one
property every failed candidate in this project lacked: it is not a function of past
returns.

Effective rank
--------------
Build the theme-occurrence matrix over the winning side: rows are holdings, columns are
themes, entries are how often that theme appears in coverage of that company. Its
**effective rank** is the exponential of the entropy of its normalised singular values.

* effective rank near the number of holdings: every company has its own story;
* effective rank near one: coverage of many companies is saying one thing.

This is deliberately the same shape as the crowding measure that replicated out of sample
at a 3.06-fold crash-day lift, applied to a non-return-derived input.

Why themes rather than embeddings, for now
------------------------------------------
Every embedding model reachable today was trained on text that postdates every episode
studied, so an embedding of 2019 coverage is produced by something that knows how 2019
ended. The archive's theme vocabulary is fixed, published, assigned by a classifier that
was not trained on the outcome, and available at the timestamp. It is coarser and it is
clean. Embeddings are the next rung of the ladder, not a replacement for this one, and they
have to beat it to earn their place.

The join is the dangerous part
------------------------------
Matching an organisation string to a holding is a fuzzy join wearing an exact-match
costume. A name matching two companies, or a company appearing under three spellings, will
quietly distort a concentration measure, and the failure looks like a result rather than an
error. :func:`build_alias_map` therefore drops ambiguous names rather than guessing, and
reports what it dropped.
"""

from __future__ import annotations

import datetime as dt
import io
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from ..config import RAW, USER_AGENT

GKG_URL = "http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"

#: GKG 2.1 column positions actually used. The file has 27 columns and no header.
COL_DATE, COL_SOURCE, COL_URL, COL_THEMES, COL_ORGS, COL_TONE = 1, 3, 4, 7, 13, 15

#: Corporate suffixes carry no identifying information and differ between the index's
#: spelling and the archive's. Stripped from both sides before matching.
_SUFFIX = re.compile(
    r"\b(inc|corp|corporation|company|co|ltd|limited|plc|holdings?|group|"
    r"the|sa|nv|ag|lp|llc|class [abc]|cl [abc])\b", re.I)
_PUNCT = re.compile(r"[^a-z0-9 ]+")

#: A name shorter than this matches too much to be trusted as an identifier.
MIN_NAME_LEN = 6


def normalise(name: str) -> str:
    s = _PUNCT.sub(" ", (name or "").lower())
    s = _SUFFIX.sub(" ", s)
    return " ".join(s.split())


def build_alias_map(holdings: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    """Normalised company name -> ticker, dropping anything ambiguous.

    Returns the map and a table of what was refused, because a silent drop here shrinks
    the universe in a way that looks like a finding.
    """
    by_norm: dict[str, set[str]] = defaultdict(set)
    for t, n in zip(holdings["ticker"], holdings["name"], strict=False):
        k = normalise(n)
        if len(k) >= MIN_NAME_LEN:
            by_norm[k].add(t)

    keep, refused = {}, []
    for k, tick in by_norm.items():
        if len(tick) == 1:
            keep[k] = next(iter(tick))
        else:
            refused.append({"name": k, "tickers": ",".join(sorted(tick)),
                            "reason": "one name maps to several holdings"})
    short = [n for n in holdings["name"] if len(normalise(n)) < MIN_NAME_LEN]
    refused += [{"name": n, "tickers": "", "reason": "name too short to identify"}
                for n in short]
    return keep, pd.DataFrame(refused)


def slots(day: dt.date) -> list[str]:
    """The 96 fifteen-minute stamps of one day."""
    return [f"{day:%Y%m%d}{h:02d}{m:02d}00" for h in range(24) for m in (0, 15, 30, 45)]


def _fetch_slot(session: requests.Session, stamp: str) -> list[str] | None:
    try:
        r = session.get(GKG_URL.format(stamp=stamp), timeout=120)
        if r.status_code != 200 or not r.content:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        return z.read(z.namelist()[0]).decode("utf-8", "replace").split("\n")
    except Exception:          # noqa: BLE001 - one absent slot must never kill a window
        return None


def harvest_window(
    start: dt.date, end: dt.date, alias: dict[str, str], *, workers: int = 6,
    every: int = 1,
) -> pd.DataFrame:
    """Stream every slot in a date range, keep only rows mentioning a holding.

    Around 19 gigabytes flow through for a 25-day window and nothing is written to disk:
    each archive file is filtered in memory and discarded. Only matched rows survive, which
    is a few megabytes.

    One absent or malformed slot is skipped rather than allowed to abort the window, and
    the count of misses is returned, because a silently short window would read as quiet
    news rather than as missing data.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})

    stamps: list[str] = []
    d = start
    while d <= end:
        stamps += slots(d)[::every]
        d += dt.timedelta(days=1)

    rows, misses = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for stamp, lines in zip(stamps, ex.map(lambda x: _fetch_slot(s, x), stamps),
                                strict=False):
            if lines is None:
                misses += 1
                continue
            for line in lines:
                f = line.split("\t")
                if len(f) <= COL_TONE or not f[COL_ORGS]:
                    continue
                hits = {alias[k] for k in
                        (normalise(o) for o in f[COL_ORGS].split(";")) if k in alias}
                if not hits:
                    continue
                themes = [t for t in f[COL_THEMES].split(";") if t]
                try:
                    tone = float(f[COL_TONE].split(",")[0])
                except (ValueError, IndexError):
                    tone = float("nan")
                for tick in hits:
                    rows.append({"stamp": f[COL_DATE], "source": f[COL_SOURCE],
                                 "url": f[COL_URL], "ticker": tick,
                                 "themes": themes, "tone": tone,
                                 "n_names": len(hits)})
    out = pd.DataFrame(rows)
    out.attrs["slots_requested"] = len(stamps)
    out.attrs["slots_missing"] = misses
    return out


#: Attribution ladder. A name match says a company is mentioned; it does not say the
#: article is about that company. On the September 2019 window, plain name matching gave
#: 75 percent of winning-side articles to a single newspaper publisher that is itself a
#: holding, because its name appears in the organisation field of everything that cites it.
#: Each rung must beat the one below it before its cost is justified.
_BIZ_PREFIX = ("ECON_", "BUS_")
_BIZ_CONTAINS = ("EARNINGS", "STOCKMARKET", "IPO", "MERGER", "BANKRUPTCY")


def has_business_theme(themes) -> bool:
    """Rung 1: the article is about business at all."""
    return any(t.startswith(_BIZ_PREFIX) or any(k in t for k in _BIZ_CONTAINS)
               for t in themes)


def headline_from_url(url: str) -> str:
    """The publisher's slug, which usually carries the headline. No body text is available."""
    s = re.sub(r"https?://[^/]+/", "", url or "")
    s = re.sub(r"\.(html?|php|aspx?|amp)$", " ", s)
    return re.sub(r"[/_\-]+", " ", s).lower().strip()


def named_in_headline(headline: str, company_name: str) -> bool:
    """Rung 2: the company is named in the headline, not merely somewhere in the body."""
    toks = [t for t in company_name.split() if len(t) > 3][:2]
    return bool(toks) and all(t in headline for t in toks)


def attribute(docs: pd.DataFrame, alias_by_ticker: dict[str, str],
              rung: int = 2) -> pd.DataFrame:
    """Filter a matched corpus to articles plausibly *about* their attributed company."""
    d = docs.copy()
    if rung >= 1:
        d = d[d["themes"].apply(has_business_theme)]
    if rung >= 2:
        d["headline"] = d["url"].apply(headline_from_url)
        d = d[[named_in_headline(h, alias_by_ticker.get(t, ""))
               for h, t in zip(d["headline"], d["ticker"], strict=False)]]
    return d


def theme_matrix(docs: pd.DataFrame, members: list[str]) -> tuple[np.ndarray, list[str]]:
    """Holdings-by-themes occurrence matrix for one side of the book."""
    sub = docs[docs["ticker"].isin(members)]
    if sub.empty:
        return np.zeros((0, 0)), []
    counts: dict[str, Counter] = defaultdict(Counter)
    for t, th in zip(sub["ticker"], sub["themes"], strict=False):
        counts[t].update(th)
    themes = sorted({x for c in counts.values() for x in c})
    m = np.array([[counts[t].get(x, 0) for x in themes] for t in sorted(counts)],
                 dtype=float)
    return m, sorted(counts)


def effective_rank(m: np.ndarray) -> float:
    """Exponential of the entropy of the normalised singular values.

    One means every row says the same thing. The number of rows means every company has
    its own story. Rows are normalised first so a heavily covered company cannot dominate
    by volume alone, which would turn this back into a count.
    """
    if m.size == 0 or m.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    x = m / np.where(norms == 0, 1, norms)
    sv = np.linalg.svd(x, compute_uv=False)
    p = sv / sv.sum() if sv.sum() > 0 else sv
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def leg_distance(docs: pd.DataFrame, winners: list[str], losers: list[str]) -> float:
    """Distance between the two sides' theme distributions.

    Near zero means one story is driving both sides of the book, which is what a single
    macro bet looks like from the outside.
    """
    def dist(members):
        c = Counter()
        for th in docs[docs["ticker"].isin(members)]["themes"]:
            c.update(th)
        return c
    a, b = dist(winners), dist(losers)
    keys = sorted(set(a) | set(b))
    if not keys:
        return float("nan")
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    if va.sum() == 0 or vb.sum() == 0:
        return float("nan")
    va, vb = va / va.sum(), vb / vb.sum()
    return float(0.5 * np.abs(va - vb).sum())          # total variation distance


def window_measures(docs: pd.DataFrame, winners: list[str], losers: list[str]) -> dict:
    mw, wnames = theme_matrix(docs, winners)
    ml, _ = theme_matrix(docs, losers)
    return {
        "articles": int(len(docs)),
        "winners_covered": len(wnames),
        "eff_rank_winners": effective_rank(mw),
        "eff_rank_losers": effective_rank(ml),
        "eff_rank_ratio": (effective_rank(mw) / len(wnames)) if wnames else float("nan"),
        "leg_distance": leg_distance(docs, winners, losers),
        "tone_dispersion": float(docs[docs["ticker"].isin(winners)]["tone"].std()),
        "slots_missing": docs.attrs.get("slots_missing", 0),
    }


def cache_path(tag: str):
    d = RAW / "gdelt_gkg"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.parquet"
