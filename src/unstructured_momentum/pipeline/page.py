"""exp-068 — the PM-facing page, assembled from certified numbers only.

One month-end in, one page out. Every numeric field carries a provenance record naming the
artifact it came from and that artifact's sha256, so the composition verifier can check the
document rather than trust it. No language model participates in composing this page: the
narrative comparison enters as an already quote-gated record from the prior stage, and
nothing else here is model-written.

What the page is FOR
--------------------
It is a screening artifact, not a forecast. It says what state the sensor is in, what the
momentum sort has manufactured, which historical months resemble this one and WHY in terms
of the blocks that drove the distance, and what could not be measured. The severity number
comes from vol-scaled climatology because this project measured seven candidates losing to
trailing volatility, the analogue engine among them.

The could-not-measure section is a feature
------------------------------------------
It is expected to be long, and a page that omitted it would be claiming coverage it does not
have. Each entry names the reason, and the reasons are inherited from the gates that failed
rather than composed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SV = Path("data/processed/state_vector.parquet")
SPEC = Path("data/processed/state_vector_spec.json")
CONFERRED = Path("data/processed/conferred_monthly.parquet")
NARRATIVE = Path("reports/stage5/gate.json")

#: Minimum formation observations for a conferred loading, per trp-65.
MIN_FORMATION_OBS = 120
K_ANALOGUES = 5
HORIZONS_M = (1, 3)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "MISSING"


@dataclass
class Num:
    """A number that knows where it came from. The verifier checks this, not the value."""

    value: float | int | None
    source: str                      # artifact path
    sha256: str                      # that artifact's hash
    field: str                       # the column or key inside it
    note: str = ""

    def as_dict(self) -> dict:
        return {"value": self.value, "provenance": {"source": self.source,
                                                    "sha256": self.sha256,
                                                    "field": self.field},
                "note": self.note}


@dataclass
class Page:
    as_of: str
    sections: dict = field(default_factory=dict)
    could_not_measure: list = field(default_factory=list)
    falsifier: dict | None = None

    def as_dict(self) -> dict:
        return {"as_of": self.as_of, "sections": self.sections,
                "could_not_measure": self.could_not_measure, "falsifier": self.falsifier}


# ----------------------------------------------------------------------------------------
# Severity — vol-scaled climatology, the incumbent that beat the analogue engine
# ----------------------------------------------------------------------------------------

def vol_scaled_climatology(wml_m: pd.Series, rv: pd.Series, asof: pd.Timestamp,
                           h: int) -> dict:
    """Each past outcome divided by the trailing vol prevailing THEN, rescaled to today's.

    One line of arithmetic and it beat the analogue ensemble on 6,269 dates. Only history
    strictly before `asof` enters, and the forward window of each historical observation must
    have closed before `asof` -- the same stricter-than-standard eligibility the retrieval
    engine uses, for the same reason.
    """
    fwd = (1 + wml_m).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
    elig = fwd.index[fwd.index + pd.offsets.MonthEnd(h) < asof]
    past = fwd.reindex(elig).dropna()
    v = rv.reindex(past.index)
    ok = v.notna() & (v > 0)
    past, v = past[ok], v[ok]
    if len(past) < 60 or asof not in rv.index or not np.isfinite(rv.loc[asof]):
        return {}
    scaled = (past / v) * float(rv.loc[asof])
    return {"n_observations": int(len(scaled)),
            "q05": float(scaled.quantile(0.05)), "q10": float(scaled.quantile(0.10)),
            "q50": float(scaled.quantile(0.50)), "q90": float(scaled.quantile(0.90)),
            "trailing_vol_annualised": float(rv.loc[asof])}


# ----------------------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------------------

class Composer:
    def __init__(self) -> None:
        from ..analogue.retrieve import Engine
        from ..events.registry import EPISODES

        self.sv = pd.read_parquet(SV)
        self.spec = json.loads(SPEC.read_text())
        self.sv_sha = self.spec["sha256"][:16]
        self.eng = Engine("deep")
        self.episodes = {pd.Timestamp(e.trough) + pd.offsets.MonthEnd(0): e for e in EPISODES}

        c = pd.read_parquet(CONFERRED)
        c = c.mask(c["n_obs"] < MIN_FORMATION_OBS, other=np.nan)
        # Resample to CALENDAR month-ends. The panel is indexed by the holdings date, which
        # is the last trading day of the month and only sometimes the calendar month-end --
        # 2022-12-30 against 2022-12-31. Looking it up by month-end silently dropped the
        # manufactured-bet block on every month whose last trading day was not the last
        # calendar day, which is most of them, and the page said nothing.
        c.index = pd.to_datetime(c.index)
        self.conferred = c.resample("ME").last()
        self.conf_sha = _sha(CONFERRED)

        from ..data import french
        w = french.momentum()
        self.wml_m = (1 + w).resample("ME").prod() - 1
        self.rv_m = (w.rolling(126, min_periods=100).std() * np.sqrt(252)).resample("ME").last()

        self.regimes = self._fit_regimes()
        self.narrative = self._load_narrative()

    def _fit_regimes(self) -> pd.Series:
        from ..analogue import jump
        cols = ["z_factor_dispersion", "z_rv126", "z_joint_z", "z_wml_asymmetry"]
        X = self.sv[cols].dropna()
        f = jump.fit(X.to_numpy(float), k=3, lam=16, n_init=12, seed=20260830)
        return pd.Series(f.states, index=X.index, name="regime")

    def _load_narrative(self) -> dict:
        if not NARRATIVE.exists():
            return {}
        d = json.loads(NARRATIVE.read_text())
        return {(r["query_month"], r["analogue_month"]): r for r in d.get("records", [])}

    # -- sections -----------------------------------------------------------------------

    def compose(self, asof) -> Page:
        m = pd.Timestamp(asof) + pd.offsets.MonthEnd(0)
        p = Page(as_of=str(m.date()))

        p.sections["state"] = self._state(m)
        p.sections["severity"] = self._severity(m)
        p.sections["manufactured_bet"] = self._bet(m, p)
        p.sections["regime"] = self._regime(m, p)
        p.sections["analogues"] = self._analogues(m, p)
        p.sections["describable_analogues"] = self._describable(m, p)
        self._standing_limits(p)
        p.falsifier = self._falsifier(m)
        return p

    def _state(self, m: pd.Timestamp) -> dict:
        if m not in self.sv.index:
            return {"available": False}
        row = self.sv.loc[m]
        out = {"available": True, "coverage": float(row["coverage"]),
               "blocks_present": int(row["n_blocks"]),
               # Read from the spec, not hardcoded. A ninth block was added and the page
               # printed "9 of 8 blocks", which is the kind of detail a reader uses to
               # decide whether to trust the rest of the number.
               "blocks_total": len(self.spec["blocks"]), "blocks": {}}
        for b, cols in self.spec["blocks"].items():
            vals = {}
            for c in cols:
                z = row.get(f"z_{c}")
                if pd.notna(z):
                    pctile = float((self.sv[f"z_{c}"].dropna() <= z).mean() * 100)
                    vals[c] = Num(round(float(z), 3), str(SV), self.sv_sha, f"z_{c}",
                                  f"{pctile:.0f}th percentile of own history").as_dict()
            if vals:
                out["blocks"][b] = vals
        return out

    def _severity(self, m: pd.Timestamp) -> dict:
        out = {"method": "vol-scaled climatology",
               "why": ("Seven candidate signals have lost to trailing volatility in this "
                       "project, the analogue engine among them. The number comes from the "
                       "method that won."),
               "horizons": {}}
        for h in HORIZONS_M:
            s = vol_scaled_climatology(self.wml_m, self.rv_m, m, h)
            if not s:
                continue
            out["horizons"][f"{h}m"] = {
                k: Num(round(v, 4) if isinstance(v, float) else v,
                       "french:momentum_daily", "derived", k).as_dict()
                for k, v in s.items()}
        return out

    def _bet(self, m: pd.Timestamp, p: Page) -> dict:
        if m not in self.conferred.index or pd.isna(self.conferred.loc[m, "beta_spread"]):
            p.could_not_measure.append({
                "item": "what the sort manufactured",
                "reason": ("The conferred-tilt panel needs at least "
                           f"{MIN_FORMATION_OBS} formation-window observations; the holdings "
                           "panel is monthly before mid-2013 and a loading cannot be "
                           "estimated from twelve points."),
                "trap": "trp-65"})
            return {"available": False}
        r = self.conferred.loc[m]
        tilt = float(r["beta_spread"])
        hist = self.conferred["beta_spread"].dropna()
        return {"available": True,
                "conferred_beta_spread": Num(round(tilt, 3), str(CONFERRED), self.conf_sha,
                                             "beta_spread",
                                             "negative means the sort built a DEFENSIVE tilt"
                                             ).as_dict(),
                "own_history_percentile": Num(round(float((hist <= tilt).mean() * 100), 1),
                                              str(CONFERRED), self.conf_sha, "beta_spread",
                                              f"against {len(hist)} months").as_dict(),
                "formation_observations": Num(int(r["n_obs"]), str(CONFERRED), self.conf_sha,
                                              "n_obs").as_dict()}

    def _regime(self, m: pd.Timestamp, p: Page) -> dict:
        if m not in self.regimes.index:
            return {"available": False}
        st = int(self.regimes.loc[m])
        hist = self.regimes.loc[:m]
        sw = hist[hist != hist.shift()].index
        last = sw[-1] if len(sw) else hist.index[0]
        return {"available": True,
                "state": Num(st, "jump model k=3 lam=16", self.sv_sha, "regime",
                             "0 calmest, 2 most extreme; ordered by centroid norm").as_dict(),
                "months_in_state": Num(int((hist.index > last).sum()), "jump model",
                                       self.sv_sha, "regime").as_dict(),
                "last_switch": str(last.date()),
                "caveat": ("The learned regime layer did NOT displace the incumbent "
                           "two-channel indicator: it failed its registered separation gate, "
                           "and so did the incumbent, because a monthly panel cannot see a "
                           "four-day episode. Reported as context, not as a selector.")}

    def _analogues(self, m: pd.Timestamp, p: Page) -> dict:
        if m not in self.eng.index:
            p.could_not_measure.append({
                "item": "historical analogues",
                "reason": "No complete state row for this month in the deep-archive arm."})
            return {"available": False, "matches": []}
        out = []
        for n in self.eng.query(m, k=K_ANALOGUES, horizon_months=3):
            reg = (int(self.regimes.loc[n.date]) if n.date in self.regimes.index else None)
            here = int(self.regimes.loc[m]) if m in self.regimes.index else None
            key = (str(pd.Period(m, freq="M")), str(pd.Period(n.date, freq="M")))
            rec = self.narrative.get(key)
            out.append({
                "date": str(n.date.date()), "rank": n.rank,
                "distance": Num(round(n.distance, 4), "analogue.retrieve:Engine(deep)",
                                self.sv_sha, "cosine on unit-normalised weighted key").as_dict(),
                # Two different questions, and the first version of this page conflated them.
                # `block_share` is the share of the REMAINING DISTANCE, so a high share means
                # the two months stay most unlike on that block. Printing it as "matched on"
                # told the reader the opposite of the truth on every page of the replay.
                "why_it_matched": {b: f"{v:.2f}x" for b, v in
                                   sorted(n.block_closeness.items(), key=lambda x: x[1])},
                "why_it_matched_units": ("per-block distance as a multiple of the median for "
                                         "two arbitrary months; below 1.00 means closer than "
                                         "chance, and lower is more of the reason"),
                "residual_distance_share": {b: f"{v:.0%}" for b, v in
                                            sorted(n.block_share.items(), key=lambda x: -x[1])},
                "is_analogue": n.is_analogue,
                "cross_regime": (None if reg is None or here is None else bool(reg != here)),
                "regime_state": reg,
                "near_registered_episode": next(
                    (e.key for d, e in self.episodes.items()
                     if abs((d - n.date).days) <= 62), None),
                "narrative_comparison": (
                    {"shared_conditions": rec["comparison"]["shared_conditions"],
                     "disanalogies": rec["comparison"]["disanalogies"],
                     "what_followed_then": rec["comparison"]["what_followed_then"],
                     "hindsight_disclosure": (
                         "Every claim here is quote-verified against its source and cannot "
                         "have been invented. EMPHASIS is not bounded: a frontier model "
                         "placed 93.3 percent of these masked blocks within three months of "
                         "their true date, so which true quotes were surfaced may be "
                         "informed by knowing when this was.")}
                    if rec else None),
                "narrative_unavailable_because": (
                    None if rec else
                    ("the analogue predates the current-events corpus, which begins 2018-01"
                     if n.date < pd.Timestamp("2018-01-31") else
                     "no comparison was computed for this pair")),
            })
        n_narr = sum(1 for o in out if o["narrative_comparison"])
        if n_narr == 0 and out:
            p.could_not_measure.append({
                "item": "narrative comparison for these analogues",
                "reason": ("None of the retrieved analogues falls inside the current-events "
                           "corpus, which begins 2018-01. The deep archive reaches 1930 and "
                           "the text does not, so most precedents this system finds can be "
                           "explained numerically and not narratively."),
                "experiment": "exp-067"})
        return {"available": True, "k": K_ANALOGUES, "matches": out,
                "n_with_narrative": n_narr}

    def _describable(self, m: pd.Timestamp, p: Page) -> dict:
        """The closest states we can also DESCRIBE, and what that restriction costs.

        The deep archive reaches 1930 and the current-events corpus begins 2018. Across a
        372-page replay the two never met: the median retrieved analogue is 19.9 years old and
        only 7.7 percent of retrieved analogues fall inside the corpus. That is not a defect
        to hide, it is the finding -- the better the retrieval reaches, the less likely a
        narrative comparison exists for what it finds.
        So the page carries a SECOND, clearly separate block: the closest states restricted to
        months the corpus can speak about, with the distance penalty of that restriction
        printed beside it. A reader can then see both what the archive says is closest and
        what we can actually put words to, and the gap between them.
        """
        era = pd.Timestamp("2018-01-31")
        if m < era or m not in self.eng.index:
            return {"available": False,
                    "reason": "the query month itself predates the current-events corpus"}
        pool = [d for d in self.eng.eligible(m, 3) if d >= era]
        if len(pool) < 1:
            return {"available": False,
                    "reason": "no eligible month inside the corpus for this query"}

        qi = self.eng.index.get_loc(m)
        pi = self.eng.index.get_indexer(pool)
        dist = np.sqrt(((self.eng._U[pi] - self.eng._U[qi]) ** 2).sum(1))
        order = np.argsort(dist)[:K_ANALOGUES]
        best_all = self.eng.query(m, k=1, horizon_months=3)
        penalty = (float(dist[order[0]] - best_all[0].distance) if best_all else None)

        out = []
        for i in order:
            am = pool[int(i)]
            rec = self.narrative.get((str(pd.Period(m, freq="M")),
                                      str(pd.Period(am, freq="M"))))
            out.append({"date": str(am.date()), "distance": round(float(dist[i]), 4),
                        "narrative_comparison": (
                            {"shared_conditions": rec["comparison"]["shared_conditions"],
                             "disanalogies": rec["comparison"]["disanalogies"],
                             "what_followed_then": rec["comparison"]["what_followed_then"],
                             "hindsight_disclosure": (
                                 "Every claim is quote-verified against its source and cannot "
                                 "have been invented. EMPHASIS is not bounded: a frontier "
                                 "model placed 93.3 percent of these masked blocks within "
                                 "three months of their true date.")}
                            if rec else None)})
        p.could_not_measure.append({
            "item": "a narrative comparison for the CLOSEST states",
            "reason": ("The closest states are outside the current-events corpus. The block "
                       "below shows the closest states the corpus can describe instead, and "
                       "they sit " + (f"{penalty:+.3f} further away" if penalty is not None
                                      else "further away") + " in the key. Across a 372-page "
                       "replay only 7.7 percent of retrieved analogues fell inside the corpus "
                       "at all."),
            "experiment": "exp-068"})
        return {"available": True, "restricted_to": "2018-01 onward",
                "distance_penalty_vs_closest_overall": penalty,
                "n_with_narrative": sum(1 for o in out if o["narrative_comparison"]),
                "matches": out}

    def _standing_limits(self, p: Page) -> None:
        """Limits inherited from gates that failed. Not composed here; quoted from them."""
        p.could_not_measure += [
            {"item": "which channel dated a regime break first",
             "reason": ("PELT under a false-alarm discipline of under 10 percent per decade "
                        "needs a three standard deviation break. Recovery of a one-sigma "
                        "break runs 0.00, 0.12, 0.33, 0.35 at 84, 168, 336 and 672 months, "
                        "so more history does not fix it."),
             "experiment": "exp-065"},
            {"item": "whether text improves which precedents are retrieved",
             "reason": ("Only three episode queries have a same-mechanism precedent inside "
                        "the 8-K era and all three share one target month, giving a "
                        "sign-test floor of p=0.125. Text hurt on all three, which is a "
                        "direction and not an answer."),
             "experiment": "exp-066"},
            {"item": "probability of a reversal",
             "reason": ("The effective sample is 8 to 17 independent regimes. Severity is "
                        "answerable from thousands of daily observations; a probability for "
                        "a rare event is not."),
             "claim": "clm-power"},
            {"item": "a hindsight-free narrative comparison",
             "reason": ("Period recovery from masked filings is 93.3 percent within three "
                        "months. The signal is lexical and distributed: masking dates, "
                        "names, figures and filenames changes nothing, and only scrubbing "
                        "the vocabulary works, which removes the content."),
             "experiment": "exp-067"},
        ]

    def _falsifier(self, m: pd.Timestamp) -> dict:
        """One machine-checkable condition, so the page's own reading can be scored later.

        Names a variable, a comparator, a threshold and a horizon, and nothing else, because
        a falsifier a machine cannot evaluate is a sentence rather than a commitment.
        """
        if m not in self.sv.index or pd.isna(self.sv.loc[m, "z_rv126"]):
            return {"available": False,
                    "reason": "no volatility-regime reading at this month-end"}
        z = float(self.sv.loc[m, "z_rv126"])
        elevated = z > 0
        return {
            "available": True,
            "variable": "z_rv126",
            "source": str(SV), "sha256": self.sv_sha,
            "comparator": "<" if elevated else ">",
            "threshold": 0.0,
            "horizon_months": 3,
            "reading_it_tests": ("elevated volatility regime" if elevated
                                 else "benign volatility regime"),
            "statement": (
                f"This page reads the volatility regime as "
                f"{'elevated' if elevated else 'benign'} at z={z:.2f}. If z_rv126 is "
                f"{'below' if elevated else 'above'} 0.0 at any month-end within 3 months, "
                "that reading was wrong."),
        }


def render(p: Page) -> str:
    """Plain text for a human. Deterministic, no model, no adjectives that are not measured."""
    d = p.as_dict()
    L = [f"MOMENTUM REVERSAL SCREEN  ---  {d['as_of']}", "=" * 74, ""]

    st = d["sections"]["state"]
    if st.get("available"):
        L.append(f"STATE   coverage {st['coverage']:.0%}, {st['blocks_present']} of "
                 f"{st.get('blocks_total', len(st['blocks']))} blocks")
        for b, vals in st["blocks"].items():
            bits = ", ".join(f"{c} {v['value']:+.2f} ({v['note'].split(' ')[0]})"
                             for c, v in vals.items())
            L.append(f"   {b:11s} {bits}")
        L.append("")

    sev = d["sections"]["severity"]
    if sev.get("horizons"):
        L.append(f"SEVERITY   {sev['method']}")
        for h, q in sev["horizons"].items():
            L.append(f"   {h:3s} forward WML:  5th pct {q['q05']['value']:+.1%}   "
                     f"median {q['q50']['value']:+.1%}   90th {q['q90']['value']:+.1%}"
                     f"   (n={q['n_observations']['value']})")
        L.append(f"   {sev['why']}")
        L.append("")

    bet = d["sections"]["manufactured_bet"]
    if bet.get("available"):
        v = bet["conferred_beta_spread"]["value"]
        L.append("WHAT THE SORT MANUFACTURED")
        L.append(f"   conferred beta spread {v:+.3f} "
                 f"({'defensive' if v < 0 else 'aggressive'} tilt), "
                 f"{bet['own_history_percentile']['value']:.0f}th percentile of own history")
        L.append("")

    reg = d["sections"]["regime"]
    if reg.get("available"):
        L.append(f"REGIME   state {reg['state']['value']} for "
                 f"{reg['months_in_state']['value']} months (last switch {reg['last_switch']})")
        L.append("")

    an = d["sections"]["analogues"]
    if an.get("available"):
        L.append(f"CLOSEST HISTORICAL STATES   (k={an['k']}, deep archive)")
        for mm in an["matches"]:
            tag = "" if mm["is_analogue"] else "   [beyond the analogue floor]"
            ep = f"   near {mm['near_registered_episode']}" if mm["near_registered_episode"] else ""
            xr = "   [cross-regime]" if mm["cross_regime"] else ""
            L.append(f"   {mm['rank']}. {mm['date']}  d={mm['distance']['value']:.3f}{tag}{ep}{xr}")
            L.append("      alike on: " + ", ".join(f"{k} {v}" for k, v in
                                                     mm["why_it_matched"].items())
                     + "   (x median random pair; <1.00 = closer than chance)")
            nc = mm.get("narrative_comparison")
            if nc:
                for it in nc["disanalogies"][:2]:
                    L.append(f"      DIFFERS: {it['claim']}")
                L.append("      " + nc["hindsight_disclosure"][:96] + "...")
        L.append("")

    da = d["sections"].get("describable_analogues", {})
    if da.get("available"):
        pen = da.get("distance_penalty_vs_closest_overall")
        L.append("CLOSEST STATES WE CAN ALSO DESCRIBE   (restricted to "
                 f"{da['restricted_to']}; {pen:+.3f} further away than the closest overall)")
        for mm in da["matches"]:
            L.append(f"   {mm['date']}  d={mm['distance']:.3f}")
            nc = mm.get("narrative_comparison")
            if nc:
                # The model sometimes emits one claim twice with different supporting quotes.
                # Both verify, so the quote gate keeps both; a reader wants the claim once.
                def uniq(items, n):
                    seen, out = set(), []
                    for it in items:
                        if it["claim"] not in seen:
                            seen.add(it["claim"]); out.append(it)
                        if len(out) == n:
                            break
                    return out
                for it in uniq(nc["shared_conditions"], 2):
                    L.append(f"      SHARED:  {it['claim']}")
                for it in uniq(nc["disanalogies"], 2):
                    L.append(f"      DIFFERS: {it['claim']}")
                for it in uniq(nc["what_followed_then"], 1):
                    L.append(f"      FOLLOWED: {it['claim']}")
                L.append(f"      {nc['hindsight_disclosure'][:110]}...")
        L.append("")

    L.append("COULD NOT MEASURE")
    for c in d["could_not_measure"]:
        L.append(f"   - {c['item']}: {c['reason']}")
    L.append("")

    f = d["falsifier"]
    if f and f.get("available"):
        L.append("FALSIFIER   " + f["statement"])
    return "\n".join(L)
