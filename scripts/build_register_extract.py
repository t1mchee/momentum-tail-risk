"""The package's register: the pre-registration record for what the package claims, and nothing more.

The working repository keeps a register of 91 experiments and 97 instrument faults, rendered into
a 268 KB ledger. That is a lab notebook. This package is the proof of concept, so it ships the
rows behind the three artifacts it actually shows -- each with the hypothesis, prediction and
method frozen BEFORE the run, and the verdict that came back -- and leaves the rest where it
belongs.

Selection is mechanical rather than curated, so a reader can check that nothing flattering was
kept and nothing awkward dropped:

  * an experiment is included if its `reproduce` command names a script that ships in this
    package. Two are added by hand and each says why in its own row.
  * an instrument fault is included if it was found during the work that produced these
    artifacts -- on or after 2026-09-01, when this build began -- or if one of those experiments
    names it. Faults from retired phases are not.

Every row keeps its verdict verbatim, including `refuted`, `inconclusive` and
`instrument_limited`. Four of the thirteen are failures and they are the reason the register is
here at all: a null is only worth reading if the prediction that it contradicts was written down
first.

    uv run python scripts/build_register_extract.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# The working repository writes the template the bundler renders; the package rewrites its
# own copy in place, so `make register` re-derives exactly the file the reader is holding.
OUT = (ROOT / "dist_files" / "REGISTER.md" if (ROOT / "dist_files").is_dir()
       else ROOT / "REGISTER.md")

#: When work on the three artifacts began. Faults found from here on touched them.
BUILD_BEGAN = "2026-09-01"

def shipped_scripts() -> set[str]:
    """The scripts that ship with the package.

    In the working repository that set lives in the bundler's SCRIPTS list, and is read from
    there so the two cannot drift apart. Inside the built package the bundler is not present --
    it is the working repository's tool, not the package's -- and the scripts directory IS the
    shipped set, so the same rule is applied to it directly. Both give the same answer; the
    second is the one a reader can run, which is the point of `make register`.
    """
    bundler = ROOT / "scripts" / "build_submission.py"
    if bundler.exists():
        src = bundler.read_text()
        block = src[src.index("SCRIPTS = ["):src.index("]", src.index("SCRIPTS = ["))]
        return {m.removesuffix(".py") for m in re.findall(r'"([\w]+\.py)"', block)}
    return {f.stem for f in (ROOT / "scripts").glob("*.py")}

#: Added by hand, with the reason printed in the row. Nothing else is added by hand.
EXTRA = {
    "exp-082": "the analogue panel the page renders; its artifact ships, its runner does not, "
               "because the panel is not re-run for the page",
    "exp-087": "blocked, and the reason the extraction check is a planted-span test rather than "
               "hand labels; a blocked experiment is part of the record",
}


def selection() -> tuple[list, list, dict]:
    """The experiments and faults this package claims, by the rule in the module docstring.

    Returned as data rather than only rendered, so the bundler ships exactly these entries.
    A package holding a 91-experiment register beside a 14-row extract makes claims about
    work it does not present, and its own reproduce guard then fails on generators that were
    deliberately not shipped.
    """
    exps = yaml.safe_load((ROOT / "project" / "experiments.yaml").read_text())
    traps = yaml.safe_load((ROOT / "project" / "traps.yaml").read_text())
    ships = shipped_scripts()

    keep, why = [], {}
    for e in exps:
        cmd = str(e.get("reproduce") or "")
        named = re.findall(r"(?:scripts/|-m\s+[\w.]*\.)([\w]+)", cmd)
        if any(n in ships for n in named):
            keep.append(e)
            why[e["id"]] = f"reproduces via `{next(n for n in named if n in ships)}.py`"
        elif e["id"] in EXTRA:
            keep.append(e)
            why[e["id"]] = EXTRA[e["id"]]
    got = {e["id"] for e in keep}
    missing = set(EXTRA) - got
    if missing:
        raise SystemExit(f"named extras not found in the register: {sorted(missing)}")
    keep.sort(key=lambda e: e["id"])
    ran = [e for e in keep if e.get("verdict") != "not_yet"]

    # Faults found during this build, plus any these experiments name. A fault from a retired
    # phase is not this package's business.
    blob = yaml.safe_dump(keep)
    named_traps = set(re.findall(r"trp-\d+", blob))
    tk = sorted((t for t in traps
                 if t["id"] in named_traps or str(t["found_on"]) >= BUILD_BEGAN),
                key=lambda t: t["id"])
    return keep, tk, why


def main() -> None:
    keep, tk, why = selection()
    ran = [e for e in keep if e.get("verdict") != "not_yet"]

    L = [
        "# The register",
        "",
        "Every experiment in this package was registered **before it ran**, with the hypothesis,",
        "the prediction and the method frozen at that point, and a power statement saying what the",
        "test could and could not have seen. The verdict was then recorded whichever way it fell.",
        "",
        "That order is the whole point. A null is a claim about the world only once the instrument",
        "is shown able to see the effect, and a prediction is only evidence if it was written down",
        "before the answer was known. Without this file the negative results in the README would",
        "read as after-the-fact rationalisation; with it they read as what they are.",
        "",
        f"**{len(keep)} experiments**, of which **{len(ran)} ran** and "
        f"**{sum(1 for e in ran if e.get('verdict') != 'supported')} of those did not support "
        "their hypothesis**, and "
        f"**{len(tk)} instrument faults** that touched them. A blocked experiment is counted as "
        "neither: it did not fail to support its hypothesis, it never tested it.",
        "",
        "The working repository holds a larger register covering retired phases. This is the",
        "extract for what this package claims, selected mechanically: an experiment is here if its",
        "reproduce command names a script that ships, plus two added by hand that say so in their",
        "own row.",
        "",
        "---",
        "",
        "## Every experiment at a glance",
        "",
        "| verdict | the question it was registered to answer |",
        "|---|---|",
    ] + [
        f"| **{(e.get('verdict') or e.get('status'))}** | {e['title'].strip().rstrip('?')}? |"
        for e in sorted(keep, key=lambda x: (x.get("verdict") != "refuted",
                                             x.get("verdict") != "instrument_limited",
                                             x["id"]))
    ] + [
        "",
        "---",
        "",
        "## Experiments, in full",
        "",
    ]
    for e in keep:
        v = e.get("verdict") or e.get("status")
        L += [f"### {e['title'].strip()}", "",
              f"**Verdict: {v}**  ·  registered {str(e.get('registered_at'))[:10]}"
              f"  ·  *in this package because it {why[e['id']]}*", ""]
        for field, head in (("hypothesis", "Hypothesis, frozen before the run"),
                            ("prediction", "Prediction, frozen before the run"),
                            ("power_statement", "What this test could and could not see")):
            if e.get(field):
                L += [f"**{head}.** {' '.join(str(e[field]).split())}", ""]
        res = (e.get("result") or {}).get("headline")
        if res:
            L += [f"**Result.** {' '.join(str(res).split())}", ""]
        if e.get("reproduce"):
            L += [f"**Reproduce.** `{e['reproduce']}`", ""]
        L += ["---", ""]

    L += ["## Instrument faults", "",
          "Each of these produced a number that parsed cleanly and was wrong. They are listed",
          "because the count of bugs a process caught is evidence about the process, and because",
          "several of them changed a headline figure after it had already been written down.", ""]
    for t in tk:
        L += [f"### {t['id']} · {' '.join(str(t['one_breath']).split())}", "",
              f"**What happened.** {' '.join(str(t['symptom']).split())}", "",
              f"**Why it parsed.** {' '.join(str(t['why_it_parsed']).split())}", "",
              f"**The defence.** {' '.join(str(t['defence']).split())}", "", "---", ""]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(L))
    print(f"{len(keep)} experiments ({len(ran)} ran), {len(tk)} faults -> {OUT} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")
    for e in keep:
        print(f"  {e['id']}  {str(e.get('verdict') or e.get('status')):18s} {e['title'][:58]}")
    print("  faults: " + ", ".join(t["id"] for t in tk))


if __name__ == "__main__":
    main()
