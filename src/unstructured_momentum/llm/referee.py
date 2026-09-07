"""The referee: can a model name an experiment's structural flaw from its registration alone?

Rebuilt from its outputs, like the risk-pair module and for the same reason -- it was written,
run, cited under a supported experiment and a claim, and never committed.

The design is the interesting part and it is what the surviving files record. Eight registrations
were handed to a model with EVERY result stripped: no result block, no verdict, no reasoning, no
caveats, no history, no peek notes. The model saw only what a referee would see before the work
ran. It was asked to name the structural flaw that would make the experiment uninterpretable.
Its answers were then scored against the defect the project had actually recorded later -- a
truth set fixed by history rather than by the scorer's judgement at scoring time.

Three of eight hit, two partial, three miss. The comparison that matters is not the hit rate but
the alternative: a generic code-review checklist would have named none of the eight, because
every recorded defect is specific to the design in front of it -- a universe mismatch, a control
added mid-run, a null that absorbed the treatment.

What is NOT recovered is the generation call. The prompt is gone. The scoring, the truth set and
the score are all on disk, so the RESULT is reproducible and the generation is not; that split is
stated rather than papered over, and `--score` reproduces the recorded number from the surviving
answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTERIM = Path("data/interim")
INPUT = INTERIM / "referee_input.json"
OUTPUT = INTERIM / "referee_output.json"
SCORED = INTERIM / "referee_scored.json"
TRUTH = INTERIM / "referee_truth.json"

#: The recorded outcome a rebuild must return.
RECOVERY = {"hit": 3, "partial": 2, "miss": 3, "n": 8}

#: Fields stripped from a registration before the model sees it. The whole design rests on this
#: being complete: one leaked result turns a referee test into a reading-comprehension test.
WITHHELD = ("result", "verdict", "reasoning", "caveats", "history", "peek", "numbers", "headline")


class RecoveryFailed(RuntimeError):
    """The rebuild did not reproduce the recorded score from surviving answers."""


def blind(registration: dict) -> dict:
    """Strip every field that could reveal what happened. Used to build the prompt input."""
    return {k: v for k, v in registration.items()
            if not any(w in k.lower() for w in WITHHELD)}


def score() -> dict:
    """Reproduce the recorded score from the surviving answers and truth set."""
    for p in (OUTPUT, SCORED, TRUTH):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; the surviving outputs are the input here")
    scored = json.loads(SCORED.read_text())["score"]
    notes = json.loads(SCORED.read_text()).get("notes", {})
    tally = {k: sum(1 for v in scored.values() if v == k) for k in ("hit", "partial", "miss")}
    tally["n"] = len(scored)

    print(f"referee: {tally['n']} registrations, every result field withheld")
    for k in ("hit", "partial", "miss", "n"):
        want = RECOVERY[k]
        if tally[k] != want:
            raise RecoveryFailed(f"{k}: got {tally[k]}, recorded {want}")
        print(f"  {k:8s} {tally[k]}  recorded {want}  RECOVERED")

    print("\n  what it named, from the registration alone:")
    for eid, verdict in scored.items():
        if verdict == "hit":
            print(f"    {eid}  {notes.get(eid, '')[:88]}")
    print("\n  The comparison is not the hit rate. A generic code-review checklist names none of")
    print("  these: every recorded defect is specific to the design in front of it.")
    return tally


def verify_blinding() -> dict:
    """Confirm no result leaked into what the model was shown.

    The design's one hard requirement, and cheap to check against the surviving input.
    """
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    shown = json.loads(INPUT.read_text())
    leaks = {}
    for eid, payload in shown.items():
        blob = json.dumps(payload).lower()
        hit = [w for w in WITHHELD if f'"{w}"' in blob]
        if hit:
            leaks[eid] = hit
    print(f"blinding check across {len(shown)} registrations: "
          f"{'CLEAN' if not leaks else f'LEAKED {leaks}'}")
    return leaks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="referee", description=__doc__.split("\n")[0])
    ap.add_argument("--score", action="store_true", help="reproduce the recorded score")
    ap.add_argument("--verify-blinding", action="store_true",
                    help="confirm no result field reached the prompt")
    a = ap.parse_args(argv)
    if not (a.score or a.verify_blinding):
        ap.error("choose --score or --verify-blinding")
    if a.verify_blinding:
        verify_blinding()
        print()
    if a.score:
        score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
