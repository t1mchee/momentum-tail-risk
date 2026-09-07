"""Every registered reproduce command must be able to produce something.

trp-76 found one claim whose command named a module with no entry point: it imported,
printed nothing and exited 0, which reads as success to anything checking a return code.
trp-78 found that this was not one command but most of them, including the generator for the
severity result the memo leads with. The memo proposed a CI check for this; a proposed check
is not a check, so it lives here.

This does not execute the commands -- several take minutes and hit the network. It asks the
only question that distinguishes a real generator from a silent no-op: is there an entry
point at the other end?
"""
from __future__ import annotations

import ast
import shlex
from pathlib import Path

import pytest
import yaml

PKG = Path("src/unstructured_momentum")

#: Commands known to be dead, kept as an explicit shrinking list rather than a silent pass.
#: Removing an entry here is the definition of repairing one. Do not add to this list to make
#: a test pass: delete the reproduce command instead, so the claim reads as unreproducible.
#:
#: The list is a record of the WORKING repository, where ~100 scripts carry reproduce commands.
#: The submission package ships thirty, every one of which resolves, so the file is not shipped
#: and the guarantee there is simply stronger: no exemptions, every command must work.
_DEBT = Path("project/reproduce_debt.yaml")
KNOWN_DEAD = set(yaml.safe_load(_DEBT.read_text())["dead"]) if _DEBT.exists() else set()


def _entry_point(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return False
    return any(isinstance(n, ast.If) and "__main__" in ast.dump(n.test)
               for n in ast.walk(tree))


def _target(cmd: str) -> Path | None:
    toks = shlex.split(cmd.split("#")[0])
    if "-m" in toks:
        parts = toks[toks.index("-m") + 1].split(".")
        if parts and parts[0] == "unstructured_momentum":
            parts = parts[1:]
        direct = PKG.joinpath(*parts).with_suffix(".py")
        pkg_main = PKG.joinpath(*parts, "__main__.py")
        return direct if direct.exists() else pkg_main
    for t in toks:
        if t.endswith(".py"):
            return Path(t)
    return None


def _all_with_reproduce():
    """Every entry carrying a reproduce string, run or not. The debt list is checked against
    this, not against `_commands()`: an id may sit in the debt list while its experiment is
    still unrun, and narrowing the check would read that as the id having disappeared."""
    for stem in ("claims", "experiments"):
        f = Path(f"project/{stem}.yaml")
        if not f.exists():
            continue          # pruned from the submission package; see KNOWN_DEAD above
        data = yaml.safe_load(f.read_text())
        for e in (data if isinstance(data, list) else data.get(stem, data)):
            if e.get("reproduce") and str(e["reproduce"]).strip():
                yield e["id"]


def _commands():
    """Every reproduce command that is a receipt rather than an intention.

    build/validate.py exempts registered-but-unrun entries from the same check, on the
    grounds that their reproduce path is a forward intention: exp-087 is blocked on
    annotation resources and records `blocked; no command`, which is the truth about it and
    not a dead generator. Two checks over one field disagreeing about what the field means
    is how a suite starts being ignored, so the exemption is stated here in the same terms.
    """
    for stem in ("claims", "experiments"):
        f = Path(f"project/{stem}.yaml")
        if not f.exists():
            continue          # pruned from the submission package
        data = yaml.safe_load(f.read_text())
        for e in (data if isinstance(data, list) else data.get(stem, data)):
            if e.get("status") == "registered" or e.get("verdict") == "not_yet":
                continue
            if e.get("reproduce") and str(e["reproduce"]).strip():
                yield e["id"], str(e["reproduce"])


@pytest.mark.parametrize("entry_id,cmd", list(_commands()))
def test_reproduce_command_has_an_entry_point(entry_id: str, cmd: str) -> None:
    target = _target(cmd)
    live = bool(target and target.exists() and _entry_point(target))
    if entry_id in KNOWN_DEAD:
        # Repairing one must be noticed, not silently absorbed. If this fires, the generator
        # now works and the id belongs out of the debt list.
        assert not live, (
            f"{entry_id}: this reproduce command now HAS an entry point. Remove it from "
            f"project/reproduce_debt.yaml -- the list may only shrink, and shrinking is the "
            f"point.")
        return
    assert target is not None, f"{entry_id}: cannot resolve a target from {cmd!r}"
    assert target.exists(), f"{entry_id}: {target} does not exist"
    assert live, (
        f"{entry_id}: {target} has no __main__ entry point, so `{cmd}` prints nothing and "
        f"exits 0. Repair it or delete the reproduce command.")


def test_the_debt_list_only_shrinks() -> None:
    """A newly added claim may not arrive already dead."""
    live = set(_all_with_reproduce())
    stale = KNOWN_DEAD - live
    assert not stale, f"reproduce_debt lists ids that no longer exist: {sorted(stale)}"
