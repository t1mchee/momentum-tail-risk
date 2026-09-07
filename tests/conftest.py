"""Shared fixtures, and one rule about the distribution.

The submission package ships a pruned slice: about a third of the scripts, a row-and-column
filtered data directory, and the reports the three artifacts are composed from. That is
deliberate -- the brief says volume of code and data is not a quality measure -- but it means
some guards in this suite have no input to run against once the package is built.

A guard with no input must SKIP AND SAY WHY. It must not fail, because a wall of red in a
package whose subject is rigour reads as a broken deliverable rather than as a pruned one, and
it must not silently pass, because that is a guard that has stopped guarding.

The discriminator is `MANIFEST.json`, which the bundler writes at the package root and which
does not exist in the working repository. So in the working repository every one of these
absences is still a hard failure, which is where it should be: that is where the file being
missing means something is wrong rather than something was pruned.
"""
from __future__ import annotations

from pathlib import Path

import pytest

#: True only inside the built submission package.
IS_DISTRIBUTION = Path("MANIFEST.json").exists()


def needs(path: str | Path, what: str) -> None:
    """Skip, in the distribution only, when a deliberately pruned input is absent."""
    if Path(path).exists():
        return
    if IS_DISTRIBUTION:
        pytest.skip(f"not shipped in the submission package: {path} ({what})")
    raise FileNotFoundError(
        f"{path} is missing from the working repository ({what}). In the package this would be "
        f"a declared prune; here it is a fault.")
