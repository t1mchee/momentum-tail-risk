"""Resource guards for long local jobs, so a job dies instead of the machine.

This exists because the machine kernel-panicked. Not from a bug in any calculation: a section
extraction held every filing's text in one list, swap grew from 1 GB to 10 GB against a 9.2 GB
backing store on a volume with 1.7 GB free, and with nowhere for swap to grow, page faults
queued until `watchdogd` missed its check-in for 92 seconds and the kernel force-panicked.

The lesson is which resource binds. Every long job here already had a DISK floor, and the disk
floor did not fire, because disk was not the binding constraint -- swap was, and swap is invisible
to a `shutil.disk_usage` check until it has already eaten the volume. A guard that watches only
free bytes will watch this failure happen.

Reboot reclaimed everything, which is the other lesson: swap does not shrink while the machine
runs. Noticing that and continuing anyway is how the crash happened, twice noted and twice
ignored.
"""

from __future__ import annotations

import re
import shutil
import subprocess


class ResourceExhausted(RuntimeError):
    """Raised when a guard trips. Callers should exit cleanly, not retry."""


#: sysctl prints "total = 9216.00M  used = 8082.00M  free = 1134.00M". A first version walked
#: tokens and index-searched the string for context; it returned 0.0 on real input and could not
#: be told apart from a machine with no swap, which is the exact reading a guard must never get
#: wrong. One regex, tested against a known-loaded string rather than against a quiet machine.
_SWAP = re.compile(r"used\s*=\s*([0-9.]+)\s*([MG])", re.I)
_SWAP_FREE = re.compile(r"free\s*=\s*([0-9.]+)\s*([MG])", re.I)


def _gb(m) -> float:
    v = float(m.group(1))
    return v / 1024.0 if m.group(2).upper() == "M" else v


def swap_used_gb() -> float:
    """Swap in use, in GB. Returns 0.0 only when swap really is zero or sysctl is unavailable."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return 0.0
    m = _SWAP.search(out)
    return _gb(m) if m else 0.0


def swap_free_gb() -> float:
    """Unused space in the swap backing store, in GB."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return 0.0
    m = _SWAP_FREE.search(out)
    return _gb(m) if m else 0.0


def headroom_gb() -> float:
    """How much further the machine can page before it is cornered.

    macOS grows the swap backing store into free disk on demand, so the resource that actually
    runs out is the SUM of free swap and free disk. The panic this module exists to prevent was
    exactly that sum going to zero: a 9.2 GB store with nowhere left on a volume holding 1.7 GB.
    Absolute swap-used, the quantity the first version watched, does not distinguish that state
    from a machine carrying a large but stable baseline with eight gigabytes of disk to grow
    into -- and it was the latter reading, at 5.0 GB against a 4.0 GB ceiling, that refused to
    let a well-behaved job start on 2026-08-25.
    """
    return swap_free_gb() + free_disk_gb()


def free_disk_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def check(*, min_disk_gb: float = 2.0, max_swap_gb: float = 4.0, where: str = "job",
          baseline_swap_gb: float | None = None, min_headroom_gb: float = 3.0) -> None:
    """Raise before the machine is in trouble, rather than after.

    Three tests, because the first version had one and it was the wrong one.

    ``min_disk_gb`` is the original floor. It did not fire during the panic and is kept only
    because a full volume corrupts a part-written store.

    ``min_headroom_gb`` is the test that names the panic condition directly: free swap plus free
    disk, the pool macOS can still page into. This is the one that would have fired.

    ``max_swap_gb`` is now a ceiling on swap this JOB caused, measured against
    ``baseline_swap_gb`` captured at start. A machine carrying five gigabytes of stale Electron
    pages is not a job thrashing, and treating the two as the same reading blocks work that is
    behaving. Passing no baseline keeps the original absolute reading, so existing callers are
    unchanged.
    """
    d, s, h = free_disk_gb(), swap_used_gb(), headroom_gb()
    if d < min_disk_gb:
        raise ResourceExhausted(
            f"{where}: {d:.1f} GB disk free, floor {min_disk_gb}. Stopping while the store is "
            "still consistent.")
    if h < min_headroom_gb:
        raise ResourceExhausted(
            f"{where}: {h:.1f} GB of headroom left (free swap plus free disk), floor "
            f"{min_headroom_gb}. This is the state the machine panicked in: swap with nowhere "
            "to grow.")
    caused = s if baseline_swap_gb is None else s - baseline_swap_gb
    if caused > max_swap_gb:
        label = ("of swap in use" if baseline_swap_gb is None
                 else f"of swap added by this job (baseline {baseline_swap_gb:.1f} GB)")
        raise ResourceExhausted(
            f"{where}: {caused:.1f} GB {label}, ceiling {max_swap_gb}. The process is "
            "thrashing; swap does not shrink until reboot, so continuing risks the machine "
            "rather than the job.")


def report() -> str:
    return (f"disk {free_disk_gb():.1f} GB free, swap {swap_used_gb():.1f} GB used, "
            f"headroom {headroom_gb():.1f} GB")
