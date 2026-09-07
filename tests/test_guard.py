"""Tests for the resource guard.

Written after a kernel panic. The guard's job is to stop a job before the machine stops, and
its one hard requirement is that it can tell a loaded machine from a quiet one -- a swap reader
that always returns zero is indistinguishable from a healthy system and would have watched the
panic happen exactly as the disk floor did.
"""

from __future__ import annotations

import pytest

from unstructured_momentum import guard


def test_swap_parser_reads_a_loaded_machine():
    """The failure mode: a parser that returns 0.0 on real input reads as 'healthy'."""
    loaded = "total = 9216.00M  used = 8082.00M  free = 1134.00M  (encrypted)"
    assert guard._SWAP.search(loaded), "regex did not match real sysctl output"
    m = guard._SWAP.search(loaded)
    assert abs(float(m.group(1)) / 1024 - 7.89) < 0.01


def test_swap_parser_reads_a_quiet_machine_as_zero():
    m = guard._SWAP.search("total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)")
    assert m and float(m.group(1)) == 0.0


def test_guard_raises_on_swap_not_only_disk():
    """The disk floor did not fire during the run that panicked; swap was what bound."""
    with pytest.raises(guard.ResourceExhausted, match="swap"):
        guard.check(min_disk_gb=0.0, max_swap_gb=-1.0, where="test")


def test_guard_raises_on_disk():
    with pytest.raises(guard.ResourceExhausted, match="disk free"):
        guard.check(min_disk_gb=10_000.0, max_swap_gb=999.0, where="test")


def test_guard_passes_when_healthy():
    guard.check(min_disk_gb=0.0, max_swap_gb=999.0, where="test")


def test_report_names_both_resources():
    r = guard.report()
    assert "disk" in r and "swap" in r


def test_headroom_names_the_panic_condition(monkeypatch):
    """The panic was swap with nowhere to grow. That is free swap PLUS free disk, and the
    original guard watched neither -- it watched absolute swap used, which does not tell a
    cornered machine apart from a busy but roomy one."""
    monkeypatch.setattr(guard, "swap_free_gb", lambda: 0.1)
    monkeypatch.setattr(guard, "free_disk_gb", lambda: 2.0)
    monkeypatch.setattr(guard, "swap_used_gb", lambda: 1.0)
    with pytest.raises(guard.ResourceExhausted, match="headroom"):
        guard.check(min_disk_gb=1.0, min_headroom_gb=3.0)


def test_headroom_lets_a_roomy_machine_work(monkeypatch):
    """Five gigabytes of stale swap with nine gigabytes of disk to grow into is not the panic
    state, and blocking a well-behaved job on that reading is a false positive that cost a
    rebuild on 2026-08-25."""
    monkeypatch.setattr(guard, "swap_free_gb", lambda: 1.0)
    monkeypatch.setattr(guard, "free_disk_gb", lambda: 9.0)
    monkeypatch.setattr(guard, "swap_used_gb", lambda: 5.0)
    guard.check(min_disk_gb=1.5, max_swap_gb=2.0, baseline_swap_gb=5.0, min_headroom_gb=3.0)


def test_growth_ceiling_catches_the_job_that_is_actually_thrashing(monkeypatch):
    """Same roomy machine, but this job has added three gigabytes of its own."""
    monkeypatch.setattr(guard, "swap_free_gb", lambda: 1.0)
    monkeypatch.setattr(guard, "free_disk_gb", lambda: 9.0)
    monkeypatch.setattr(guard, "swap_used_gb", lambda: 8.0)
    with pytest.raises(guard.ResourceExhausted, match="added by this job"):
        guard.check(min_disk_gb=1.5, max_swap_gb=2.0, baseline_swap_gb=5.0, min_headroom_gb=3.0)


def test_no_baseline_keeps_the_original_strict_reading(monkeypatch):
    """Callers that pass no baseline must behave exactly as before, or the change is a silent
    weakening of every guard in the repo rather than a new option."""
    monkeypatch.setattr(guard, "swap_free_gb", lambda: 1.0)
    monkeypatch.setattr(guard, "free_disk_gb", lambda: 9.0)
    monkeypatch.setattr(guard, "swap_used_gb", lambda: 5.0)
    with pytest.raises(guard.ResourceExhausted, match="of swap in use"):
        guard.check(min_disk_gb=1.5, max_swap_gb=4.0)
