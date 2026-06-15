"""Trivial passing test — proves the test harness + import path work."""

from __future__ import annotations

import mlops_drift


def test_package_imports():
    assert mlops_drift.__version__ == "0.1.0"


def test_arithmetic_sanity():
    assert 2 + 2 == 4
