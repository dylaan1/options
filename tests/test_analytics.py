from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gld_mc.analytics import exit_day_bin_edges


def test_exit_day_bin_edges_returns_default_for_empty_iterable() -> None:
    edges = exit_day_bin_edges([])
    assert edges == [1, 2]


def test_exit_day_bin_edges_uses_max_hit_day() -> None:
    pd = pytest.importorskip("pandas")
    if not hasattr(pd, "Series"):
        pytest.skip("pandas Series unavailable")
    series = pd.Series([1, 2, float("nan"), 4.2, 7.9])
    edges = exit_day_bin_edges(series)
    assert edges == list(range(1, 9))
