from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from gld_mc.plotting import compute_hit_day_bin_edges, extract_hit_days


def test_extract_hit_days_filters_and_orders() -> None:
    details = pd.DataFrame(
        {
            "hit_target": [True, False, True, True],
            "hit_day": [1, 2, float("nan"), 7],
        }
    )

    result = extract_hit_days(details)

    assert list(result) == [1.0, 7.0]


def test_compute_hit_day_bin_edges_includes_latest_exit() -> None:
    details = pd.DataFrame(
        {
            "hit_target": [True, True, True],
            "hit_day": [1, 4, 15],
        }
    )
    hit_days = extract_hit_days(details)

    edges = compute_hit_day_bin_edges(hit_days)

    assert edges == list(range(1, 17))


def test_compute_hit_day_bin_edges_with_empty_series() -> None:
    details = pd.DataFrame(
        {
            "hit_target": [False, False],
            "hit_day": [None, None],
        }
    )
    hit_days = extract_hit_days(details)

    edges = compute_hit_day_bin_edges(hit_days)

    assert edges == [1, 2]
