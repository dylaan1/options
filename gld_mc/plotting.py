from __future__ import annotations

import math
from collections.abc import Sequence

import os
import matplotlib.pyplot as plt
import pandas as pd


def extract_hit_days(
    details: pd.DataFrame,
    *,
    day_column: str = "hit_day",
    target_column: str = "hit_target",
) -> pd.Series:
    """Return the trading-day values for rows that actually exited via the target."""

    if day_column not in details.columns:
        return pd.Series(dtype=float)

    hit_day_values = details[day_column]
    if target_column in details.columns:
        mask = details[target_column].fillna(False)
        hit_day_values = hit_day_values[mask]

    return pd.to_numeric(hit_day_values, errors="coerce").dropna()


def compute_hit_day_bin_edges(
    hit_days: Sequence[float] | pd.Series,
    *,
    first_day: int = 1,
) -> list[int]:
    """Compute 1-day histogram bin edges that include the latest exit."""

    if isinstance(hit_days, pd.Series):
        series = pd.to_numeric(hit_days, errors="coerce").dropna()
    else:
        series = pd.to_numeric(pd.Series(hit_days), errors="coerce").dropna()

    if series.empty:
        return list(range(first_day, first_day + 2))

    max_day = max(first_day, int(math.ceil(series.max())))
    return list(range(first_day, max_day + 2))


def plot_results(details: pd.DataFrame, out_dir: str, tag: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.hist(details["final_pl"], bins=80)
    plt.title(f"Final P&L Distribution — {tag}")
    plt.xlabel("P&L per contract ($)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_pnl_hist.png"), dpi=140)
    plt.close()

    if details["hit_target"].any():
        plt.figure()
        hit_days = details.loc[details["hit_target"], "hit_day"].dropna()
        if len(hit_days) > 0:
            max_day = int(hit_days.max())
            bin_edges = range(1, max_day + 2)
        else:
            bin_edges = range(1, 2)
        plt.hist(hit_days, bins=bin_edges)
        plt.title(f"Exit Day Distribution (Hit Target) — {tag}")
        plt.xlabel("Trading day of exit")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_exit_day_hist.png"), dpi=140)
        plt.close()
