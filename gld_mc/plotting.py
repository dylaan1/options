from __future__ import annotations

import math
from collections.abc import Sequence

import os
import matplotlib.pyplot as plt
import pandas as pd

from .analytics import exit_day_bin_edges

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
        hit_days = details.loc[details["hit_target"], "hit_day"]
        bin_edges = exit_day_bin_edges(hit_days)
        plt.hist(pd.to_numeric(hit_days, errors="coerce").dropna(), bins=bin_edges)
        plt.title(f"Exit Day Distribution (Hit Target) — {tag}")
        plt.xlabel("Trading day of exit")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_exit_day_hist.png"), dpi=140)
        plt.close()
