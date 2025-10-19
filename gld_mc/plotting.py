from __future__ import annotations
import os
import matplotlib.pyplot as plt
import pandas as pd

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
