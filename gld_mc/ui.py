#!/usr/bin/env python3
# Simple Tkinter UI for GLD Long Call Monte Carlo (Single + Batch tabs)
from __future__ import annotations
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .config import SimConfig
from .sim import simulate

PAD = 10

class SimUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GLD Long Call Monte Carlo")
        self.geometry("940x780")
        self._build_notebook()

    def _build_notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        # ---------- Single Run Tab ----------
        self.single_tab = ttk.Frame(nb)
        nb.add(self.single_tab, text="Single")

        self._build_single_tab(self.single_tab)

        # ---------- Batch Run Tab ----------
        self.batch_tab = ttk.Frame(nb)
        nb.add(self.batch_tab, text="Batch")

        self._build_batch_tab(self.batch_tab)

    # ======= SINGLE TAB =======
    def _build_single_tab(self, root):
        root = ttk.Frame(root, padding=PAD)
        root.pack(fill="both", expand=True)

        # --- Market & Contract
        msec = ttk.LabelFrame(root, text="Market & Contract", padding=PAD)
        msec.pack(fill="x", expand=False, pady=(0, PAD))

        self.var_spot   = tk.DoubleVar(value=364.38)
        self.var_strike = tk.DoubleVar(value=370.0)
        self.var_dte    = tk.IntVar(value=32)
        self.var_rfr    = tk.DoubleVar(value=0.02)
        self.var_annual = tk.IntVar(value=252)

        self._add_labeled_entry(msec, "Spot", self.var_spot, 0, 0)
        self._add_labeled_entry(msec, "Strike", self.var_strike, 0, 1)
        self._add_labeled_entry(msec, "DTE (calendar days)", self.var_dte, 1, 0)
        self._add_labeled_entry(msec, "Risk-free rate (annual)", self.var_rfr, 1, 1)
        self._add_labeled_entry(msec, "Trading days/year", self.var_annual, 2, 0)

        # --- Volatility
        vsec = ttk.LabelFrame(root, text="Volatility", padding=PAD)
        vsec.pack(fill="x", expand=False, pady=(0, PAD))

        self.var_iv_mode  = tk.StringVar(value="uniform")
        self.var_iv_fixed = tk.DoubleVar(value=0.21)
        self.var_iv_min   = tk.DoubleVar(value=0.17)
        self.var_iv_max   = tk.DoubleVar(value=0.25)

        self._add_combo(vsec, "IV mode", self.var_iv_mode, ["fixed", "uniform"], 0, 0)
        self._add_labeled_entry(vsec, "IV (fixed)", self.var_iv_fixed, 0, 1)
        self._add_labeled_entry(vsec, "IV min", self.var_iv_min, 1, 0)
        self._add_labeled_entry(vsec, "IV max", self.var_iv_max, 1, 1)

        # --- Simulation
        ssec = ttk.LabelFrame(root, text="Simulation", padding=PAD)
        ssec.pack(fill="x", expand=False, pady=(0, PAD))

        self.var_trials   = tk.IntVar(value=20000)
        self.var_seed     = tk.IntVar(value=7)

        self._add_labeled_entry(ssec, "Trials", self.var_trials, 0, 0)
        self._add_labeled_entry(ssec, "Seed", self.var_seed, 0, 1)

        # --- Trade Management
        tsec = ttk.LabelFrame(root, text="Trade Management", padding=PAD)
        tsec.pack(fill="x", expand=False, pady=(0, PAD))

        self.var_entry   = tk.DoubleVar(value=5.50)
        self.var_comm    = tk.DoubleVar(value=0.65)
        self.var_target  = tk.DoubleVar(value=800.0)
        self.var_stop    = tk.DoubleVar(value=3.00)
        self.var_avoid   = tk.IntVar(value=0)

        self._add_labeled_entry(tsec, "Entry price ($)", self.var_entry, 0, 0)
        self._add_labeled_entry(tsec, "Commission/side ($)", self.var_comm, 0, 1)
        self._add_labeled_entry(tsec, "Target profit ($)", self.var_target, 1, 0)
        self._add_labeled_entry(tsec, "Stop (option price)", self.var_stop, 1, 1)
        self._add_labeled_entry(tsec, "Avoid final N days (0=off)", self.var_avoid, 2, 0)

        # --- Drift
        dsec = ttk.LabelFrame(root, text="Drift (Price Dynamics)", padding=PAD)
        dsec.pack(fill="x", expand=False, pady=(0, PAD))

        self.var_mu_mode   = tk.StringVar(value="risk_neutral")
        self.var_mu_custom = tk.DoubleVar(value=0.10)

        self._add_combo(dsec, "μ mode", self.var_mu_mode, ["risk_neutral", "custom"], 0, 0)
        self._add_labeled_entry(dsec, "μ custom (annual)", self.var_mu_custom, 0, 1)

        # --- Buttons
        btn_row = ttk.Frame(root, padding=(0, PAD))
        btn_row.pack(fill="x", expand=False)
        ttk.Button(btn_row, text="Run Simulation", command=self._run_single).pack(side="left")
        ttk.Button(btn_row, text="Quit", command=self.destroy).pack(side="right")

        ttk.Label(root, text="Charts appear in a separate Results window. CSVs saved to ./out/").pack(
            side="bottom", anchor="w"
        )

    # ======= BATCH TAB =======
    def _build_batch_tab(self, root):
        root = ttk.Frame(root, padding=PAD)
        root.pack(fill="both", expand=True)

        # Reuse same-style blocks but with separate variables (so Single and Batch are independent)

        # --- Market & Contract
        msec = ttk.LabelFrame(root, text="Market & Contract (shared across strikes)", padding=PAD)
        msec.pack(fill="x", expand=False, pady=(0, PAD))

        self.b_spot   = tk.DoubleVar(value=364.38)
        self.b_dte    = tk.IntVar(value=32)
        self.b_rfr    = tk.DoubleVar(value=0.02)
        self.b_annual = tk.IntVar(value=252)

        self._add_labeled_entry(msec, "Spot", self.b_spot, 0, 0)
        self._add_labeled_entry(msec, "DTE (calendar days)", self.b_dte, 0, 1)
        self._add_labeled_entry(msec, "Risk-free rate (annual)", self.b_rfr, 1, 0)
        self._add_labeled_entry(msec, "Trading days/year", self.b_annual, 1, 1)

        # --- Strikes
        ssec = ttk.LabelFrame(root, text="Strikes (comma-separated)", padding=PAD)
        ssec.pack(fill="x", expand=False, pady=(0, PAD))
        self.b_strikes_text = tk.StringVar(value="365, 370, 375")
        frm = ttk.Frame(ssec)
        frm.grid(row=0, column=0, sticky="ew")
        frm.columnconfigure(0, weight=1)
        ttk.Entry(frm, textvariable=self.b_strikes_text, width=40).grid(row=0, column=0, sticky="ew", padx=(0, PAD))
        ttk.Label(frm, text="e.g., 365,370,375").grid(row=0, column=1, sticky="w")

        # --- Volatility
        vsec = ttk.LabelFrame(root, text="Volatility", padding=PAD)
        vsec.pack(fill="x", expand=False, pady=(0, PAD))

        self.b_iv_mode  = tk.StringVar(value="uniform")
        self.b_iv_fixed = tk.DoubleVar(value=0.21)
        self.b_iv_min   = tk.DoubleVar(value=0.17)
        self.b_iv_max   = tk.DoubleVar(value=0.25)

        self._add_combo(vsec, "IV mode", self.b_iv_mode, ["fixed", "uniform"], 0, 0)
        self._add_labeled_entry(vsec, "IV (fixed)", self.b_iv_fixed, 0, 1)
        self._add_labeled_entry(vsec, "IV min", self.b_iv_min, 1, 0)
        self._add_labeled_entry(vsec, "IV max", self.b_iv_max, 1, 1)

        # --- Simulation
        simsec = ttk.LabelFrame(root, text="Simulation", padding=PAD)
        simsec.pack(fill="x", expand=False, pady=(0, PAD))

        self.b_trials   = tk.IntVar(value=20000)
        self.b_seed     = tk.IntVar(value=7)

        self._add_labeled_entry(simsec, "Trials", self.b_trials, 0, 0)
        self._add_labeled_entry(simsec, "Seed", self.b_seed, 0, 1)

        # --- Trade Mgmt
        tsec = ttk.LabelFrame(root, text="Trade Management", padding=PAD)
        tsec.pack(fill="x", expand=False, pady=(0, PAD))

        self.b_entry   = tk.DoubleVar(value=5.50)
        self.b_comm    = tk.DoubleVar(value=0.65)
        self.b_target  = tk.DoubleVar(value=800.0)
        self.b_stop    = tk.DoubleVar(value=3.00)
        self.b_avoid   = tk.IntVar(value=0)

        self._add_labeled_entry(tsec, "Entry price ($)", self.b_entry, 0, 0)
        self._add_labeled_entry(tsec, "Commission/side ($)", self.b_comm, 0, 1)
        self._add_labeled_entry(tsec, "Target profit ($)", self.b_target, 1, 0)
        self._add_labeled_entry(tsec, "Stop (option price)", self.b_stop, 1, 1)
        self._add_labeled_entry(tsec, "Avoid final N days (0=off)", self.b_avoid, 2, 0)

        # --- Drift
        dsec = ttk.LabelFrame(root, text="Drift (Price Dynamics)", padding=PAD)
        dsec.pack(fill="x", expand=False, pady=(0, PAD))

        self.b_mu_mode   = tk.StringVar(value="risk_neutral")
        self.b_mu_custom = tk.DoubleVar(value=0.10)

        self._add_combo(dsec, "μ mode", self.b_mu_mode, ["risk_neutral", "custom"], 0, 0)
        self._add_labeled_entry(dsec, "μ custom (annual)", self.b_mu_custom, 0, 1)

        # --- Buttons
        btn_row = ttk.Frame(root, padding=(0, PAD))
        btn_row.pack(fill="x", expand=False)
        ttk.Button(btn_row, text="Run Batch", command=self._run_batch).pack(side="left")
        ttk.Button(btn_row, text="Quit", command=self.destroy).pack(side="right")

        ttk.Label(root, text="Overlaid histograms will show each strike with a different color + legend. CSVs saved to ./out/").pack(
            side="bottom", anchor="w"
        )

    # ======= Helpers =======
    def _add_labeled_entry(self, parent, text, var, row, col):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=col, sticky="w", padx=(0, PAD), pady=5)
        ttk.Label(frm, text=text).pack(side="top", anchor="w")
        ttk.Entry(frm, textvariable=var, width=22).pack(side="top", anchor="w")

    def _add_combo(self, parent, label, var, values, row, col):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=col, sticky="w", padx=(0, PAD), pady=5)
        ttk.Label(frm, text=label).pack(side="top", anchor="w")
        ttk.Combobox(frm, textvariable=var, values=values, width=18, state="readonly").pack(side="top", anchor="w")

    def _collect_config_single(self) -> SimConfig:
        return SimConfig(
            spot=self.var_spot.get(),
            strike=self.var_strike.get(),
            dte_calendar=self.var_dte.get(),
            annual_trading_days=self.var_annual.get(),
            risk_free_rate=self.var_rfr.get(),
            iv_mode=self.var_iv_mode.get(),
            iv_fixed=self.var_iv_fixed.get(),
            iv_min=self.var_iv_min.get(),
            iv_max=self.var_iv_max.get(),
            num_trials=self.var_trials.get(),
            seed=self.var_seed.get(),
            entry_price=self.var_entry.get(),
            commission_per_side=self.var_comm.get(),
            target_profit=self.var_target.get(),
            stop_option_price=self.var_stop.get(),
            avoid_final_days=self.var_avoid.get(),
            mu_mode=self.var_mu_mode.get(),
            mu_custom=self.var_mu_custom.get(),
        )

    def _collect_config_batch_common(self) -> SimConfig:
        # shared settings across all strikes in the batch
        return SimConfig(
            spot=self.b_spot.get(),
            dte_calendar=self.b_dte.get(),
            annual_trading_days=self.b_annual.get(),
            risk_free_rate=self.b_rfr.get(),
            iv_mode=self.b_iv_mode.get(),
            iv_fixed=self.b_iv_fixed.get(),
            iv_min=self.b_iv_min.get(),
            iv_max=self.b_iv_max.get(),
            num_trials=self.b_trials.get(),
            seed=self.b_seed.get(),
            entry_price=self.b_entry.get(),
            commission_per_side=self.b_comm.get(),
            target_profit=self.b_target.get(),
            stop_option_price=self.b_stop.get(),
            avoid_final_days=self.b_avoid.get(),
            mu_mode=self.b_mu_mode.get(),
            mu_custom=self.b_mu_custom.get(),
        )

    # ======= Actions =======
    def _run_single(self):
        try:
            cfg = self._collect_config_single()
        except Exception as e:
            messagebox.showerror("Input error", f"Could not read inputs:\n{e}")
            return

        summary, details = simulate(cfg)
        os.makedirs("out", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = f"GLD_{int(cfg.strike)}C_{cfg.dte_calendar}DTE_tr{cfg.num_trials}_{stamp}"
        summary.to_csv(os.path.join("out", f"{tag}_summary.csv"), index=False)
        details.to_csv(os.path.join("out", f"{tag}_details.csv"), index=False)

        self._show_results_window_single(details, tag)

    def _run_batch(self):
        # Parse strikes
        try:
            strikes = [float(s.strip()) for s in self.b_strikes_text.get().split(",") if s.strip()]
            if not strikes:
                raise ValueError("Provide at least one strike (e.g., 365, 370, 375).")
        except Exception as e:
            messagebox.showerror("Input error", f"Bad strikes list:\n{e}")
            return

        base = self._collect_config_batch_common()
        os.makedirs("out", exist_ok=True)

        results = []  # list of (strike, summary_df, details_df)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        for k in strikes:
            cfg = SimConfig(**{**base.__dict__, "strike": k})
            summary, details = simulate(cfg)

            tag = f"GLD_{int(k)}C_{cfg.dte_calendar}DTE_tr{cfg.num_trials}_{stamp}"
            summary.to_csv(os.path.join("out", f"{tag}_summary.csv"), index=False)
            details.to_csv(os.path.join("out", f"{tag}_details.csv"), index=False)

            s_copy = summary.copy()
            s_copy.insert(0, "Strike", k)
            results.append((k, s_copy, details))

        # Save combined batch summary
        combo = pd.concat([r[1] for r in results], ignore_index=True)
        combo.to_csv(os.path.join("out", f"batch_summary_{stamp}.csv"), index=False)

        # Show batch results window (overlaid charts)
        self._show_results_window_batch(results, stamp)

    # ======= Result windows =======
    def _show_results_window_single(self, details: pd.DataFrame, tag: str):
        win = tk.Toplevel(self)
        win.title(f"Results — {tag}")
        win.geometry("1150x720")

        container = ttk.Frame(win, padding=PAD)
        container.pack(fill="both", expand=True)

        # Figure 1: Final P&L histogram
        fig1 = plt.Figure(figsize=(6.4, 4.8), dpi=100)
        ax1 = fig1.add_subplot(111)
        ax1.hist(details["final_pl"], bins=80)
        ax1.set_title("Final P&L Distribution")
        ax1.set_xlabel("P&L per contract ($)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        canvas1 = FigureCanvasTkAgg(fig1, master=container)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side="left", fill="both", expand=True, padx=(0, PAD))

        # Figure 2: Exit Day histogram (target hits only)
        fig2 = plt.Figure(figsize=(6.4, 4.8), dpi=100)
        ax2 = fig2.add_subplot(111)
        if details["hit_target"].any():
            ax2.hist(details.loc[details["hit_target"], "hit_day"], bins=range(1, 60))
        ax2.set_title("Exit Day Distribution (Hit Target)")
        ax2.set_xlabel("Trading day of exit")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
        canvas2 = FigureCanvasTkAgg(fig2, master=container)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)

        ttk.Button(win, text="Close", command=win.destroy).pack(side="bottom", pady=PAD)

    def _show_results_window_batch(self, results, stamp_tag: str):
        """
        results: list of (strike, summary_df, details_df)
        Draw overlaid histograms with distinct colors + legend.
        """
        win = tk.Toplevel(self)
        win.title(f"Batch Results — {stamp_tag}")
        win.geometry("1200x780")

        container = ttk.Frame(win, padding=PAD)
        container.pack(fill="both", expand=True)

        # Qualitative palette for distinct colors (user requested distinct colors + labels)
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]

        # --- Figure 1: Overlaid Final P&L hist
        fig1 = plt.Figure(figsize=(6.8, 5.0), dpi=100)
        ax1 = fig1.add_subplot(111)

        for idx, (k, _summary, details) in enumerate(results):
            color = palette[idx % len(palette)]
            ax1.hist(
                details["final_pl"],
                bins=70,
                alpha=0.35,
                label=f"{int(k)}C",
                color=color
            )

        ax1.set_title("Final P&L Distribution — Overlaid by Strike")
        ax1.set_xlabel("P&L per contract ($)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        ax1.legend(title="Strike")

        canvas1 = FigureCanvasTkAgg(fig1, master=container)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side="left", fill="both", expand=True, padx=(0, PAD))

        # --- Figure 2: Overlaid Exit Day hist (hits only)
        fig2 = plt.Figure(figsize=(6.8, 5.0), dpi=100)
        ax2 = fig2.add_subplot(111)

        for idx, (k, _summary, details) in enumerate(results):
            color = palette[idx % len(palette)]
            if details["hit_target"].any():
                ax2.hist(
                    details.loc[details["hit_target"], "hit_day"],
                    bins=range(1, 60),
                    alpha=0.45,
                    label=f"{int(k)}C",
                    color=color
                )

        ax2.set_title("Exit Day Distribution (Hit Target) — Overlaid by Strike")
        ax2.set_xlabel("Trading day of exit")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
        ax2.legend(title="Strike")

        canvas2 = FigureCanvasTkAgg(fig2, master=container)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)

        ttk.Button(win, text="Close", command=win.destroy).pack(side="bottom", pady=PAD)

def main():
    app = SimUI()
    app.mainloop()

if __name__ == "__main__":
    main()
