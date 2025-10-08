#!/usr/bin/env python3
# Simple Tkinter UI for GLD Long Call Monte Carlo (Single + Batch tabs)
from __future__ import annotations

import os
import time
import tkinter as tk
from datetime import datetime, timezone
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .config import DataProviderConfig, SimConfig
from .data_provider import QuoteStreamHandle, create_data_provider
from .sim import simulate

PAD = 10


def contract_key(row: pd.Series | None) -> Optional[tuple[Any, ...]]:
    if row is None:
        return None
    return (
        (row.get("symbol") or "").upper(),
        row.get("expiration"),
        (row.get("option_type") or "").lower(),
        round(float(row.get("strike", 0.0)), 4),
    )


class ContractCell(tk.Frame):
    """Visual representation of a single call or put contract."""

    DEFAULT_BG = "white"
    SELECT_BG = "#fffbe6"
    SELECT_BORDER = "#8cb6ff"

    def __init__(
        self,
        viewer: "OptionsChainViewer",
        parent,
        contract: pd.Series | None,
        *,
        side: str,
    ) -> None:
        super().__init__(parent, bd=1, relief="solid", background=self.DEFAULT_BG, highlightthickness=0)
        self.viewer = viewer
        if contract is not None and not isinstance(contract, pd.Series):
            contract = pd.Series(contract)
        self.contract: pd.Series | None = contract.copy() if isinstance(contract, pd.Series) else None
        self.side = side
        self.key = contract_key(self.contract) if self.contract is not None else None

        text = self._build_text()
        self.label = tk.Label(
            self,
            text=text,
            justify="left",
            anchor="nw",
            bg=self.DEFAULT_BG,
            font=("TkDefaultFont", 9),
        )
        self.label.pack(fill="both", expand=True, padx=6, pady=4)

        if self.contract is not None:
            for widget in (self, self.label):
                widget.bind("<Button-1>", self._on_click)
                widget.bind("<Double-Button-1>", self._on_double_click)

    def _build_text(self) -> str:
        metrics = self.viewer.metrics
        header = (self.contract.get("option_type") if self.contract is not None else self.side).upper()
        lines = [header]
        for label, column in metrics:
            value = self.contract.get(column) if self.contract is not None else None
            lines.append(f"{label}: {self.viewer.format_metric(column, value)}")
        return "\n".join(lines)

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.configure(
                highlightthickness=2,
                highlightbackground=self.SELECT_BORDER,
                highlightcolor=self.SELECT_BORDER,
                background=self.SELECT_BG,
            )
            self.label.configure(bg=self.SELECT_BG)
        else:
            self.configure(highlightthickness=0, background=self.DEFAULT_BG)
            self.label.configure(bg=self.DEFAULT_BG)

    def _on_click(self, _event) -> None:
        self.viewer._handle_cell_click(self, double=False)

    def _on_double_click(self, _event) -> None:
        self.viewer._handle_cell_click(self, double=True)


class OptionsChainViewer(ttk.Frame):
    """Scrollable matrix of call/put contracts grouped by strike."""

    metrics = [
        ("DTE", "dte"),
        ("Mark", "mark"),
        ("Trade Price", "trade_price"),
        ("P/L Open", "pl_open"),
        ("P/L %", "pl_pct"),
        ("Delta", "delta"),
        ("Theta", "theta"),
        ("Vega", "vega"),
        ("IV%", "iv_percent"),
        ("Volume", "volume"),
        ("Open Interest", "open_interest"),
    ]

    def __init__(
        self,
        parent,
        on_select: Callable[[pd.Series], None] | None = None,
        *,
        on_activate: Callable[[pd.Series], None] | None = None,
    ) -> None:
        super().__init__(parent, padding=PAD)

        self._on_select = on_select
        self._on_activate = on_activate
        self._data = pd.DataFrame()
        self._cells_by_key: dict[tuple[Any, ...], ContractCell] = {}
        self._selected_cell: ContractCell | None = None
        self._selected_key: tuple[Any, ...] | None = None

        self._last_update = tk.StringVar(value="")
        self._underlying_var = tk.StringVar(value="")

        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, PAD))
        ttk.Label(header, text="Live Option Chain", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self._underlying_var).pack(side="right")
        ttk.Label(header, textvariable=self._last_update).pack(side="right", padx=(0, PAD))

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        vsb.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=vsb.set)

        self._table = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self._table, anchor="nw")
        self._table.bind("<Configure>", self._on_table_configure)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_table_configure(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_resize(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def clear(self) -> None:
        for child in self._table.winfo_children():
            child.destroy()
        self._cells_by_key.clear()
        if self._selected_cell is not None:
            self._selected_cell.set_selected(False)
        self._selected_cell = None
        self._selected_key = None

    def update_underlying(self, price: float | None) -> None:
        if price is None:
            self._underlying_var.set("")
        else:
            self._underlying_var.set(f"Spot: {price:.2f}")

    def update_from_dataframe(self, df: pd.DataFrame, timestamp: str | None = None) -> None:
        current_key = self._selected_key
        self._data = df.reset_index(drop=True)
        self.clear()

        if df.empty:
            self._last_update.set("No data")
        else:
            grouped: dict[tuple[Any, float], dict[str, pd.Series]] = {}
            for _, row in self._data.iterrows():
                strike = float(row.get("strike", 0.0))
                expiration = row.get("expiration")
                option_type = (row.get("option_type") or "").lower()
                key = (expiration, strike)
                grouped.setdefault(key, {})[option_type] = row

            self._build_header_row()

            for idx, ((expiration, strike), legs) in enumerate(
                sorted(grouped.items(), key=lambda item: (item[0][0] or "", item[0][1]))
            , start=1):
                call_row = legs.get("call")
                put_row = legs.get("put")
                call_cell = self._create_cell(call_row, side="call")
                call_cell.grid(row=idx, column=0, sticky="nsew", padx=(0, PAD // 2), pady=4)

                strike_text = f"{strike:.2f}\n{expiration or ''}"
                ttk.Label(
                    self._table,
                    text=strike_text,
                    anchor="center",
                    font=("TkDefaultFont", 10, "bold"),
                ).grid(row=idx, column=1, sticky="nsew", padx=4, pady=4)

                put_cell = self._create_cell(put_row, side="put")
                put_cell.grid(row=idx, column=2, sticky="nsew", padx=(PAD // 2, 0), pady=4)

                self._table.rowconfigure(idx, weight=1)

            self._table.columnconfigure(0, weight=1)
            self._table.columnconfigure(1, weight=0)
            self._table.columnconfigure(2, weight=1)

            if timestamp:
                self._last_update.set(f"Updated {timestamp}")
            else:
                self._last_update.set("Updated")

            if current_key is not None:
                self.select_by_key(current_key)

    def _build_header_row(self) -> None:
        ttk.Label(
            self._table,
            text="CALLS",
            font=("TkDefaultFont", 10, "bold"),
            anchor="center",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(
            self._table,
            text="STRIKE",
            font=("TkDefaultFont", 10, "bold"),
            anchor="center",
        ).grid(row=0, column=1, sticky="ew", pady=(0, 4))
        ttk.Label(
            self._table,
            text="PUTS",
            font=("TkDefaultFont", 10, "bold"),
            anchor="center",
        ).grid(row=0, column=2, sticky="ew", pady=(0, 4))

    def _create_cell(self, row: pd.Series | None, *, side: str) -> ContractCell:
        cell = ContractCell(self, self._table, row, side=side)
        if cell.contract is not None and cell.key is not None:
            self._cells_by_key[cell.key] = cell
        return cell

    def _handle_cell_click(self, cell: ContractCell, *, double: bool) -> None:
        if cell.contract is None:
            return
        self._set_selection(cell, notify=not double)
        if double and self._on_activate is not None:
            self._on_activate(cell.contract)
        elif not double and self._on_select is not None:
            self._on_select(cell.contract)

    def _set_selection(self, cell: ContractCell, *, notify: bool = True) -> None:
        if self._selected_cell is cell:
            return
        if self._selected_cell is not None:
            self._selected_cell.set_selected(False)
        self._selected_cell = cell
        self._selected_key = cell.key
        cell.set_selected(True)
        if notify and self._on_select is not None and cell.contract is not None:
            self._on_select(cell.contract)

    def clear_selection(self) -> None:
        if self._selected_cell is not None:
            self._selected_cell.set_selected(False)
        self._selected_cell = None
        self._selected_key = None

    def select_by_key(self, key: tuple[Any, ...]) -> None:
        cell = self._cells_by_key.get(key)
        if cell is not None:
            self._set_selection(cell, notify=False)

    def get_selected_row(self) -> pd.Series | None:
        if self._selected_cell is not None:
            return self._selected_cell.contract
        return None

    def format_metric(self, column: str, value: Any) -> str:
        if value is None or (isinstance(value, str) and not value) or pd.isna(value):
            return "-f"
        try:
            if column == "dte":
                return f"{int(round(float(value)))}"
            if column in {"mark", "trade_price", "pl_open"}:
                return f"{float(value):.2f}"
            if column in {"pl_pct", "iv_percent"}:
                pct = float(value)
                if column == "iv_percent" and abs(pct) < 1:
                    pct *= 100.0
                return f"{pct:.2f}%"
            if column in {"delta", "theta", "vega"}:
                return f"{float(value):.4f}"
            if column in {"volume", "open_interest"}:
                return f"{int(round(float(value)))}"
        except Exception:  # noqa: BLE001
            return str(value)
        return str(value)

class SimUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options Monte Carlo Simulator")
        self.geometry("1050x820")
        self.data_settings = DataProviderConfig()
        self.data_provider = create_data_provider(self.data_settings)
        self._chain_handle: QuoteStreamHandle | None = None
        self._latest_chain: pd.DataFrame | None = None
        self._latest_underlying_price: float | None = None
        self._selected_contract: pd.Series | None = None
        self._selected_contract_key: Optional[tuple[Any, ...]] = None
        self._selection_status = tk.StringVar(value="No contract selected")
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

        default_symbol = self.data_settings.default_symbol or "GLD"

        # --- Instrument ---
        isec = ttk.LabelFrame(root, text="Instrument", padding=PAD)
        isec.pack(fill="x", expand=False, pady=(0, PAD))

        self.var_symbol = tk.StringVar(value=default_symbol)
        self.var_option_type = tk.StringVar(value=self.data_settings.default_option_type)
        self.var_expiration = tk.StringVar(value=self.data_settings.default_expiration or "")

        self._add_labeled_entry(isec, "Symbol", self.var_symbol, 0, 0)
        self._add_combo(isec, "Option type", self.var_option_type, ["call", "put"], 0, 1)
        self._add_labeled_entry(isec, "Expiration (YYYY-MM-DD)", self.var_expiration, 1, 0)
        mult_frame = ttk.Frame(isec)
        mult_frame.grid(row=1, column=1, sticky="w", padx=(0, PAD), pady=5)
        ttk.Label(mult_frame, text="Contract multiplier").pack(side="top", anchor="w")
        ttk.Label(mult_frame, text="100 (fixed)").pack(side="top", anchor="w")

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

        # --- Live Market Data
        mdat = ttk.LabelFrame(root, text="Live Market Data", padding=PAD)
        mdat.pack(fill="both", expand=True, pady=(0, PAD))

        controls = ttk.Frame(mdat)
        controls.pack(fill="x", pady=(0, PAD))
        ttk.Button(controls, text="Start Stream", command=self._start_chain_stream).pack(side="left")
        ttk.Button(controls, text="Stop Stream", command=self._stop_chain_stream).pack(side="left", padx=(PAD, 0))
        ttk.Button(controls, text="Simulate Trade", command=self._select_contract_for_simulation).pack(
            side="left", padx=(PAD, 0)
        )
        ttk.Button(controls, text="Copy Last Chain", command=self._copy_chain_to_clipboard).pack(side="right")

        self.chain_view = OptionsChainViewer(
            mdat,
            on_select=self._handle_chain_preview,
            on_activate=self._on_contract_activate,
        )
        self.chain_view.pack(fill="both", expand=True)
        ttk.Label(mdat, textvariable=self._selection_status).pack(anchor="w", pady=(PAD // 2, 0))

        # --- Buttons
        btn_row = ttk.Frame(root, padding=(0, PAD))
        btn_row.pack(fill="x", expand=False)
        ttk.Button(btn_row, text="Run Simulation", command=self._run_single).pack(side="left")
        ttk.Button(btn_row, text="Quit", command=self.destroy).pack(side="right")

        ttk.Label(root, text="Charts appear in a separate Results window. CSVs saved to ./out/").pack(side="bottom", anchor="w")

        if self.data_settings.auto_start_stream:
            self.after(200, self._start_chain_stream)

    # ======= BATCH TAB =======
    def _build_batch_tab(self, root):
        root = ttk.Frame(root, padding=PAD)
        root.pack(fill="both", expand=True)

        # Reuse same-style blocks but with separate variables (so Single and Batch are independent)

        # --- Instrument ---
        b_inst = ttk.LabelFrame(root, text="Instrument", padding=PAD)
        b_inst.pack(fill="x", expand=False, pady=(0, PAD))

        self.b_symbol = tk.StringVar(value=self.data_settings.default_symbol or "GLD")
        self.b_option_type = tk.StringVar(value=self.data_settings.default_option_type)
        self.b_expiration = tk.StringVar(value=self.data_settings.default_expiration or "")

        self._add_labeled_entry(b_inst, "Symbol", self.b_symbol, 0, 0)
        self._add_combo(b_inst, "Option type", self.b_option_type, ["call", "put"], 0, 1)
        self._add_labeled_entry(b_inst, "Expiration (YYYY-MM-DD)", self.b_expiration, 1, 0)
        mult_frame = ttk.Frame(b_inst)
        mult_frame.grid(row=1, column=1, sticky="w", padx=(0, PAD), pady=5)
        ttk.Label(mult_frame, text="Contract multiplier").pack(side="top", anchor="w")
        ttk.Label(mult_frame, text="100 (fixed)").pack(side="top", anchor="w")

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

    # ======= Market data =======
    def _start_chain_stream(self) -> None:
        symbol = self.var_symbol.get().strip().upper() or (self.data_settings.default_symbol or "GLD")
        expiration = self.var_expiration.get().strip() or None
        option_filter = None

        params = dict(self.data_settings.params) if self.data_settings.params else {}

        self._stop_chain_stream()
        self._selected_contract = None
        self._selected_contract_key = None
        self._selection_status.set("No contract selected")
        self.chain_view.clear_selection()

        try:
            self._chain_handle = self.data_provider.stream_option_chain(
                symbol=symbol,
                expiration=expiration,
                option_type=option_filter,
                params=params,
                poll_interval=self.data_settings.poll_interval,
                on_update=self._handle_chain_update,
                on_error=self._handle_chain_error,
            )
        except Exception as exc:  # noqa: BLE001 - present provider errors to the user
            self._chain_handle = None
            messagebox.showerror("Market data", f"Failed to start option chain stream:\n{exc}")

    def _stop_chain_stream(self) -> None:
        if self._chain_handle is not None:
            self._chain_handle.stop()
            self._chain_handle = None

    def _handle_chain_update(self, df: pd.DataFrame) -> None:
        timestamp = time.strftime("%H:%M:%S")
        prepared = self._prepare_chain_dataframe(df)

        underlying = None
        if hasattr(df, "attrs"):
            underlying = df.attrs.get("underlying_price")
        if underlying is None and "underlying_price" in prepared.columns:
            series = prepared["underlying_price"].dropna()
            if not series.empty:
                underlying = float(series.iloc[0])
        if underlying is not None:
            self._latest_underlying_price = float(underlying)

        self._latest_chain = prepared

        def _update() -> None:
            self.chain_view.update_underlying(self._latest_underlying_price)
            self.chain_view.update_from_dataframe(prepared, timestamp)
            self._refresh_selected_contract()

        self.after(0, _update)

    def _handle_chain_error(self, exc: Exception) -> None:
        def _show() -> None:
            messagebox.showerror("Market data stream", str(exc))

        self.after(0, _show)
        self._stop_chain_stream()

    def _copy_chain_to_clipboard(self) -> None:
        if self._latest_chain is None or self._latest_chain.empty:
            messagebox.showinfo("Copy option chain", "No option chain data available yet.")
            return

        try:
            csv_text = self._latest_chain.to_csv(index=False)
            self.clipboard_clear()
            self.clipboard_append(csv_text)
            messagebox.showinfo("Copy option chain", "Latest option chain copied to clipboard (CSV format).")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Copy option chain", f"Failed to copy option chain:\n{exc}")

    def _handle_chain_preview(self, row: pd.Series) -> None:
        self._selection_status.set(self._format_contract_caption(row, prefix="Preview"))

    def _on_contract_activate(self, row: pd.Series) -> None:
        self._select_contract_for_simulation(row)

    def _select_contract_for_simulation(self, row: pd.Series | None = None) -> None:
        if self._latest_chain is None or self._latest_chain.empty:
            messagebox.showinfo("Simulate trade", "Stream an option chain before selecting a contract.")
            return

        if row is None:
            row = self.chain_view.get_selected_row()
            if row is None:
                messagebox.showinfo("Simulate trade", "Single-click a contract, then press Simulate Trade.")
                return

        inputs = self._extract_contract_inputs(row)
        if inputs["entry"] is None:
            messagebox.showerror(
                "Simulate trade",
                "The selected contract is missing a mark or trade price. Wait for data or choose another strike.",
            )
            return
        if inputs["spot"] is None:
            messagebox.showerror(
                "Simulate trade",
                "Unable to determine the underlying spot price from the live chain.",
            )
            return
        if inputs["dte"] is None:
            messagebox.showerror(
                "Simulate trade",
                "Days to expiration are unavailable for the selected contract.",
            )
            return

        self._apply_contract_to_forms(inputs)
        self._selected_contract = row.copy()
        self._selected_contract_key = contract_key(row)
        if self._selected_contract_key is not None:
            self.chain_view.select_by_key(self._selected_contract_key)
        self._selection_status.set(self._format_contract_caption(row, prefix="Selected"))

    def _format_contract_caption(self, row: pd.Series | None, *, prefix: str) -> str:
        if row is None:
            return "No contract selected"
        symbol = (row.get("symbol") or "").upper()
        option_type = (row.get("option_type") or "").upper()
        strike = self._safe_float(row, "strike")
        expiration = row.get("expiration") or ""
        strike_text = f"{strike:.2f}" if strike is not None else "?"
        exp_text = f" exp {expiration}" if expiration else ""
        return f"{prefix}: {symbol} {option_type} {strike_text}{exp_text}"

    def _safe_float(self, row: pd.Series, column: str) -> Optional[float]:
        if column not in row:
            return None
        value = row.get(column)
        if value is None or (isinstance(value, str) and not value):
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:  # noqa: BLE001
            pass
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return None

    def _resolve_dte(self, row: pd.Series) -> Optional[int]:
        dte_value = row.get("dte")
        if dte_value is not None:
            try:
                if not pd.isna(dte_value):
                    return max(int(round(float(dte_value))), 0)
            except Exception:  # noqa: BLE001
                pass

        expiration = row.get("expiration")
        if not expiration or pd.isna(expiration):
            return None

        quote_time = self._safe_float(row, "quote_time")
        if quote_time is not None:
            base_dt = datetime.fromtimestamp(quote_time / 1000.0, tz=timezone.utc)
        else:
            base_dt = datetime.now(timezone.utc)

        try:
            exp_date = datetime.fromisoformat(str(expiration)[:10]).date()
        except ValueError:
            return None

        return max((exp_date - base_dt.date()).days, 0)

    def _resolve_spot(self, row: pd.Series) -> Optional[float]:
        spot = self._safe_float(row, "underlying_price")
        if spot is None and self._latest_underlying_price is not None:
            spot = float(self._latest_underlying_price)
        if spot is None:
            spot = self._safe_float(row, "spot")
        return spot

    def _extract_contract_inputs(self, row: pd.Series) -> Dict[str, Any]:
        strike = self._safe_float(row, "strike")
        mark = self._safe_float(row, "mark")
        trade_price = self._safe_float(row, "trade_price")
        if trade_price is None:
            trade_price = self._safe_float(row, "last")
        entry = mark if mark is not None and mark > 0 else trade_price

        iv = self._safe_float(row, "iv")
        if iv is None or iv <= 0:
            iv_percent = self._safe_float(row, "iv_percent")
            if iv_percent is not None:
                iv = iv_percent / 100.0 if abs(iv_percent) > 1 else iv_percent

        dte = self._resolve_dte(row)
        spot = self._resolve_spot(row)

        expiration = row.get("expiration")
        if isinstance(expiration, str):
            expiration_value = expiration
        else:
            expiration_value = None

        symbol = (row.get("symbol") or self.var_symbol.get() or "").upper()
        option_type = (row.get("option_type") or self.var_option_type.get() or "call").lower()

        return {
            "symbol": symbol,
            "option_type": option_type,
            "expiration": expiration_value,
            "strike": strike,
            "mark": mark,
            "trade_price": trade_price,
            "entry": entry,
            "iv": iv,
            "dte": dte,
            "spot": spot,
        }

    def _apply_contract_to_forms(self, inputs: Dict[str, Any]) -> None:
        symbol = inputs["symbol"]
        option_type = inputs["option_type"]
        expiration = inputs.get("expiration")
        strike = inputs.get("strike")
        entry = inputs.get("entry")
        dte = inputs.get("dte")
        spot = inputs.get("spot")
        iv = inputs.get("iv")

        if symbol:
            self.var_symbol.set(symbol)
            self.b_symbol.set(symbol)
        if option_type:
            self.var_option_type.set(option_type)
            self.b_option_type.set(option_type)
        if expiration:
            self.var_expiration.set(expiration)
            self.b_expiration.set(expiration)
        if strike is not None:
            self.var_strike.set(float(strike))
        if entry is not None:
            self.var_entry.set(float(entry))
            self.b_entry.set(float(entry))
        if dte is not None:
            self.var_dte.set(int(dte))
            self.b_dte.set(int(dte))
        if spot is not None:
            self.var_spot.set(float(spot))
            self.b_spot.set(float(spot))
        if iv is not None and iv > 0:
            self.var_iv_mode.set("fixed")
            self.var_iv_fixed.set(float(iv))
            self.b_iv_mode.set("fixed")
            self.b_iv_fixed.set(float(iv))

        if strike is not None:
            strike_text = f"{float(strike):.2f}"
            current = [s.strip() for s in self.b_strikes_text.get().split(",") if s.strip()]
            if strike_text not in current:
                current.append(strike_text)
                self.b_strikes_text.set(", ".join(current))

    def _prepare_chain_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data.attrs = dict(getattr(df, "attrs", {}))
        if data.empty:
            return data

        if "option_type" in data.columns:
            data["option_type"] = data["option_type"].fillna("").astype(str).str.lower()
        if "symbol" in data.columns:
            data["symbol"] = data["symbol"].fillna("").astype(str).str.upper()
        if "strike" in data.columns:
            data["strike"] = pd.to_numeric(data["strike"], errors="coerce")
        if "mark" in data.columns:
            data["mark"] = pd.to_numeric(data["mark"], errors="coerce")

        if "trade_price" not in data.columns:
            data["trade_price"] = data.get("last", pd.NA)
        if "pl_open" not in data.columns:
            data["pl_open"] = pd.NA
        if "pl_pct" not in data.columns:
            data["pl_pct"] = pd.NA
        if "iv_percent" not in data.columns:
            if "iv" in data.columns:
                data["iv_percent"] = data["iv"] * 100.0
            else:
                data["iv_percent"] = pd.NA
        if "quote_time" not in data.columns:
            data["quote_time"] = pd.NA
        if "underlying_price" not in data.columns:
            data["underlying_price"] = pd.NA

        data["dte"] = data.apply(self._resolve_dte, axis=1)
        return data

    def _find_contract_by_key(
        self,
        key: tuple[Any, ...],
        df: pd.DataFrame | None = None,
    ) -> Optional[pd.Series]:
        data = df if df is not None else self._latest_chain
        if data is None or data.empty:
            return None

        symbol, expiration, option_type, strike = key
        mask = data["symbol"].str.upper() == symbol
        if expiration:
            mask &= data["expiration"] == expiration
        mask &= data["option_type"] == option_type
        mask &= data["strike"].sub(strike).abs() < 1e-6
        matches = data.loc[mask]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _refresh_selected_contract(self) -> None:
        if self._selected_contract_key is None:
            return
        match = self._find_contract_by_key(self._selected_contract_key)
        if match is None:
            self._selected_contract = None
            self._selected_contract_key = None
            self._selection_status.set("No contract selected")
            self.chain_view.clear_selection()
        else:
            self._selected_contract = match.copy()
            self.chain_view.select_by_key(self._selected_contract_key)
            self._selection_status.set(self._format_contract_caption(match, prefix="Selected"))

    def _find_contract(
        self,
        *,
        symbol: str,
        option_type: str,
        strike: float,
        expiration: Optional[str] = None,
    ) -> Optional[pd.Series]:
        if self._latest_chain is None or self._latest_chain.empty:
            return None

        df = self._latest_chain
        mask = (df["symbol"] == symbol.upper()) & (df["option_type"] == option_type)
        mask &= df["strike"].sub(strike).abs() < 1e-6
        if expiration:
            mask &= df["expiration"] == expiration
        matches = df.loc[mask]
        if matches.empty:
            return None
        return matches.iloc[0]

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

    @staticmethod
    def _option_code(option_type: str) -> str:
        return {"call": "C", "put": "P"}.get(option_type.lower(), option_type.upper())

    def _single_overrides(self) -> Dict[str, Any]:
        symbol = self.var_symbol.get().strip().upper() or (self.data_settings.default_symbol or "GLD")
        expiration = self.var_expiration.get().strip() or None
        option_type = self.var_option_type.get().strip().lower()
        return {
            "symbol": symbol,
            "option_type": option_type,
            "expiration": expiration,
            "annual_trading_days": self.var_annual.get(),
            "risk_free_rate": self.var_rfr.get(),
            "iv_fixed": self.var_iv_fixed.get(),
            "iv_min": self.var_iv_min.get(),
            "iv_max": self.var_iv_max.get(),
            "num_trials": self.var_trials.get(),
            "seed": self.var_seed.get(),
            "commission_per_side": self.var_comm.get(),
            "target_profit": self.var_target.get(),
            "stop_option_price": self.var_stop.get(),
            "avoid_final_days": self.var_avoid.get(),
            "mu_mode": self.var_mu_mode.get(),
            "mu_custom": self.var_mu_custom.get(),
        }

    def _batch_overrides(self) -> Dict[str, Any]:
        symbol = self.b_symbol.get().strip().upper() or (self.data_settings.default_symbol or "GLD")
        expiration = self.b_expiration.get().strip() or None
        option_type = self.b_option_type.get().strip().lower()
        return {
            "symbol": symbol,
            "option_type": option_type,
            "expiration": expiration,
            "annual_trading_days": self.b_annual.get(),
            "risk_free_rate": self.b_rfr.get(),
            "iv_fixed": self.b_iv_fixed.get(),
            "iv_min": self.b_iv_min.get(),
            "iv_max": self.b_iv_max.get(),
            "num_trials": self.b_trials.get(),
            "seed": self.b_seed.get(),
            "commission_per_side": self.b_comm.get(),
            "target_profit": self.b_target.get(),
            "stop_option_price": self.b_stop.get(),
            "avoid_final_days": self.b_avoid.get(),
            "mu_mode": self.b_mu_mode.get(),
            "mu_custom": self.b_mu_custom.get(),
        }

    def _build_config_from_contract(self, contract: pd.Series, overrides: Dict[str, Any]) -> SimConfig:
        inputs = self._extract_contract_inputs(contract)
        strike = inputs.get("strike")
        entry = inputs.get("entry")
        dte = inputs.get("dte")
        spot = inputs.get("spot")
        iv = inputs.get("iv")

        if strike is None or entry is None or dte is None or spot is None:
            raise ValueError("Contract is missing required pricing inputs from the live chain.")

        iv_value = iv if iv is not None and iv > 0 else overrides.get("iv_fixed")
        if iv_value is None or iv_value <= 0:
            raise ValueError("Implied volatility unavailable; wait for chain data or adjust overrides.")

        symbol = (overrides.get("symbol") or inputs.get("symbol") or "").upper()
        option_type = (inputs.get("option_type") or overrides.get("option_type", "call")).lower()
        expiration = inputs.get("expiration") or overrides.get("expiration")

        return SimConfig(
            symbol=symbol,
            option_type=option_type,
            expiration=expiration,
            contract_multiplier=100,
            data_provider=self.data_settings,
            spot=float(spot),
            strike=float(strike),
            dte_calendar=int(dte),
            annual_trading_days=int(overrides["annual_trading_days"]),
            risk_free_rate=float(overrides["risk_free_rate"]),
            iv_mode="fixed",
            iv_fixed=float(iv_value),
            iv_min=float(overrides.get("iv_min", iv_value)),
            iv_max=float(overrides.get("iv_max", iv_value)),
            num_trials=int(overrides["num_trials"]),
            seed=int(overrides["seed"]),
            entry_price=float(entry),
            commission_per_side=float(overrides["commission_per_side"]),
            target_profit=float(overrides["target_profit"]),
            stop_option_price=float(overrides["stop_option_price"]),
            avoid_final_days=int(overrides["avoid_final_days"]),
            mu_mode=overrides["mu_mode"],
            mu_custom=float(overrides["mu_custom"]),
        )

    # ======= Actions =======
    def _run_single(self):
        if self._selected_contract is None:
            messagebox.showinfo(
                "Run simulation",
                "Double-click a contract or use Simulate Trade to select one before running.",
            )
            return

        try:
            overrides = self._single_overrides()
            cfg = self._build_config_from_contract(self._selected_contract, overrides)
        except ValueError as exc:
            messagebox.showerror("Simulation error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Simulation error", f"Could not prepare simulation:\n{exc}")
            return

        summary, details = simulate(cfg)
        os.makedirs("out", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        code = self._option_code(cfg.option_type)
        tag = f"{cfg.symbol}_{int(cfg.strike)}{code}_{cfg.dte_calendar}DTE_tr{cfg.num_trials}_{stamp}"
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

        if self._latest_chain is None or self._latest_chain.empty:
            messagebox.showinfo("Batch simulation", "Stream an option chain before running a batch.")
            return

        overrides = self._batch_overrides()
        os.makedirs("out", exist_ok=True)

        results = []  # list of result dicts
        stamp = time.strftime("%Y%m%d_%H%M%S")

        for k in strikes:
            contract = self._find_contract(
                symbol=overrides["symbol"],
                option_type=overrides["option_type"],
                strike=k,
                expiration=overrides.get("expiration"),
            )
            if contract is None:
                messagebox.showerror(
                    "Batch simulation",
                    f"No live chain data for strike {k:.2f} {overrides['option_type']}.",
                )
                return

            try:
                cfg = self._build_config_from_contract(contract, overrides)
            except ValueError as exc:
                messagebox.showerror("Batch simulation", str(exc))
                return

            summary, details = simulate(cfg)

            code = self._option_code(cfg.option_type)
            tag = f"{cfg.symbol}_{int(k)}{code}_{cfg.dte_calendar}DTE_tr{cfg.num_trials}_{stamp}"
            summary.to_csv(os.path.join("out", f"{tag}_summary.csv"), index=False)
            details.to_csv(os.path.join("out", f"{tag}_details.csv"), index=False)

            s_copy = summary.copy()
            s_copy.insert(0, "Strike", k)
            s_copy.insert(1, "Option Type", cfg.option_type)
            results.append({
                "strike": k,
                "option_type": cfg.option_type,
                "summary": s_copy,
                "details": details,
            })

        # Save combined batch summary
        combo = pd.concat([r["summary"] for r in results], ignore_index=True)
        combo.to_csv(os.path.join("out", f"batch_summary_{stamp}.csv"), index=False)

        # Show batch results window (overlaid charts)
        self._show_results_window_batch(results, stamp)

    def destroy(self):
        self._stop_chain_stream()
        super().destroy()

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
        """Draw overlaid histograms with distinct colors + legend."""
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

        for idx, result in enumerate(results):
            k = result["strike"]
            option_type = result["option_type"]
            details = result["details"]
            color = palette[idx % len(palette)]
            ax1.hist(
                details["final_pl"],
                bins=70,
                alpha=0.35,
                label=f"{int(k)}{self._option_code(option_type)}",
                color=color
            )

        ax1.set_title("Final P&L Distribution — Overlaid by Strike")
        ax1.set_xlabel("P&L per contract ($)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        ax1.legend(title="Strike/Type")

        canvas1 = FigureCanvasTkAgg(fig1, master=container)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side="left", fill="both", expand=True, padx=(0, PAD))

        # --- Figure 2: Overlaid Exit Day hist (hits only)
        fig2 = plt.Figure(figsize=(6.8, 5.0), dpi=100)
        ax2 = fig2.add_subplot(111)

        for idx, result in enumerate(results):
            k = result["strike"]
            option_type = result["option_type"]
            details = result["details"]
            color = palette[idx % len(palette)]
            if details["hit_target"].any():
                ax2.hist(
                    details.loc[details["hit_target"], "hit_day"],
                    bins=range(1, 60),
                    alpha=0.45,
                    label=f"{int(k)}{self._option_code(option_type)}",
                    color=color
                )

        ax2.set_title("Exit Day Distribution (Hit Target) — Overlaid by Strike")
        ax2.set_xlabel("Trading day of exit")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
        ax2.legend(title="Strike/Type")

        canvas2 = FigureCanvasTkAgg(fig2, master=container)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)

        ttk.Button(win, text="Close", command=win.destroy).pack(side="bottom", pady=PAD)

def main():
    app = SimUI()
    app.mainloop()

if __name__ == "__main__":
    main()
