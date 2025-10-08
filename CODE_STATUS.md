# Code Status Snapshot

## Current Architecture Summary

- **Configuration (`gld_mc/config.py`)** – Centralizes simulation inputs, market-data settings, and Schwab API credentials.
- **Market data (`gld_mc/data_provider.py`, `gld_mc/schwab.py`)** – Provides a pluggable data layer with a mock generator, Schwab REST adapter, OAuth handling, and polling stream abstraction.
- **Simulation core (`gld_mc/sim.py`, `gld_mc/pricing.py`)** – Runs Monte Carlo paths for long calls or puts, records per-trial exit details, and computes reporting statistics.
- **User interfaces (`gld_mc/ui.py`, `gld_mc/cli.py`)** – Offer a Tkinter desktop UI with live chain streaming, contract auto-population, and result visualizations alongside a CLI for scripted runs.

These components currently enable streaming an option chain, selecting a contract, auto-filling simulation forms, and reviewing simulation outputs with histograms and progression charts.

## Code Health Observations

### 1. Schwab chain flattening drops call legs
- **Issue** – `SchwabDataProvider._parse_option_chain` merges `callExpDateMap` and `putExpDateMap` using a dictionary union, so any expiration present in both maps keeps only the last map processed. Calls vanish whenever puts share the same key.
- **Fix** – Iterate the call and put maps separately, preserving both legs while keeping the shared expiration metadata intact.
- **Engage Task** – [Task 1: Fix Schwab chain flattening](TASKS.md#task-1-fix-schwab-chain-flattening)

### 2. Mock provider lacks put legs and multiple expirations
- **Issue** – `MockDataProvider.get_option_chain` fabricates only one option type per call (defaulting to calls) and a single expiration, leaving the UI’s mirrored matrix sparsely populated when running offline.
- **Fix** – Generate paired call/put rows for each strike and include at least a couple of expirations so the mock feed better matches Schwab’s schema.
- **Engage Task** – [Task 2: Expand mock data coverage](TASKS.md#task-2-expand-mock-data-coverage)

### 3. Chain viewer columns don’t match requested metrics
- **Issue** – `OptionsChainViewer` currently shows `{IV %, Open Int, Volume, Gamma, Vega, Theta, Delta, Mark}` for calls and the mirrored subset for puts. The requested columns (`Mark`, `Trade Price`, `P/L Open`, `P/L %`, `Delta`, `Theta`, `Vega`, `IV %`, `Volume`, `Open Interest`, and shared DTE) are missing.
- **Fix** – Rework the metric definitions to surface the required fields, incorporate DTE alongside strikes, and ensure placeholder `-f` text appears when data is unavailable.
- **Engage Task** – [Task 3: Align chain viewer metrics](TASKS.md#task-3-align-chain-viewer-metrics)

### 4. P&L progression chart styling incomplete
- **Issue** – The overlay plot in `SimUI._plot_pl_paths` renders horizontal guide lines and resets axis label colors after styling, diverging from the requested vertical semi-axis guides and titanium-colored axes.
- **Fix** – Replace the horizontal guides with vertical lines spaced by 10 % of the x-axis span, set axis colors after updating label text, and double-check the palette matches the spec.
- **Engage Task** – [Task 4: Finalize progression chart styling](TASKS.md#task-4-finalize-progression-chart-styling)
