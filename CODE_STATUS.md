# Code Status Snapshot

## Current Architecture Summary

- **Configuration (`options_mc/config.py`)** – Centralizes simulation inputs, market-data settings, and Schwab API credentials.
- **Market data (`options_mc/data_provider.py`, `options_mc/schwab.py`)** – Provides a pluggable data layer with a mock generator, Schwab REST adapter, OAuth handling, and polling stream abstraction.
- **Simulation core (`options_mc/sim.py`, `options_mc/pricing.py`)** – Runs Monte Carlo paths for long calls or puts, records per-trial exit details, and computes reporting statistics.
- **User interfaces (`options_mc/ui.py`, `options_mc/cli.py`)** – Offer a Tkinter desktop UI with live chain streaming, contract auto-population, and result visualizations alongside a CLI for scripted runs.

These components currently enable streaming an option chain, selecting a contract, auto-filling simulation forms, and reviewing simulation outputs with histograms and progression charts.

## Code Health Observations

### 1. P&L progression chart styling incomplete
- **Issue** – The overlay plot in `SimUI._plot_pl_paths` renders horizontal guide lines and resets axis label colors after styling, diverging from the requested vertical semi-axis guides and titanium-colored axes.
- **Fix** – Replace the horizontal guides with vertical lines spaced by 10 % of the x-axis span, set axis colors after updating label text, and double-check the palette matches the spec.
- **Engage Task** – [Task 4: Finalize progression chart styling](TASKS.md#task-4-finalize-progression-chart-styling)

### 2. Schwab REST client lacks refresh and rate limiting
- **Issue** – `SchwabRESTClient` assumes a valid access token and does not enforce Schwab’s 120-requests-per-minute ceiling, so long-running streams may fail once the token expires or limits are exceeded.
- **Fix** – Implement encrypted refresh-token storage, automatic token rotation, and lightweight throttling inside the REST adapter.
- **Engage Task** – [Task 5: Implement Schwab token refresh & throttling](TASKS.md#task-5-implement-schwab-token-refresh--throttling)

### 3. Stream resilience and UI messaging
- **Issue** – Provider errors currently propagate without retries and the UI offers limited feedback when the stream stalls or a selection disappears mid-refresh.
- **Fix** – Add retry/backoff to `QuoteStreamHandle`, surface status messages in the chain viewer, and guard callbacks with user-facing alerts.
- **Engage Task** – [Task 6: Harden streaming/UI error handling](TASKS.md#task-6-harden-streamingui-error-handling)

### 4. CLI output parity lagging
- **Issue** – The command-line workflow still emits the legacy summary columns and lacks the richer charts introduced in the UI.
- **Fix** – Expand CLI exports to include the new analytics and generate the same visual artifacts used in the desktop app.
- **Engage Task** – [Task 7: Extend CLI outputs with new analytics](TASKS.md#task-7-extend-cli-outputs-with-new-analytics)
