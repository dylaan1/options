# Options Monte Carlo Simulator

An end-to-end toolkit for exploring single-leg option trades. The project bundles a
vectorized Monte Carlo engine, a command-line workflow for exporting results, and a
Tkinter desktop interface that can stream live market data.

---

## 1. Installation

### 1.1 Prerequisites
- Python 3.10 or newer.
- macOS, Windows, or Linux (the UI uses Tkinter, which is installed with the
  default Python builds on each platform).

### 1.2 Create an isolated environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 1.3 Core simulation dependencies
Install the numeric stack that powers the simulator:
```bash
pip install -r gld_mc/Requirements
```
The list currently includes:
- `numpy` – vectorized Gaussian paths and matrix operations
- `pandas` – tabular summaries and CSV output
- `matplotlib` – histogram visualizations
- `mpmath` – Black–Scholes helper for scalar fallbacks

### 1.4 Optional Schwab API integration
To pull live option chains you will also need:
```bash
pip install requests cryptography
```
`requests` performs the HTTPS calls and `cryptography` protects the cached OAuth
refresh token. No broker account is required if you only intend to use the mock
market-data feed.

---

## 2. Running simulations from the command line
The CLI exposes every knob on `SimConfig` and writes results to disk.

```bash
python -m gld_mc.cli --symbol GLD --option-type call \
  --spot 364.38 --strike 370 --dte 32 --trials 20000 --seed 7 \
  --entry 5.50 --target 800 --stop 3.00 --iv-mode uniform \
  --iv-min 0.17 --iv-max 0.25 --out out --tag gld_run
```

### 2.1 Key arguments
| Flag | Purpose |
|------|---------|
| `--symbol`, `--option-type`, `--expiration` | Define the contract under study. |
| `--spot`, `--strike`, `--dte` | Set the underlying price, strike, and days to expiry. |
| `--trials`, `--seed` | Control Monte Carlo path count and RNG seed. |
| `--entry`, `--target`, `--stop` | Trade management rules in dollars per contract. |
| `--iv-mode`, `--iv-fixed`, `--iv-min`, `--iv-max` | Volatility sampling behaviour (fixed vs. uniform range). |
| `--mu-mode`, `--mu-custom` | Choose risk-neutral drift or supply a custom annual drift. |
| `--batch` | Comma-separated strike list. Produces one simulation per strike and aggregates the summaries. |
| `--out`, `--tag` | Output directory and file prefix. |

Run `python -m gld_mc.cli --help` to see the full option list.

### 2.2 Outputs
Each run produces:
- `<tag>_summary.csv` – one-row overview with P&L statistics and hit rates.
- `<tag>_details.csv` – full path-level log (daily marks, exit flags, reasons).
- `<tag>_pnl_hist.png` – histogram of per-path profit & loss.
Batch runs also emit `batch_summary.csv` with a comparison table across strikes.

---

## 3. Desktop interface
The Tkinter UI combines the simulator with a live option-chain viewer.

```bash
python -m gld_mc.ui
```

### 3.1 Layout highlights
- **Live Option Chain (left/right panes):** Streams bids/asks, Greeks, and
  liquidity stats for calls and puts at each strike.
- **Scenario Form (top-right):** Mirrors the CLI arguments. When a row is selected
  in the chain, click **Use Selected Contract** to populate symbol/strike/expiration.
- **Results Tabs:** Review summary tables, daily P&L curves, and histogram plots
  immediately after each simulation.

Use the gear icon to switch between the bundled mock feed and the Schwab backend
(section 4). The UI automatically starts streaming when the provider is configured
with `auto_start_stream=True` (default).

---

## 4. Market-data providers

### 4.1 Mock provider (default)
No setup required. Synthetic spot prices and option chains are generated locally.
Ideal for demonstrations or offline experimentation.

### 4.2 Schwab provider
1. **Set credentials** – supply your Schwab Developer App metadata via environment variables:
   ```bash
   export SCHWAB_CLIENT_ID="YOUR_APP_ID"
   export SCHWAB_CLIENT_SECRET="YOUR_SECRET"
   export SCHWAB_REDIRECT_URI="https://example.com/callback"
   export SCHWAB_TOKEN_PASSPHRASE="a strong passphrase"
   ```
   The passphrase encrypts the cached refresh token at `~/.schwab/tokens.dat`.
2. **Authorize once** – run the OAuth bootstrap to capture refresh tokens:
   ```bash
   python -m gld_mc.schwab
   ```
   Open the printed URL, log in, and paste the returned `code` back into the prompt.
3. **Select the backend** – pass a `DataProviderConfig` to the simulator or choose
   "Schwab" from the UI provider menu. For scripts:
   ```python
   from gld_mc.config import DataProviderConfig, SchwabAPIConfig, SimConfig

   sim_cfg = SimConfig(
       symbol="AAPL",
       option_type="call",
       data_provider=DataProviderConfig(
           backend="schwab",
           poll_interval=10.0,
           schwab=SchwabAPIConfig(),
       ),
   )
   ```
The Schwab client uses a token bucket limiter (default 120 requests/minute) to
respect published API guidance.

---

## 5. Understanding simulation inputs
`SimConfig` (see `gld_mc/config.py`) holds every field used by the engine. In
addition to the CLI flags, you can tune:
- `contract_multiplier` and `commission_per_side` – adapt for index options or
  brokers with different fee schedules.
- `avoid_final_days` – skip trading during the last *n* calendar days prior to
  expiry before evaluating stops/targets.
- `profile` / `profile_output` – enable per-run `cProfile` stats, optionally
  writing them to disk.
- `vectorized_paths` – disable vectorized math if you need to diagnose the scalar
  reference loop.

---

## 6. Benchmarking and regression testing
- **Performance sampling:** `python benchmarks/benchmark_sim.py` profiles common
  scenarios and reports per-trial wall times. Use `--show-profile` to print the
  hottest functions and `--disable-vectorized` to compare against the scalar baseline.
- **Automated tests:**
  ```bash
  pytest
  ```
  Markers:
  - `mathstack` – exercises the full scientific stack (NumPy/Pandas/Matplotlib).
    Run with `pytest -m mathstack` after installing those dependencies or skip
    them via `pytest -m "not mathstack"`.

---

## 7. Directory guide
```
gld_mc/
  cli.py          # CLI entry point for batch & single simulations
  ui.py           # Tkinter front-end with live option-chain viewer
  sim.py          # Monte Carlo engine and runtime profiling hooks
  pricing.py      # Black–Scholes helpers for calls/puts
  analytics.py    # Aggregations used by the UI and tests
  data_provider.py# Mock feed, Schwab client, and streaming abstractions
  plotting.py     # Matplotlib rendering utilities
  config.py       # Dataclasses that describe simulation & provider settings
benchmarks/
  benchmark_sim.py# Repeatable performance harness
tests/            # Unit tests for analytics, providers, and simulations
```

---

## 8. Next steps
- Integrate with custom data providers by subclassing the `MarketDataProvider`
  protocol defined in `gld_mc/data_provider.py`.
- Extend the CLI to iterate over expirations or volatility regimes by scripting
  around `SimConfig`.
- Plug the summary CSVs into your analytics pipeline (e.g., load in pandas or Excel).

Happy simulating!
