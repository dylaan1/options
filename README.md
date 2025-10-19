# Options Monte Carlo Simulator

Interactive Monte Carlo pricing toolkit for single-leg option trades. The project now includes a pluggable
market-data layer with live Schwab integration and a Tkinter desktop UI.

## Features

- Simulate long calls or puts with configurable exits, drift, and volatility models.
- Stream a live option chain into the UI, preview contracts, and push selections directly into the
  simulation form.
- Swap between the bundled mock market feed and a real Schwab connection without modifying the core code.

## Configuring the Schwab backend

1. **Install dependencies**

   ```bash
   pip install requests cryptography
   ```

2. **Provide Schwab OAuth credentials**

   Set the following environment variables (replace with your Schwab app metadata):

   ```bash
   export SCHWAB_CLIENT_ID={YOUR_CLIENT_ID}
   export SCHWAB_CLIENT_SECRET={YOUR_SECRET_KEY}
   export SCHWAB_REDIRECT_URI={YOUR_CALLBACK_URL}
   export SCHWAB_TOKEN_PASSPHRASE={YOUR_TOKEN_PASSPHRASE}
   ```

   Alternatively, populate these values directly on `DataProviderConfig.schwab.oauth`. The passphrase is
   used to encrypt the cached token bundle on disk (default location: `~/.schwab/tokens.dat`).

3. **Run the one-time authorization flow**

   Schwab issues both an access token and a refresh token during the OAuth code exchange. The refresh token
   is required to automatically obtain new access tokens when the short-lived access token expires.

   ```bash
   python -m gld_mc.schwab
   ```

   Follow the printed instructions: open the authorization URL, complete the broker login, copy the `code`
   query parameter from the redirected URL, and paste it back into the prompt. Successful completion stores
   the encrypted refresh/ access token bundle locally.

4. **Enable the Schwab data provider**

   Configure the simulator to use the Schwab backend, for example:

   ```python
   from gld_mc.config import DataProviderConfig, SchwabAPIConfig, SimConfig

   cfg = SimConfig(
       symbol="AAPL",
       option_type="call",
       data_provider=DataProviderConfig(
           backend="schwab",
           poll_interval=10.0,
           schwab=SchwabAPIConfig(
               market_data_base="https://api.schwabapi.com/marketdata/v1",
               trader_base="https://api.schwabapi.com/trader/v1",
           ),
       ),
   )
   ```

   The Tkinter UI (`python -m gld_mc.ui`) reads the same configuration. Once the chain stream is running,
   select a row and press **Use Selected Contract** to populate the simulation inputs with the live quote.

## Common questions

- **Where do I find the refresh token?** — The refresh token is returned alongside the access token during
  the OAuth code exchange performed by `python -m gld_mc.schwab`. It is stored (encrypted) in
  `~/.schwab/tokens.dat` and refreshed automatically whenever the simulator polls the API.
- **Can I store credentials elsewhere?** — Adjust `SchwabOAuthConfig.token_cache` to change the file
  location. The passphrase environment variable controls the encryption key.
- **How often can I poll?** — The default rate limiter caps requests at 120 per minute to match Schwab’s
  published guidance. Adjust `SchwabRateLimit.max_requests_per_minute` if your entitlement differs.

## Performance benchmarking & regression guidance

Monte Carlo workloads can become CPU-bound, so the simulator now exposes lightweight profiling hooks and a
repeatable benchmark driver:

- Set `SimConfig.profile = True` to capture wall-clock timing and a `cProfile` summary for each run. The
  `profile_output` field optionally dumps the raw profiler table to disk.
- Run `python benchmarks/benchmark_sim.py` to sample representative scenarios (short/long DTE and varied
  `num_trials`). Use `--disable-vectorized` to compare against the scalar Gaussian draw loop and
  `--show-profile` to print the hottest functions.

On the reference development container a 45 DTE, 10k-trial configuration completes in roughly **2.03 seconds**
(≈0.203 ms per trial) using the optimized loop structure. Treat per-trial latencies above **0.25 ms** as
a regression warning and re-run the benchmark script to confirm.

## Testing

The optional scientific stack tests are decorated with the `mathstack` marker. They exercise the full
NumPy/Pandas/Black–Scholes pipeline and are skipped automatically if those dependencies are unavailable.

- Run only these tests with `pytest -m mathstack` after installing the scientific stack.
- Skip them explicitly (even when the dependencies are present) with `pytest -m "not mathstack"`.

