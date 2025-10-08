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

