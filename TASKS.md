# Next Tasks

Below is the prioritized queue of work items to continue generalizing the simulator. Check each item off as it is completed. Use the **View task** links to jump directly to detailed notes for that task.

- [x] Task 1: Data provider abstraction ([View task](#task-1-data-provider-abstraction))
- [ ] Task 2: Simulation input plumbing ([View task](#task-2-simulation-input-plumbing))
- [ ] Task 3: Live volatility integration ([View task](#task-3-live-volatility-integration))
- [ ] Task 4: CLI/UX polish ([View task](#task-4-cliux-polish))

---

## Task 1: Data provider abstraction
- **Objective:** Introduce a pluggable data-access layer that can load spot prices and option chains from Schwab or a mock source for offline testing.
- **Key steps:**
  - Define an interface (e.g., `DataProvider`) responsible for fetching spot, option quotes, and contract metadata.
  - Implement a Schwab-backed provider that authenticates, requests the option chain, and maps results to the simulator schema.
  - Provide an in-memory/mock provider for unit tests and demos.
  - Add configuration hooks so the simulator selects a provider at runtime.
- **Status:** ✅ Framework in place with `MarketDataProvider` protocol, a refresh-aware Schwab REST client with encrypted token storage, polling stream handle, mock generator, and UI integration that streams the chain into a live table. Complete the initial OAuth handshake (see README) to populate the encrypted cache before using the Schwab backend.

[View task](#next-tasks)

---

## Task 2: Simulation input plumbing
- **Objective:** Let users select a contract from the live option chain and have the simulator forms auto-populate with that contract's strike, expiration, mark price, implied volatility, and DTE.
- **Key information needed:**
  - Preferred UI gesture for selecting a contract (row click, double-click, or a dedicated "Use in simulation" button).
  - Mapping between chain columns and simulation inputs (e.g., `markPrice` → entry, `bidPrice`/`askPrice` averages, which IV field to trust, multiplier handling).
  - Fallback behavior when the chain omits a value (use manual overrides, compute DTE from expiration, prompt the user, etc.).
  - Whether batch runs should also accept auto-populated rows or only single-run forms.
- **Implementation steps:**
  - Wire the `OptionsChainViewer` selection callback to capture the selected row's data.
  - Normalize the row into a `SimConfig`-compatible payload, applying any fallbacks.
  - Update the single-run form fields in the UI (and CLI defaults where appropriate).
  - Record the auto-populated metadata in simulation summaries for auditability.
- **Status:** 🚧 Pending clarification from the user on the interaction flow and fallback rules noted above.

[View task](#next-tasks)

---

## Task 3: Live volatility integration
- **Objective:** Replace fixed IV inputs with real-time implied volatility from the option chain, while retaining overrides for scenario testing.
- **Key steps:**
  - Extend the configuration to accept per-contract IV sourced from the data provider.
  - Add fallbacks when IV is missing, including historical or user-specified values.
  - Surface controls to bias or stress-test volatility (e.g., +/- percentage adjustments).
  - Ensure simulation reporting logs the IV source and adjustments used.

[View task](#next-tasks)

---

## Task 4: CLI/UX polish
- **Objective:** Streamline the CLI so users can select contracts interactively and review scenario outputs across multiple tickers.
- **Key steps:**
  - Add commands/subcommands to list available expirations/strikes pulled from the data provider.
  - Provide presets for common strategies (single-leg, spreads) while keeping manual overrides.
  - Improve output formatting (tables/plots) and allow exporting CSV/JSON summaries.
  - Document CLI usage examples that cover both calls and puts.

[View task](#next-tasks)
