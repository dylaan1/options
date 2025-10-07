# Next Tasks

Below is the prioritized queue of work items to continue generalizing the simulator. Check each item off as it is completed. Use the **View task** links to jump directly to detailed notes for that task.

- [ ] Task 1: Data provider abstraction ([View task](#task-1-data-provider-abstraction))
- [ ] Task 2: Live volatility integration ([View task](#task-2-live-volatility-integration))
- [ ] Task 3: CLI/UX polish ([View task](#task-3-cliux-polish))

---

## Task 1: Data provider abstraction
- **Objective:** Introduce a pluggable data-access layer that can load spot prices and option chains from Schwab or a mock source for offline testing.
- **Key steps:**
  - Define an interface (e.g., `DataProvider`) responsible for fetching spot, option quotes, and contract metadata.
  - Implement a Schwab-backed provider that authenticates, requests the option chain, and maps results to the simulator schema.
  - Provide an in-memory/mock provider for unit tests and demos.
  - Add configuration hooks so the simulator selects a provider at runtime.

[View task](#next-tasks)

---

## Task 2: Live volatility integration
- **Objective:** Replace fixed IV inputs with real-time implied volatility from the option chain, while retaining overrides for scenario testing.
- **Key steps:**
  - Extend the configuration to accept per-contract IV sourced from the data provider.
  - Add fallbacks when IV is missing, including historical or user-specified values.
  - Surface controls to bias or stress-test volatility (e.g., +/- percentage adjustments).
  - Ensure simulation reporting logs the IV source and adjustments used.

[View task](#next-tasks)

---

## Task 3: CLI/UX polish
- **Objective:** Streamline the CLI so users can select contracts interactively and review scenario outputs across multiple tickers.
- **Key steps:**
  - Add commands/subcommands to list available expirations/strikes pulled from the data provider.
  - Provide presets for common strategies (single-leg, spreads) while keeping manual overrides.
  - Improve output formatting (tables/plots) and allow exporting CSV/JSON summaries.
  - Document CLI usage examples that cover both calls and puts.

[View task](#next-tasks)
