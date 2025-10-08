# Next Tasks

Below are the four highest-priority follow-ups. Use the **View task** links to jump directly to implementation notes for each item.

- [ ] Task 4: Finalize progression chart styling ([View task](#task-4-finalize-progression-chart-styling))
- [ ] Task 5: Implement Schwab token refresh & throttling ([View task](#task-5-implement-schwab-token-refresh--throttling))
- [ ] Task 6: Harden streaming/UI error handling ([View task](#task-6-harden-streamingui-error-handling))
- [ ] Task 7: Extend CLI outputs with new analytics ([View task](#task-7-extend-cli-outputs-with-new-analytics))

---

## Task 4: Finalize progression chart styling
- **Objective:** Finish the requested styling for the P&L progression overlay chart in the results windows.
- **Why it matters:** The current implementation uses horizontal guides and resets axis colors after styling, so the visualization still diverges from the spec.
- **Key steps:**
  - Replace horizontal guide lines with vertical semi-axis lines spaced every 10 % across the x-axis range.
  - Set the titanium light-orange axis line colors after updating axis labels so the styling persists.
  - Double-check the maroon, goldenrod, and purple color assignments and document the palette in code comments for future reference.
- **Status:** ⏳ Pending implementation.

[Back to top](#next-tasks)

---

## Task 5: Implement Schwab token refresh & throttling
- **Objective:** Wire up refresh-token handling and client-side rate limiting for the Schwab REST adapter.
- **Why it matters:** The REST scaffold currently expects pre-minted tokens and does not guard against the 120-requests-per-minute limit, so live polling may fail once access tokens expire or rate caps are hit.
- **Key steps:**
  - Persist the encrypted refresh token, add helpers to request new access tokens, and rotate them transparently before expiry.
  - Track request timestamps inside `SchwabRESTClient` and sleep/back off when approaching Schwab’s 120-request limit.
  - Surface credential or throttling errors through the UI status banner so users know when to re-authenticate.
- **Status:** ⏳ Pending implementation.

[Back to top](#next-tasks)

---

## Task 6: Harden streaming/UI error handling
- **Objective:** Improve resiliency around the live chain stream and UI interactions.
- **Why it matters:** Network hiccups or parsing errors currently bubble straight to logs; the UI should present actionable status updates and attempt automatic recovery when feasible.
- **Key steps:**
  - Add retry/backoff logic to the `QuoteStreamHandle` loop and surface errors via banner text in the chain viewer.
  - Guard UI callbacks against missing data (e.g., selected contract disappears between refreshes) with user-friendly dialogs.
  - Capture provider exceptions in telemetry/logging so failures can be triaged post-mortem.
- **Status:** ⏳ Pending implementation.

[Back to top](#next-tasks)

---

## Task 7: Extend CLI outputs with new analytics
- **Objective:** Bring the CLI’s CSV/plot outputs to parity with the UI enhancements.
- **Why it matters:** The CLI still exports the legacy column set and lacks the progression chart/exit-time visuals introduced in the UI, limiting scripted workflows.
- **Key steps:**
  - Include ticker, option type, expiration, open/close prices, dollar & percent P/L, and calendar days in CLI summaries.
  - Generate the same P&L and holding-period histograms plus the progression overlay when running in batch mode (saving to files).
  - Document the new artifacts in the README so automated users know where to find them.
- **Status:** ⏳ Pending implementation.

[Back to top](#next-tasks)

---

## Completed Work
- ✅ Task 1: Fix Schwab chain flattening.
- ✅ Task 2: Expand mock data coverage.
- ✅ Task 3: Align chain viewer metrics.
- ✅ Pluggable market-data layer with Schwab OAuth support and encrypted token storage.
- ✅ Tkinter UI overhaul with mirrored call/put matrix, symbol history, and simulation auto-population.
- ✅ Monte Carlo engine upgrades for put pricing, per-path P&L capture, and enhanced reporting outputs.
