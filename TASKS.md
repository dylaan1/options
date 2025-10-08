# Next Tasks

Below are the four highest-priority follow-ups. Use the **View task** links to jump directly to implementation notes for each item.

- [ ] Task 1: Fix Schwab chain flattening ([View task](#task-1-fix-schwab-chain-flattening))
- [ ] Task 2: Expand mock data coverage ([View task](#task-2-expand-mock-data-coverage))
- [ ] Task 3: Align chain viewer metrics ([View task](#task-3-align-chain-viewer-metrics))
- [ ] Task 4: Finalize progression chart styling ([View task](#task-4-finalize-progression-chart-styling))

---

## Task 1: Fix Schwab chain flattening
- **Objective:** Ensure call and put legs share the same expiration key without overwriting each other when flattening Schwab option-chain payloads.
- **Why it matters:** The UI’s mirrored matrix currently loses either the call or put side for expirations returned in both maps, leaving half of the chain blank even with live data.
- **Key steps:**
  - Iterate the `callExpDateMap` and `putExpDateMap` separately, tagging each row with its option type.
  - Preserve the original expiration/DTE hints while deduplicating strike values.
  - Add a regression-friendly unit test (or doctest) that feeds a minimal Schwab payload with both legs and verifies both appear in the resulting DataFrame.
- **Status:** ⏳ Pending implementation.

[Back to top](#next-tasks)

---

## Task 2: Expand mock data coverage
- **Objective:** Update the mock provider so offline runs mimic Schwab’s structure with paired call/put rows across multiple expirations.
- **Why it matters:** The UI should remain fully functional during development demos without live credentials; today the mock chain only populates a single option type, leaving the matrix asymmetric.
- **Key steps:**
  - Generate both call and put legs for every strike produced by the mock RNG.
  - Include at least two expirations and corresponding DTE hints so the expiration dropdown and matrix pagination can be exercised.
  - Ensure placeholder fields (`trade_price`, `pl_open`, `pl_pct`, etc.) are populated so formatting stays consistent with live data.
- **Status:** ⏳ Pending implementation.

[Back to top](#next-tasks)

---

## Task 3: Align chain viewer metrics
- **Objective:** Bring the displayed columns in `OptionsChainViewer` in line with the requested layout (Mark ↔ IV %, Trade Price, P/L Open, P/L %, Delta, Theta, Vega, Volume, Open Interest, and shared DTE alongside strikes).
- **Why it matters:** The current column set omits several broker-style metrics, reducing the usefulness of the live chain for decision making.
- **Key steps:**
  - Redefine the call/put metric lists so they mirror each other around the strike column while including the requested data points.
  - Surface DTE in the central strike panel (e.g., stacked label or combined text) while keeping the strike anchored.
  - Verify the formatting helper continues to render `-f` when data is missing and adjust column widths if necessary.
- **Status:** ⏳ Pending implementation.

[Back to top](#next-tasks)

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

## Completed Work
- ✅ Pluggable market-data layer with Schwab OAuth support and encrypted token storage.
- ✅ Tkinter UI overhaul with mirrored call/put matrix, symbol history, and simulation auto-population.
- ✅ Monte Carlo engine upgrades for put pricing, per-path P&L capture, and enhanced reporting outputs.
