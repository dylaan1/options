from __future__ import annotations

import math
from collections.abc import Iterable

try:  # pragma: no cover - exercised when pandas is present
    import pandas as pd
except Exception:  # pragma: no cover - fallback when pandas is unavailable
    pd = None  # type: ignore[assignment]


def exit_day_bin_edges(hit_days: Iterable[float] | pd.Series) -> list[int]:
    """Return one-day bin edges up to the maximum observed trading day.

    The helper normalizes inputs from either plain iterables or pandas Series,
    coercing values to floats and discarding non-numeric entries so both the CLI
    and Tkinter UI can render consistent histograms.
    """

    if pd is not None and isinstance(hit_days, pd.Series):
        numeric = pd.to_numeric(hit_days, errors="coerce").dropna().tolist()
    else:
        numeric = []
        for raw in hit_days:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                numeric.append(value)

    if not numeric:
        return [1, 2]

    max_day = int(max(numeric))
    return list(range(1, max_day + 2))
