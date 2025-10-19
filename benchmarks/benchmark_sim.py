from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import types
from dataclasses import replace
from pathlib import Path
from textwrap import indent


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - defensive path injection
    sys.path.insert(0, str(ROOT))

mpmath_stub = types.ModuleType("mpmath")
mpmath_stub.erfc = math.erfc
sys.modules.setdefault("mpmath", mpmath_stub)

try:  # pragma: no cover - optional dependency bootstrap
    import numpy  # noqa: F401
    import pandas  # noqa: F401
except Exception:  # pragma: no cover - fallback stub loader
    stub_path = ROOT / "tests" / "test_data_provider.py"
    spec = importlib.util.spec_from_file_location("benchmarks._stub_loader", stub_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("benchmarks._stub_loader", module)
        spec.loader.exec_module(module)

import numpy as np  # type: ignore

if hasattr(np.random, "default_rng"):
    _probe_rng = np.random.default_rng()

    if not hasattr(_probe_rng, "uniform"):

        class _BenchmarkRNG:
            def __init__(self, seed=None):
                self._random = random.Random(seed)

            def uniform(self, low, high, size=None):
                if size is None:
                    return self._random.uniform(low, high)
                return _make_array(size, lambda: self._random.uniform(low, high))

            def standard_normal(self, size=None):
                if size is None:
                    return self._random.gauss(0.0, 1.0)
                return _make_array(size, lambda: self._random.gauss(0.0, 1.0))

        def _make_array(shape, generator):
            def _next():
                return generator() if callable(generator) else generator

            if isinstance(shape, tuple):
                if len(shape) == 2:
                    rows, cols = shape
                    data = [[_next() for _ in range(cols)] for _ in range(rows)]
                else:
                    data = [_next() for _ in range(shape[0])]
            else:
                data = [_next() for _ in range(shape)]
            return np.array(data, dtype=float)

        np.random.default_rng = lambda seed=None: _BenchmarkRNG(seed)

if not hasattr(np, "asarray"):

    def _asarray(values, dtype=float):
        is_scalar = getattr(np, "isscalar", lambda obj: not isinstance(obj, (list, tuple)))(values)
        if is_scalar:
            try:
                return dtype(values)
            except Exception:  # noqa: BLE001
                return values
        return np.array(values, dtype=dtype)

    np.asarray = _asarray

for _func in ("log", "sqrt", "exp"):
    if not hasattr(np, _func):
        setattr(np, _func, getattr(math, _func))

from gld_mc.config import SimConfig
from gld_mc import sim as sim_module
from gld_mc.sim import simulate

if not hasattr(np, "__version__"):

    def _norm_cdf_scalar(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    def _bs_call_scalar(S, K, T, r, sigma):
        tiny = 1e-12
        if T <= tiny:
            return max(S - K, 0.0)
        T_safe = max(T, tiny)
        sqrt_T = math.sqrt(T_safe)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_safe) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        price = (S * _norm_cdf_scalar(d1)) - (K * math.exp(-r * T_safe) * _norm_cdf_scalar(d2))
        return price

    def _bs_put_scalar(S, K, T, r, sigma):
        tiny = 1e-12
        if T <= tiny:
            return max(K - S, 0.0)
        T_safe = max(T, tiny)
        sqrt_T = math.sqrt(T_safe)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T_safe) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        price = (K * math.exp(-r * T_safe) * _norm_cdf_scalar(-d2)) - (S * _norm_cdf_scalar(-d1))
        return price

    sim_module.black_scholes_call = _bs_call_scalar
    sim_module.black_scholes_put = _bs_put_scalar


def _format_ms(value: float | None) -> str:
    if value is None or value != value:
        return "nan"
    return f"{value * 1_000:.3f}"


def run_case(label: str, cfg: SimConfig) -> tuple[dict[str, object], str | None]:
    summary, _ = simulate(cfg)
    runtime = summary.attrs.get("runtime", {})
    stats_text = runtime.get("profile_stats") if isinstance(runtime, dict) else None
    result = {
        "case": label,
        "dte": cfg.dte_calendar,
        "num_trials": cfg.num_trials,
        "vectorized": getattr(cfg, "vectorized_paths", False),
        "wall_time_s": runtime.get("total_wall_time"),
        "per_trial_ms": runtime.get("per_trial_wall_time"),
    }
    return result, stats_text if isinstance(stats_text, str) else None


def format_results(rows: list[dict[str, object]]) -> str:
    headers = ("Case", "DTE", "Trials", "Vectorized", "Wall (s)", "Per-trial (ms)")
    lines = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        wall = row.get("wall_time_s")
        per_trial = row.get("per_trial_ms")
        lines.append(
            " | ".join(
                [
                    str(row.get("case")),
                    str(row.get("dte")),
                    f"{row.get('num_trials'):,}",
                    "yes" if row.get("vectorized") else "no",
                    f"{wall:.4f}" if isinstance(wall, float) else "nan",
                    _format_ms(per_trial if isinstance(per_trial, float) else None),
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark gld_mc.sim.simulate")
    parser.add_argument(
        "--trials",
        type=int,
        nargs="*",
        default=[2_000, 20_000, 50_000],
        help="Trial counts to benchmark",
    )
    parser.add_argument(
        "--short-dte",
        type=int,
        default=7,
        help="Short-dated DTE to evaluate",
    )
    parser.add_argument(
        "--long-dte",
        type=int,
        default=45,
        help="Long-dated DTE to evaluate",
    )
    parser.add_argument(
        "--disable-vectorized",
        action="store_true",
        help="Run benchmarks without vectorized random draws",
    )
    parser.add_argument(
        "--show-profile",
        action="store_true",
        help="Print top profile entries for each scenario",
    )
    args = parser.parse_args()

    base_cfg = SimConfig(profile=True)
    rows: list[dict[str, object]] = []

    for dte in (args.short_dte, args.long_dte):
        for trials in args.trials:
            cfg = replace(
                base_cfg,
                dte_calendar=dte,
                num_trials=trials,
                vectorized_paths=not args.disable_vectorized,
                profile_output=None,
            )
            label = f"dte{dte}_trials{trials}"
            row, stats = run_case(label, cfg)
            rows.append(row)

            if args.show_profile and stats:
                print(f"\n=== Profile: {label} ===")
                print(indent(stats.strip(), "    "))

    print("\n" + format_results(rows))


if __name__ == "__main__":
    main()
