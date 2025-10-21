from __future__ import annotations

import math
import numpy as np


def _vectorized_erfc(x: np.ndarray) -> np.ndarray:
    """Return ``erfc`` for ``x`` supporting both scalars and ndarrays."""

    if hasattr(np, "erfc"):
        return np.erfc(x)

    # ``math.erfc`` handles scalars only; ``np.vectorize`` lifts it to arrays
    # while remaining compatible with scalar inputs for callers that pass
    # floats. This branch is exercised in minimal environments such as the test
    # suite where NumPy may be stubbed out.
    return np.vectorize(math.erfc)(x)  # pragma: no cover - fallback path


def norm_cdf(x):
    """Standard normal CDF for scalars or numpy arrays."""
    scalar_input = np.isscalar(x)
    x = np.asarray(x, dtype=float)
    result = 0.5 * _vectorized_erfc(-x / math.sqrt(2))
    if scalar_input:
        return float(result)
    return result

def black_scholes_call(S, K, T, r, sigma):
    """
    Vectorized Black–Scholes call price.
    S, T can be scalars or numpy arrays. If T <= 0, returns intrinsic.
    """
    scalar_input = np.isscalar(S) and np.isscalar(T)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)

    tiny = 1e-12
    T_safe = np.maximum(T, tiny)

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)

    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)

    price = (S * Nd1) - (K * np.exp(-r * T_safe) * Nd2)
    # intrinsic if effectively expired
    price = np.where(T <= tiny, np.maximum(S - K, 0.0), price)
    if scalar_input:
        return float(price)
    return price


def black_scholes_put(S, K, T, r, sigma):
    """Vectorized Black–Scholes put price."""
    scalar_input = np.isscalar(S) and np.isscalar(T)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)

    tiny = 1e-12
    T_safe = np.maximum(T, tiny)

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)

    Nd1 = norm_cdf(-d1)
    Nd2 = norm_cdf(-d2)

    price = (K * np.exp(-r * T_safe) * Nd2) - (S * Nd1)
    price = np.where(T <= tiny, np.maximum(K - S, 0.0), price)
    if scalar_input:
        return float(price)
    return price
