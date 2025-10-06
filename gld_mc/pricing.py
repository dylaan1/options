from __future__ import annotations
import numpy as np
from mpmath import erfc

def norm_cdf(x):
    """Standard normal CDF (vectorized) via erfc."""
    x = np.asarray(x, dtype=float)
    return 0.5 * erfc(-x / np.sqrt(2))

def black_scholes_call(S, K, T, r, sigma):
    """
    Vectorized Black–Scholes call price.
    S, T can be scalars or numpy arrays. If T <= 0, returns intrinsic.
    """
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
    return price
