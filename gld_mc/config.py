from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DataProviderConfig:
    """Runtime settings for the market data provider."""

    backend: str = "mock"
    poll_interval: float = 5.0
    params: Dict[str, Any] = field(default_factory=dict)
    auto_start_stream: bool = True
    default_symbol: Optional[str] = "GLD"
    default_option_type: str = "call"
    default_expiration: Optional[str] = None

@dataclass
class SimConfig:
    """Simulation inputs for a single option contract scenario."""

    # Instrument metadata
    symbol: str                  = "GLD"
    option_type: str             = "call"      # "call" or "put"
    expiration: str | None       = None        # ISO date string when available
    contract_multiplier: int     = 100

    data_provider: DataProviderConfig = field(default_factory=DataProviderConfig)



    # Market & contract
    spot: float                  = 364.38
    strike: float                = 370.0
    dte_calendar: int            = 32          # calendar days to expiry
    annual_trading_days: int     = 252
    risk_free_rate: float        = 0.02

    # Volatility
    iv_mode: str                 = "uniform"   # "fixed" or "uniform"
    iv_fixed: float              = 0.21
    iv_min: float                = 0.17
    iv_max: float                = 0.25

    # Simulation
    num_trials: int              = 20_000
    seed: int                    = 7

    # Trade management
    entry_price: float           = 5.50
    commission_per_side: float   = 0.65
    target_profit: float         = 800.0       # dollars per contract
    stop_option_price: float     = 3.00        # option mark stop
    avoid_final_days: int        = 0           # 0 disables

    # Drift
    mu_mode: str                 = "risk_neutral"   # or "custom"
    mu_custom: float             = 0.10             # used if mu_mode == "custom"
