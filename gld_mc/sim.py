from __future__ import annotations
import math
import numpy as np
import pandas as pd

from .config import SimConfig
from .pricing import black_scholes_call, black_scholes_put

def simulate(cfg: SimConfig):
    """
    Run a single Monte-Carlo simulation for a long call or put with target/stop rules.
    Returns: (summary_df, details_df)
    """
    option_type = cfg.option_type.lower()
    if option_type not in {"call", "put"}:
        raise NotImplementedError(f"option_type='{cfg.option_type}' is not yet supported")

    if option_type == "call":
        price_fn = black_scholes_call
        intrinsic_fn = lambda spot: max(spot - cfg.strike, 0.0)
    else:
        price_fn = black_scholes_put
        intrinsic_fn = lambda spot: max(cfg.strike - spot, 0.0)

    rng = np.random.default_rng(cfg.seed)

    trading_days = max(int(round(cfg.dte_calendar * (cfg.annual_trading_days / 365.0))), 1)
    dt = 1.0 / cfg.annual_trading_days

    # Volatility per path
    if cfg.iv_mode == "fixed":
        sigmas = np.full(cfg.num_trials, cfg.iv_fixed, dtype=float)
    else:
        sigmas = rng.uniform(cfg.iv_min, cfg.iv_max, size=cfg.num_trials)

    # Drift
    mu = cfg.risk_free_rate if cfg.mu_mode == "risk_neutral" else cfg.mu_custom

    # Time remaining after each day
    Ts = np.array([(trading_days - t) * dt for t in range(1, trading_days + 1)], dtype=float)

    # Outputs
    hit_target  = np.zeros(cfg.num_trials, dtype=bool)
    hit_day     = np.full(cfg.num_trials, -1, dtype=int)
    final_pl    = np.zeros(cfg.num_trials, dtype=float)
    exit_price  = np.zeros(cfg.num_trials, dtype=float)
    exit_reason = np.array(["hold"] * cfg.num_trials, dtype=object)

    entry_cash = cfg.entry_price * cfg.contract_multiplier + cfg.commission_per_side

    for i in range(cfg.num_trials):
        sigma = sigmas[i]
        S = cfg.spot
        exited = False

        for t in range(trading_days):
            z = rng.standard_normal()
            S = S * math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)

            T_rem = Ts[t]
            option_price = price_fn(S, cfg.strike, T_rem, cfg.risk_free_rate, sigma)
            exit_cash = option_price * cfg.contract_multiplier - cfg.commission_per_side
            pnl = exit_cash - entry_cash

            days_left = trading_days - (t + 1)
            can_exit  = (cfg.avoid_final_days == 0) or (days_left > cfg.avoid_final_days)

            if can_exit and pnl >= cfg.target_profit:
                hit_target[i]  = True
                hit_day[i]     = t + 1
                final_pl[i]    = pnl
                exit_price[i]  = option_price
                exit_reason[i] = "target"
                exited = True
                break

            if can_exit and option_price <= cfg.stop_option_price:
                hit_target[i]  = False
                hit_day[i]     = t + 1
                final_pl[i]    = pnl
                exit_price[i]  = option_price
                exit_reason[i] = "stop"
                exited = True
                break

        if not exited:
            intrinsic = intrinsic_fn(S)
            exit_cash = intrinsic * cfg.contract_multiplier - cfg.commission_per_side
            final_pl[i]    = exit_cash - entry_cash
            exit_price[i]  = intrinsic
            exit_reason[i] = "expiry_ITM" if intrinsic > 0 else "expiry_OTM"

    # Summary stats
    prob_hit = hit_target.mean()
    med_day  = float(np.median(hit_day[hit_day > 0])) if np.any(hit_day > 0) else float("nan")
    p5, p25, p50, p75, p95 = np.percentile(final_pl, [5, 25, 50, 75, 95])

    option_label = cfg.option_type.lower()
    option_display = {"call": "Call", "put": "Put"}.get(option_label, cfg.option_type)
    expiration_display = cfg.expiration or "n/a"

    summary = pd.DataFrame({
        "Metric": [
            "Symbol", "Option Type", "Expiration", "Contract Multiplier", "Trials",
            "Trading Days", "IV Mode", "IV Fixed", "IV Range", "Drift Mode",
            "Drift (annual)", "Entry Price", "Strike", "Target Profit",
            "Stop (option price)", "P(hit target before expiry)", "Median day to hit target",
            "Mean Final P&L", "P&L p5", "P&L p25", "P&L p50", "P&L p75", "P&L p95"
        ],
        "Value": [
            cfg.symbol, option_display, expiration_display, cfg.contract_multiplier,
            f"{cfg.num_trials:,}", trading_days, cfg.iv_mode, cfg.iv_fixed,
            f"{cfg.iv_min}–{cfg.iv_max}", cfg.mu_mode, cfg.mu_custom,
            f"${cfg.entry_price:.2f}", cfg.strike, f"${cfg.target_profit:.0f}",
            f"${cfg.stop_option_price:.2f}", f"{prob_hit*100:.1f}%", med_day,
            f"${np.mean(final_pl):,.0f}", f"${p5:,.0f}", f"${p25:,.0f}", f"${p50:,.0f}",
            f"${p75:,.0f}", f"${p95:,.0f}"
        ]
    })

    details = pd.DataFrame({
        "hit_target": hit_target,
        "hit_day": hit_day,
        "final_pl": final_pl,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "sigma": sigmas
    })

    return summary, details
