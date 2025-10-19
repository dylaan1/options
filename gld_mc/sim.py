from __future__ import annotations

import cProfile
import io
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pstats


def _supports_vectorized_numpy() -> bool:
    """Detect whether the imported numpy exposes the vectorized features we rely on."""
    try:
        sample = np.array([1.0, 2.0], dtype=float)
        idx = np.array([0, 1], dtype=int)
        _ = sample[idx]
        _ = sample * sample
        sample[idx] = sample
        np.exp(sample)
        np.full(1, 1.0, dtype=float)
        return True
    except Exception:
        return False


_CAN_VECTORIZE = _supports_vectorized_numpy()

from .config import SimConfig
from .pricing import black_scholes_call, black_scholes_put


def simulate(cfg: SimConfig):
    """
    Run a single Monte-Carlo simulation for a long call or put with target/stop rules.
    Returns: (summary_df, details_df)
    """
    do_profile = bool(getattr(cfg, "profile", False))
    profiler: cProfile.Profile | None = None

    start = perf_counter()
    if do_profile:
        profiler = cProfile.Profile()
        profiler.enable()

    summary, details, trading_days = _run_simulation(cfg)

    if profiler is not None:
        profiler.disable()

    elapsed = perf_counter() - start

    runtime: dict[str, float | int | str] = {
        "total_wall_time": elapsed,
        "per_trial_wall_time": (elapsed / cfg.num_trials) if cfg.num_trials else float("nan"),
        "num_trials": cfg.num_trials,
        "trading_days": trading_days,
    }

    if profiler is not None:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("tottime")
        stats.print_stats(25)
        profile_text = stream.getvalue()
        runtime["profile_stats"] = profile_text

        if cfg.profile_output:
            output_path = Path(cfg.profile_output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(profile_text)

    summary.attrs["runtime"] = runtime
    details.attrs["runtime"] = runtime

    return summary, details


def _run_simulation(cfg: SimConfig):
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
    sqrt_dt = math.sqrt(dt)

    # Volatility per path
    if cfg.iv_mode == "fixed":
        sigmas = np.full(cfg.num_trials, cfg.iv_fixed, dtype=float)
    else:
        sigmas = rng.uniform(cfg.iv_min, cfg.iv_max, size=cfg.num_trials)

    drift_base = cfg.risk_free_rate if cfg.mu_mode == "risk_neutral" else cfg.mu_custom

    sigma_values = [float(s) for s in sigmas] if cfg.num_trials else []
    if cfg.num_trials:
        sigmas = np.array(sigma_values, dtype=float)
        drift_terms = np.array(
            [drift_base - 0.5 * sigma * sigma for sigma in sigma_values],
            dtype=float,
        ) * dt
        vol_terms = np.array([sigma * sqrt_dt for sigma in sigma_values], dtype=float)
    else:
        sigmas = np.array([], dtype=float)
        drift_terms = np.array([], dtype=float)
        vol_terms = np.array([], dtype=float)

    # Time remaining after each day (first step keeps the full time-to-expiry)
    Ts = np.array([(trading_days - t) * dt for t in range(trading_days)], dtype=float)

    # Outputs
    hit_target = np.zeros(cfg.num_trials, dtype=bool)
    hit_day = np.full(cfg.num_trials, -1, dtype=int)
    final_pl = np.zeros(cfg.num_trials, dtype=float)
    exit_price = np.zeros(cfg.num_trials, dtype=float)
    exit_reason = np.array(["hold"] * cfg.num_trials, dtype=object)
    days_open = np.zeros(cfg.num_trials, dtype=int)
    pl_paths = np.zeros((cfg.num_trials, trading_days), dtype=float)

    contract_multiplier = cfg.contract_multiplier
    commission_per_side = cfg.commission_per_side
    entry_cash = cfg.entry_price * contract_multiplier + commission_per_side
    target_profit = cfg.target_profit
    stop_option_price = cfg.stop_option_price
    avoid_final_days = cfg.avoid_final_days
    risk_free_rate = cfg.risk_free_rate

    if avoid_final_days == 0:
        can_exit_flags = np.array([True] * trading_days, dtype=bool)
    else:
        can_exit_flags = np.array(
            [(trading_days - (idx + 1)) > avoid_final_days for idx in range(trading_days)],
            dtype=bool,
        )

    if _CAN_VECTORIZE:
        _run_vectorized_paths(
            cfg,
            price_fn,
            intrinsic_fn,
            sigmas,
            mu,
            Ts,
            rng,
            trading_days,
            dt,
            entry_cash,
            hit_target,
            hit_day,
            final_pl,
            exit_price,
            exit_reason,
            days_open,
            pl_paths,
        )
    else:
        _run_scalar_paths(
            cfg,
            price_fn,
            intrinsic_fn,
            sigmas,
            mu,
            Ts,
            rng,
            trading_days,
            dt,
            entry_cash,
            hit_target,
            hit_day,
            final_pl,
            exit_price,
            exit_reason,
            days_open,
            pl_paths,
        )

    # Summary stats
    prob_hit = hit_target.mean() if cfg.num_trials else float("nan")
    med_day = float(np.median(hit_day[hit_day > 0])) if np.any(hit_day > 0) else float("nan")
    p5, p25, p50, p75, p95 = np.percentile(final_pl, [5, 25, 50, 75, 95]) if cfg.num_trials else (float("nan"),) * 5

    option_label = cfg.option_type.lower()
    option_display = {"call": "Call", "put": "Put"}.get(option_label, cfg.option_type)
    expiration_display = cfg.expiration or "n/a"

    summary = pd.DataFrame(
        {
            "Metric": [
                "Symbol",
                "Option Type",
                "Expiration",
                "Contract Multiplier",
                "Trials",
                "Trading Days",
                "IV Mode",
                "IV Fixed",
                "IV Range",
                "Drift Mode",
                "Drift (annual)",
                "Entry Price",
                "Strike",
                "Target Profit",
                "Stop (option price)",
                "P(hit target before expiry)",
                "Median day to hit target",
                "Mean Final P&L",
                "P&L p5",
                "P&L p25",
                "P&L p50",
                "P&L p75",
                "P&L p95",
            ],
            "Value": [
                cfg.symbol,
                option_display,
                expiration_display,
                cfg.contract_multiplier,
                f"{cfg.num_trials:,}",
                trading_days,
                cfg.iv_mode,
                cfg.iv_fixed,
                f"{cfg.iv_min}–{cfg.iv_max}",
                cfg.mu_mode,
                cfg.mu_custom,
                f"${cfg.entry_price:.2f}",
                cfg.strike,
                f"${cfg.target_profit:.0f}",
                f"${cfg.stop_option_price:.2f}",
                f"{prob_hit * 100:.1f}%" if not math.isnan(prob_hit) else "nan",
                med_day,
                f"${np.mean(final_pl):,.0f}" if cfg.num_trials else "nan",
                f"${p5:,.0f}" if cfg.num_trials else "nan",
                f"${p25:,.0f}" if cfg.num_trials else "nan",
                f"${p50:,.0f}" if cfg.num_trials else "nan",
                f"${p75:,.0f}" if cfg.num_trials else "nan",
                f"${p95:,.0f}" if cfg.num_trials else "nan",
            ],
        }
    )

    calendar_days = np.maximum(days_open - 1, 0)
    entry_value = cfg.entry_price * cfg.contract_multiplier
    with np.errstate(divide="ignore", invalid="ignore"):
        pl_percent = np.where(entry_value != 0, final_pl / entry_value, np.nan)

    details = pd.DataFrame(
        {
            "hit_target": hit_target,
            "hit_day": hit_day,
            "days_open": days_open,
            "calendar_days": calendar_days,
            "final_pl": final_pl,
            "pl_percent": pl_percent,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "sigma": sigmas,
            "pl_path": [list(path) for path in pl_paths],
        }
    )

    return summary, details


def _run_vectorized_paths(
    cfg: SimConfig,
    price_fn,
    intrinsic_fn,
    sigmas,
    mu: float,
    Ts: np.ndarray,
    rng,
    trading_days: int,
    dt: float,
    entry_cash: float,
    hit_target: np.ndarray,
    hit_day: np.ndarray,
    final_pl: np.ndarray,
    exit_price: np.ndarray,
    exit_reason: np.ndarray,
    days_open: np.ndarray,
    pl_paths: np.ndarray,
) -> None:
    spots = np.full(cfg.num_trials, cfg.spot, dtype=float)
    still_open = np.ones(cfg.num_trials, dtype=bool)
    sqrt_dt = math.sqrt(dt)

    for t in range(trading_days):
        active_idx = np.flatnonzero(still_open)
        if active_idx.size == 0:
            break

        sigma_active = sigmas[active_idx]
        z = rng.standard_normal(size=active_idx.size)
        drift = (mu - 0.5 * sigma_active * sigma_active) * dt
        diffusion = sigma_active * sqrt_dt * z

        new_spots = spots[active_idx] * np.exp(drift + diffusion)
        spots[active_idx] = new_spots

        T_rem = Ts[t]
        option_price = price_fn(new_spots, cfg.strike, T_rem, cfg.risk_free_rate, sigma_active)
        exit_cash = option_price * cfg.contract_multiplier - cfg.commission_per_side
        pnl = exit_cash - entry_cash
        pl_paths[active_idx, t] = pnl

        days_left = trading_days - (t + 1)
        can_exit = (cfg.avoid_final_days == 0) or (days_left > cfg.avoid_final_days)
        if not can_exit:
            continue

        target_hits_mask = pnl >= cfg.target_profit
        if np.any(target_hits_mask):
            idx = active_idx[target_hits_mask]
            hit_target[idx] = True
            hit_day[idx] = t + 1
            final_pl[idx] = pnl[target_hits_mask]
            exit_price[idx] = option_price[target_hits_mask]
            exit_reason[idx] = np.array(["target"] * idx.size, dtype=object)
            days_open[idx] = t + 1
            pl_paths[idx, t:] = final_pl[idx][:, None]
            still_open[idx] = False

        remaining_mask = ~target_hits_mask
        if np.any(remaining_mask):
            stop_hits_mask = (option_price <= cfg.stop_option_price) & remaining_mask
            if np.any(stop_hits_mask):
                idx = active_idx[stop_hits_mask]
                hit_target[idx] = False
                hit_day[idx] = t + 1
                final_pl[idx] = pnl[stop_hits_mask]
                exit_price[idx] = option_price[stop_hits_mask]
                exit_reason[idx] = np.array(["stop"] * idx.size, dtype=object)
                days_open[idx] = t + 1
                pl_paths[idx, t:] = final_pl[idx][:, None]
                still_open[idx] = False

    open_idx = np.flatnonzero(still_open)
    if open_idx.size > 0:
        intrinsic = intrinsic_fn(spots[open_idx])
        exit_cash = intrinsic * cfg.contract_multiplier - cfg.commission_per_side
        final_values = exit_cash - entry_cash
        final_pl[open_idx] = final_values
        exit_price[open_idx] = intrinsic
        expiry_reason = np.array(["expiry_OTM"] * open_idx.size, dtype=object)
        expiry_reason[intrinsic > 0] = "expiry_ITM"
        exit_reason[open_idx] = expiry_reason
        days_open[open_idx] = trading_days
        pl_paths[open_idx, -1] = final_values


def _run_scalar_paths(
    cfg: SimConfig,
    price_fn,
    intrinsic_fn,
    sigmas,
    mu: float,
    Ts: np.ndarray,
    rng,
    trading_days: int,
    dt: float,
    entry_cash: float,
    hit_target: np.ndarray,
    hit_day: np.ndarray,
    final_pl: np.ndarray,
    exit_price: np.ndarray,
    exit_reason: np.ndarray,
    days_open: np.ndarray,
    pl_paths: np.ndarray,
) -> None:
    sqrt_dt = math.sqrt(dt)

    for i in range(cfg.num_trials):
        sigma = sigmas[i]
        S = cfg.spot
        exited = False

        for t in range(trading_days):
            z = rng.standard_normal()
            S = S * math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z)

            T_rem = Ts[t]
            option_price = price_fn(S, cfg.strike, T_rem, cfg.risk_free_rate, sigma)
            exit_cash = option_price * cfg.contract_multiplier - cfg.commission_per_side
            pnl = exit_cash - entry_cash
            pl_paths[i, t] = pnl

            days_left = trading_days - (t + 1)
            can_exit = (cfg.avoid_final_days == 0) or (days_left > cfg.avoid_final_days)

            if can_exit and pnl >= cfg.target_profit:
                hit_target[i] = True
                hit_day[i] = t + 1
                final_pl[i] = pnl
                exit_price[i] = option_price
                exit_reason[i] = "target"
                days_open[i] = t + 1
                pl_paths[i, t:] = pnl
                exited = True
                break

            if can_exit and option_price <= cfg.stop_option_price:
                hit_target[i] = False
                hit_day[i] = t + 1
                final_pl[i] = pnl
                exit_price[i] = option_price
                exit_reason[i] = "stop"
                days_open[i] = t + 1
                pl_paths[i, t:] = pnl
                exited = True
                break

        if not exited:
            intrinsic = intrinsic_fn(S)
            exit_cash = intrinsic * cfg.contract_multiplier - cfg.commission_per_side
            final_pl[i] = exit_cash - entry_cash
            exit_price[i] = intrinsic
            exit_reason[i] = "expiry_ITM" if intrinsic > 0 else "expiry_OTM"
            days_open[i] = trading_days
            pl_paths[i, -1] = final_pl[i]
        elif trading_days > days_open[i]:
            pl_paths[i, days_open[i]:] = final_pl[i]
