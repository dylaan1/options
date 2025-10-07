from __future__ import annotations
import argparse
import os
from dataclasses import asdict

import pandas as pd

from .config import SimConfig
from .sim import simulate
from .plotting import plot_results

_OPTION_CODE = {"call": "C", "put": "P"}


def _option_code(option_type: str) -> str:
    return _OPTION_CODE.get(option_type.lower(), option_type.upper())

def parse_args():
    p = argparse.ArgumentParser(description="Options Monte-Carlo simulator (single contract).")
    p.add_argument("--symbol", default="GLD", help="underlying symbol")
    p.add_argument("--option-type", choices=["call", "put"], default="call", help="option contract type")
    p.add_argument("--expiration", default="", help="expiration date (YYYY-MM-DD)")
    p.add_argument("--multiplier", type=int, default=100, help="contract multiplier (shares per contract)")
    # Primary knobs
    p.add_argument("--spot", type=float, default=364.38)
    p.add_argument("--strike", type=float, default=370.0)
    p.add_argument("--dte", type=int, default=32, help="calendar days to expiry")
    p.add_argument("--entry", type=float, default=5.50, help="entry option price ($)")
    p.add_argument("--target", type=float, default=800.0, help="target profit ($)")
    p.add_argument("--stop", type=float, default=3.00, help="stop option price ($)")
    p.add_argument("--trials", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--annual-days", type=int, default=252)
    p.add_argument("--rfr", type=float, default=0.02)
    p.add_argument("--avoid-final-days", type=int, default=0)

    # Volatility
    p.add_argument("--iv-mode", choices=["fixed", "uniform"], default="uniform")
    p.add_argument("--iv-fixed", type=float, default=0.21)
    p.add_argument("--iv-min", type=float, default=0.17)
    p.add_argument("--iv-max", type=float, default=0.25)

    # Drift
    p.add_argument("--mu-mode", choices=["risk_neutral", "custom"], default="risk_neutral")
    p.add_argument("--mu-custom", type=float, default=0.10)

    # IO
    p.add_argument("--out", default="out", help="output directory")
    p.add_argument("--tag", default="", help="tag for filenames")
    p.add_argument("--batch", type=str, default="", help='comma-separated strikes, e.g. "365,370,375"')

    return p.parse_args()

def run_one(cfg: SimConfig, out_dir: str, tag: str):
    summary, details = simulate(cfg)

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"{tag}_summary.csv")
    details_path = os.path.join(out_dir, f"{tag}_details.csv")

    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)

    plot_results(details, out_dir, tag)
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {details_path}")
    print(f"[Saved] {os.path.join(out_dir, f'{tag}_pnl_hist.png')}")

    return summary

def main():
    args = parse_args()

    expiration = args.expiration.strip() or None

    base = SimConfig(
        symbol=args.symbol,
        option_type=args.option_type,
        expiration=expiration,
        contract_multiplier=args.multiplier,
        spot=args.spot,
        strike=args.strike,
        dte_calendar=args.dte,
        annual_trading_days=args.annual_days,
        risk_free_rate=args.rfr,
        iv_mode=args.iv_mode,
        iv_fixed=args.iv_fixed,
        iv_min=args.iv_min,
        iv_max=args.iv_max,
        num_trials=args.trials,
        seed=args.seed,
        entry_price=args.entry,
        target_profit=args.target,
        stop_option_price=args.stop,
        avoid_final_days=args.avoid_final_days,
        mu_mode=args.mu_mode,
        mu_custom=args.mu_custom,
    )

    out_dir = args.out
    if args.batch.strip():
        strikes = [float(s.strip()) for s in args.batch.split(",") if s.strip()]
        rows = []
        for k in strikes:
            cfg = SimConfig(**{**asdict(base), "strike": k})
            tag = args.tag or f"{cfg.symbol}_{int(cfg.strike)}{_option_code(cfg.option_type)}_{cfg.dte_calendar}DTE_{cfg.iv_mode}_tr{cfg.num_trials}"
            s = run_one(cfg, out_dir, tag)
            s["Scenario"] = tag
            rows.append(s)
        comp = pd.concat(rows, ignore_index=True)
        comp_path = os.path.join(out_dir, "batch_summary.csv")
        comp.to_csv(comp_path, index=False)
        print(f"[Saved] {comp_path}")
    else:
        tag = args.tag or f"{base.symbol}_{int(base.strike)}{_option_code(base.option_type)}_{base.dte_calendar}DTE_{base.iv_mode}_tr{base.num_trials}"
        run_one(base, out_dir, tag)

if __name__ == "__main__":
    main()
