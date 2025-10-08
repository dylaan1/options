import math
import sys
import types

import pytest

mpmath_stub = types.ModuleType("mpmath")
mpmath_stub.erfc = math.erfc
sys.modules.setdefault("mpmath", mpmath_stub)

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

from gld_mc.config import SimConfig
from gld_mc import sim as sim_module


class _DeterministicRNG:
    def __init__(self, sigma_value: float = 0.2) -> None:
        self._sigma_value = sigma_value

    def uniform(self, low: float, high: float, size=None):
        value = (low + high) / 2.0
        if size is None:
            return value
        return np.full(size, value, dtype=float)

    def standard_normal(self):
        return 0.0


class _PriceSeries:
    def __init__(self, series):
        self._series = list(series)
        self._idx = 0

    def __call__(self, S, K, T, r, sigma):
        if self._idx < len(self._series):
            value = self._series[self._idx]
        else:
            value = self._series[-1]
        self._idx += 1
        return float(value)


@pytest.fixture
def deterministic_rng(monkeypatch):
    def _factory(seed=None):
        return _DeterministicRNG()

    monkeypatch.setattr(sim_module.np.random, "default_rng", _factory)
    return _factory


@pytest.mark.parametrize("dte_calendar", [7, 90])
def test_time_grid_uses_full_first_step(deterministic_rng, monkeypatch, dte_calendar):
    captured = []

    def fake_price(S, K, T, r, sigma):  # noqa: ARG001
        captured.append(float(T))
        return 0.0

    monkeypatch.setattr(sim_module, "black_scholes_call", fake_price)

    cfg = SimConfig(
        option_type="call",
        dte_calendar=dte_calendar,
        num_trials=1,
        iv_mode="fixed",
        iv_fixed=0.2,
        entry_price=0.0,
        commission_per_side=0.0,
        target_profit=1e9,
        stop_option_price=-1.0,
        contract_multiplier=1,
    )

    sim_module.simulate(cfg)

    trading_days = max(int(round(cfg.dte_calendar * (cfg.annual_trading_days / 365.0))), 1)
    dt = 1.0 / cfg.annual_trading_days

    assert len(captured) == trading_days
    assert captured[0] == pytest.approx(trading_days * dt)
    if trading_days > 1:
        assert captured[-1] == pytest.approx(dt)


def _run_sim_with_prices(monkeypatch, prices, avoid_final_days, target_profit=5.0, stop_price=5.0):
    monkeypatch.setattr(sim_module, "black_scholes_call", _PriceSeries(prices))

    cfg = SimConfig(
        option_type="call",
        dte_calendar=7,
        num_trials=1,
        iv_mode="fixed",
        iv_fixed=0.2,
        entry_price=10.0,
        commission_per_side=0.0,
        target_profit=target_profit,
        stop_option_price=stop_price,
        contract_multiplier=1,
        strike=1000.0,
        spot=10.0,
        avoid_final_days=avoid_final_days,
    )

    summary, details = sim_module.simulate(cfg)
    return summary, details.iloc[0], cfg


def test_target_triggers_when_exit_allowed(deterministic_rng, monkeypatch):
    prices = [10.0, 12.0, 16.0, 16.0, 16.0]
    _, row, cfg = _run_sim_with_prices(monkeypatch, prices, avoid_final_days=0)

    assert row["exit_reason"] == "target"
    assert row["days_open"] == 3


def test_target_blocked_within_avoid_window(deterministic_rng, monkeypatch):
    prices = [10.0, 12.0, 16.0, 16.0, 16.0]
    _, row, cfg = _run_sim_with_prices(monkeypatch, prices, avoid_final_days=2)

    assert row["exit_reason"] != "target"
    expected_days = max(int(round(cfg.dte_calendar * (cfg.annual_trading_days / 365.0))), 1)
    assert row["days_open"] == expected_days


def test_stop_blocked_within_avoid_window(deterministic_rng, monkeypatch):
    prices = [10.0, 9.0, 4.0, 4.0, 4.0]
    _, row, cfg = _run_sim_with_prices(
        monkeypatch,
        prices,
        avoid_final_days=2,
        stop_price=5.0,
        target_profit=100.0,
    )

    assert row["exit_reason"].startswith("expiry")
    expected_days = max(int(round(cfg.dte_calendar * (cfg.annual_trading_days / 365.0))), 1)
    assert row["days_open"] == expected_days
