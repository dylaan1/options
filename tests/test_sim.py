import math
import sys
import types

import pytest

pytestmark = pytest.mark.mathstack

mpmath_stub = types.ModuleType("mpmath")
mpmath_stub.erfc = math.erfc
sys.modules.setdefault("mpmath", mpmath_stub)

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

_rng_factory = getattr(getattr(np, "random", None), "default_rng", None)
if _rng_factory is None:
    pytest.skip("mathstack tests require numpy.random.default_rng", allow_module_level=True)

try:
    _rng_probe = _rng_factory()
except Exception:  # pragma: no cover - guard against broken RNG backends
    pytest.skip("mathstack tests require a working numpy.random.default_rng", allow_module_level=True)

if not hasattr(_rng_probe, "standard_normal"):
    pytest.skip(
        "mathstack tests require numpy.random.default_rng with standard_normal support",
        allow_module_level=True,
    )

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


def test_summary_matches_detail_statistics():
    cfg = SimConfig(
        option_type="call",
        dte_calendar=21,
        num_trials=256,
        seed=314,
        iv_mode="fixed",
        iv_fixed=0.22,
        entry_price=4.25,
        commission_per_side=0.10,
        target_profit=3.0,
        stop_option_price=0.5,
        contract_multiplier=1,
        strike=100.0,
        spot=101.0,
    )

    summary, details = sim_module.simulate(cfg)

    summary_map = {row["Metric"]: row["Value"] for _, row in summary.iterrows()}

    hit_mask = details["hit_day"] > 0
    if np.any(hit_mask):
        expected_median = float(np.median(details.loc[hit_mask, "hit_day"].to_numpy(dtype=float)))
        assert summary_map["Median day to hit target"] == pytest.approx(expected_median)
    else:
        assert math.isnan(summary_map["Median day to hit target"])

    expected_prob = details["hit_target"].mean()
    assert summary_map["P(hit target before expiry)"] == f"{expected_prob * 100:.1f}%"

    final_pl = details["final_pl"].to_numpy(dtype=float)
    expected_mean = f"${details['final_pl'].mean():,.0f}"
    assert summary_map["Mean Final P&L"] == expected_mean

    for label, percentile in [
        ("P&L p5", 5),
        ("P&L p25", 25),
        ("P&L p50", 50),
        ("P&L p75", 75),
        ("P&L p95", 95),
    ]:
        value = float(np.percentile(final_pl, percentile))
        assert summary_map[label] == f"${value:,.0f}"

    runtime = summary.attrs.get("runtime", {})
    expected_days = max(int(round(cfg.dte_calendar * (cfg.annual_trading_days / 365.0))), 1)
    assert runtime.get("num_trials") == cfg.num_trials
    assert runtime.get("trading_days") == expected_days


@pytest.mark.parametrize(
    ("mu_mode", "mu_custom", "expected_drift"),
    [
        ("risk_neutral", 0.33, 0.02),
        ("custom", 0.42, 0.42),
    ],
)
def test_summary_reports_resolved_drift(deterministic_rng, mu_mode, mu_custom, expected_drift):
    cfg = SimConfig(
        option_type="call",
        dte_calendar=5,
        num_trials=1,
        seed=123,
        iv_mode="fixed",
        iv_fixed=0.2,
        entry_price=5.0,
        commission_per_side=0.0,
        target_profit=10.0,
        stop_option_price=0.01,
        contract_multiplier=1,
        strike=100.0,
        spot=101.0,
        mu_mode=mu_mode,
        mu_custom=mu_custom,
    )

    summary, _ = sim_module.simulate(cfg)
    summary_map = {row["Metric"]: row["Value"] for _, row in summary.iterrows()}

    assert summary_map["Drift Mode"] == mu_mode
    assert summary_map["Drift (annual)"] == pytest.approx(expected_drift)
