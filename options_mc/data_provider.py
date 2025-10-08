from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np
import pandas as pd

from .config import DataProviderConfig


def _compute_dte(
    expiration: Optional[str],
    quote_time_ms: Optional[float] = None,
) -> Optional[int]:
    """Calculate calendar days to expiration when not supplied by the API."""

    if not expiration:
        return None

    try:
        exp_date = datetime.fromisoformat(expiration[:10]).date()
    except ValueError:
        return None

    if quote_time_ms:
        base_dt = datetime.fromtimestamp(quote_time_ms / 1000.0, tz=timezone.utc)
    else:
        base_dt = datetime.now(timezone.utc)

    delta = (exp_date - base_dt.date()).days
    return max(delta, 0)


class QuoteStreamHandle:
    """Handle that manages a background polling loop."""

    def __init__(
        self,
        fetch_fn: Callable[[], pd.DataFrame],
        on_update: Callable[[pd.DataFrame], None],
        on_error: Optional[Callable[[Exception], None]] = None,
        poll_interval: float = 5.0,
    ) -> None:
        self._fetch_fn = fetch_fn
        self._on_update = on_update
        self._on_error = on_error
        self._interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                df = self._fetch_fn()
                self._on_update(df)
            except Exception as exc:  # noqa: BLE001 - surface provider errors to UI
                if self._on_error is not None:
                    self._on_error(exc)
            finally:
                # Wait with cancellation support
                if self._interval <= 0:
                    break
                self._stop.wait(self._interval)


class MarketDataProvider(Protocol):
    """Common interface for spot/option chain retrieval."""

    def get_spot(self, symbol: str) -> float:
        ...

    def get_option_chain(
        self,
        symbol: str,
        *,
        expiration: Optional[str] = None,
        option_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        ...

    def stream_option_chain(
        self,
        symbol: str,
        on_update: Callable[[pd.DataFrame], None],
        *,
        expiration: Optional[str] = None,
        option_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        poll_interval: float = 5.0,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> QuoteStreamHandle:
        ...


class MockDataProvider:
    """In-memory provider that fabricates a realistic option chain."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def get_spot(self, symbol: str) -> float:
        base = 100 + (hash(symbol) % 50)
        return float(base + self._rng.normal(0, 1.5))

    def get_option_chain(
        self,
        symbol: str,
        *,
        expiration: Optional[str] = None,
        option_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        params = params or {}
        spot = params.get("spot") or self.get_spot(symbol)

        strikes = params.get("strikes")
        if not strikes:
            strikes = np.linspace(spot * 0.8, spot * 1.2, num=11)

        quote_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Build at least three expirations with corresponding DTE hints so
        # dropdowns and pagination behave like the live Schwab chain.
        base_date = datetime.now(timezone.utc).date()
        expiration_entries: list[tuple[str, int]] = []
        if params.get("expirations"):
            for entry in params["expirations"]:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    exp_str = str(entry[0])
                    try:
                        dte_hint = int(entry[1])
                    except (TypeError, ValueError):
                        dte_hint = max(_compute_dte(exp_str) or 0, 0)
                else:
                    exp_str = str(entry)
                    dte_hint = max(_compute_dte(exp_str) or 0, 0)
                expiration_entries.append((exp_str, dte_hint))
        else:
            offsets = params.get("dte_offsets") or (7, 21, 63)
            for raw_offset in offsets:
                try:
                    offset = int(raw_offset)
                except (TypeError, ValueError):
                    offset = 30
                exp_date = base_date + timedelta(days=max(offset, 0))
                expiration_entries.append((exp_date.isoformat(), max(offset, 0)))

        if not expiration_entries:
            # Fallback to a single synthetic month if params supplied unusable data.
            fallback_date = base_date + timedelta(days=30)
            expiration_entries.append((fallback_date.isoformat(), 30))

        rows = []
        for exp_date, dte_hint in expiration_entries:
            for strike in strikes:
                skew = 0.15 + 0.05 * (strike / spot - 1.0)
                iv_call = abs(skew) + 0.18 + self._rng.normal(0, 0.01)
                iv_put = abs(skew) + 0.19 + self._rng.normal(0, 0.01)

                # Price scaffolding for call
                call_intrinsic = max(spot - strike, 0.0)
                call_bid = max(0.05, call_intrinsic + self._rng.normal(0, 0.1))
                call_ask = max(call_bid + 0.05, call_bid + abs(self._rng.normal(0, 0.1)))
                call_mid = (call_bid + call_ask) / 2
                call_last = call_mid + self._rng.normal(0, 0.05)

                # Price scaffolding for put
                put_intrinsic = max(strike - spot, 0.0)
                put_bid = max(0.05, put_intrinsic + self._rng.normal(0, 0.1))
                put_ask = max(put_bid + 0.05, put_bid + abs(self._rng.normal(0, 0.1)))
                put_mid = (put_bid + put_ask) / 2
                put_last = put_mid + self._rng.normal(0, 0.05)

                strike_val = round(float(strike), 2)

                rows.append(
                    {
                        "symbol": symbol.upper(),
                        "contract_symbol": f"{symbol.upper()}{int(round(strike_val * 100)):05d}C",
                        "expiration": exp_date,
                        "dte": dte_hint,
                        "option_type": "call",
                        "strike": strike_val,
                        "bid": round(float(call_bid), 2),
                        "ask": round(float(call_ask), 2),
                        "mark": round(float(call_mid), 2),
                        "trade_price": round(float(call_last), 2),
                        "last": round(float(call_last), 2),
                        "iv": round(float(iv_call), 4),
                        "iv_percent": round(float(iv_call * 100.0), 2),
                        "delta": round(float(self._rng.normal(0.55, 0.06)), 4),
                        "gamma": round(float(self._rng.normal(0.011, 0.002)), 5),
                        "theta": round(float(self._rng.normal(-0.025, 0.01)), 4),
                        "vega": round(float(self._rng.normal(0.13, 0.02)), 4),
                        "rho": round(float(self._rng.normal(0.05, 0.01)), 4),
                        "volume": int(abs(self._rng.normal(260, 120))),
                        "open_interest": int(abs(self._rng.normal(1350, 320))),
                        "pl_open": round(float(self._rng.normal(0, 0.35)), 2),
                        "pl_pct": round(float(self._rng.normal(0, 1.5)), 2),
                        "underlying_price": round(float(spot), 2),
                        "multiplier": 100,
                        "quote_time": quote_time_ms,
                    }
                )

                rows.append(
                    {
                        "symbol": symbol.upper(),
                        "contract_symbol": f"{symbol.upper()}{int(round(strike_val * 100)):05d}P",
                        "expiration": exp_date,
                        "dte": dte_hint,
                        "option_type": "put",
                        "strike": strike_val,
                        "bid": round(float(put_bid), 2),
                        "ask": round(float(put_ask), 2),
                        "mark": round(float(put_mid), 2),
                        "trade_price": round(float(put_last), 2),
                        "last": round(float(put_last), 2),
                        "iv": round(float(iv_put), 4),
                        "iv_percent": round(float(iv_put * 100.0), 2),
                        "delta": round(float(self._rng.normal(-0.45, 0.06)), 4),
                        "gamma": round(float(self._rng.normal(0.011, 0.002)), 5),
                        "theta": round(float(self._rng.normal(-0.02, 0.01)), 4),
                        "vega": round(float(self._rng.normal(0.11, 0.02)), 4),
                        "rho": round(float(self._rng.normal(-0.04, 0.01)), 4),
                        "volume": int(abs(self._rng.normal(240, 120))),
                        "open_interest": int(abs(self._rng.normal(1250, 320))),
                        "pl_open": round(float(self._rng.normal(0, 0.35)), 2),
                        "pl_pct": round(float(self._rng.normal(0, 1.5)), 2),
                        "underlying_price": round(float(spot), 2),
                        "multiplier": 100,
                        "quote_time": quote_time_ms,
                    }
                )

        df = pd.DataFrame(rows)
        if expiration:
            df = df[df["expiration"] == expiration]
        if option_type:
            df = df[df["option_type"] == option_type.lower()]

        df = df.sort_values("strike").reset_index(drop=True)
        df.attrs["underlying_price"] = float(spot)
        return df

    def stream_option_chain(
        self,
        symbol: str,
        on_update: Callable[[pd.DataFrame], None],
        *,
        expiration: Optional[str] = None,
        option_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        poll_interval: float = 5.0,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> QuoteStreamHandle:
        def fetch() -> pd.DataFrame:
            return self.get_option_chain(
                symbol,
                expiration=expiration,
                option_type=option_type,
                params=params,
            )

        handle = QuoteStreamHandle(fetch, on_update, on_error, poll_interval)
        handle.start()
        return handle


class SchwabClient(Protocol):
    """Minimal protocol expected from a Schwab API client."""

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        ...

    def get_option_chain(self, symbol: str, **params: Any) -> Dict[str, Any]:
        ...


@dataclass
class SchwabDataProvider:
    """Adapter that converts Schwab API responses into simulator-friendly frames."""

    client: SchwabClient

    def get_spot(self, symbol: str) -> float:
        quote = self.client.get_quote(symbol)
        # Schwab market data responses may be nested under symbol keys or provide
        # historical candles; probe a few known layouts before giving up.
        candidates = []
        if isinstance(quote, dict):
            for key in (
                "mark",
                "markPrice",
                "last",
                "lastPrice",
                "lastTrade",
                "close",
                "previousClose",
                "regularMarketLastPrice",
            ):
                value = quote.get(key)
                if value is not None:
                    candidates.append(value)

            if "quote" in quote and isinstance(quote["quote"], dict):
                nested = quote["quote"]
                for key in ("mark", "lastPrice", "close", "regularMarketLastPrice"):
                    value = nested.get(key)
                    if value is not None:
                        candidates.append(value)

            if "candles" in quote and quote["candles"]:
                last_candle = quote["candles"][-1]
                for key in ("close", "last", "lastPrice"):
                    value = last_candle.get(key)
                    if value is not None:
                        candidates.append(value)

        if candidates:
            return float(candidates[0])

        raise KeyError(f"Quote response missing price fields for {symbol}: {quote}")

    def get_option_chain(
        self,
        symbol: str,
        *,
        expiration: Optional[str] = None,
        option_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        params = params.copy() if params else {}
        if expiration:
            params.setdefault("expirationDate", expiration)
        if option_type:
            params.setdefault("contractType", option_type.upper())

        raw = self.client.get_option_chain(symbol, **params)
        return self._parse_option_chain(raw)

    def stream_option_chain(
        self,
        symbol: str,
        on_update: Callable[[pd.DataFrame], None],
        *,
        expiration: Optional[str] = None,
        option_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        poll_interval: float = 5.0,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> QuoteStreamHandle:
        def fetch() -> pd.DataFrame:
            return self.get_option_chain(
                symbol,
                expiration=expiration,
                option_type=option_type,
                params=params,
            )

        handle = QuoteStreamHandle(fetch, on_update, on_error, poll_interval)
        handle.start()
        return handle

    @staticmethod
    def _parse_option_chain(raw: Dict[str, Any]) -> pd.DataFrame:
        """Flatten Schwab's nested option chain payload into a DataFrame."""

        if not raw:
            return pd.DataFrame(columns=[
                "symbol",
                "expiration",
                "dte",
                "option_type",
                "strike",
                "bid",
                "ask",
                "mark",
                "last",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
                "volume",
                "open_interest",
            ])

        rows = []
        underlying_symbol = (raw.get("symbol") or "").upper()

        def process_leg(leg_key: str, default_option_type: str) -> None:
            leg_map = raw.get(leg_key, {}) or {}
            for expiration, strikes in leg_map.items():
                # Schwab expiration keys look like "2024-12-20:30" (date:DTE)
                exp_parts = expiration.split(":")
                exp_date = exp_parts[0]
                dte_hint = (
                    int(exp_parts[1])
                    if len(exp_parts) > 1 and exp_parts[1].isdigit()
                    else None
                )

                for strike_key, contracts in (strikes or {}).items():
                    strike_hint: Optional[float] = None
                    try:
                        strike_hint = float(strike_key)
                    except (TypeError, ValueError):
                        strike_hint = None

                    for contract in contracts or []:
                        option_type = (contract.get("putCall") or default_option_type).lower()
                        quote_time = contract.get("quoteTimeInLong") or contract.get("quoteTime")
                        resolved_dte = contract.get("daysToExpiration") or dte_hint
                        if resolved_dte in (None, ""):
                            resolved_dte = _compute_dte(
                                contract.get("expirationDate") or exp_date,
                                quote_time,
                            )

                        mark = contract.get("markPrice", contract.get("mark"))
                        if mark in (None, "") and contract.get("bidPrice") is not None and contract.get("askPrice") is not None:
                            mark = (contract["bidPrice"] + contract["askPrice"]) / 2

                        trade_price = contract.get("lastPrice", contract.get("last"))
                        if trade_price in (None, ""):
                            trade_price = contract.get("closePrice")

                        raw_iv = contract.get("volatility")
                        iv_decimal = None
                        iv_percent = None
                        if raw_iv not in (None, ""):
                            iv_percent = float(raw_iv)
                            iv_decimal = iv_percent / 100.0
                        elif contract.get("iv") not in (None, ""):
                            iv_decimal = float(contract.get("iv"))
                            iv_percent = iv_decimal * 100.0

                        rows.append(
                            {
                                "symbol": (
                                    underlying_symbol
                                    or (raw.get("underlying", {}) or {}).get("symbol", "")
                                ).upper(),
                                "contract_symbol": contract.get("symbol"),
                                "expiration": exp_date,
                                "dte": resolved_dte,
                                "option_type": option_type,
                                "strike": float(
                                    contract.get("strikePrice")
                                    or strike_hint
                                    or 0.0
                                ),
                                "bid": contract.get("bidPrice", contract.get("bid", 0.0)),
                                "ask": contract.get("askPrice", contract.get("ask", 0.0)),
                                "mark": mark,
                                "trade_price": trade_price,
                                "last": trade_price,
                                "iv": iv_decimal,
                                "iv_percent": iv_percent,
                                "delta": contract.get("delta", 0.0),
                                "gamma": contract.get("gamma", 0.0),
                                "theta": contract.get("theta", 0.0),
                                "vega": contract.get("vega", 0.0),
                                "rho": contract.get("rho", 0.0),
                                "volume": contract.get("totalVolume", 0),
                                "open_interest": contract.get("openInterest", 0),
                                "multiplier": contract.get("multiplier", raw.get("multiplier", 100)),
                                "underlying_price": raw.get("underlyingPrice")
                                or contract.get("underlyingPrice")
                                or (raw.get("underlying", {}) or {}).get("mark"),
                                "quote_time": quote_time,
                                "pl_open": contract.get("netChange"),
                                "pl_pct": contract.get("markPercentChange")
                                or contract.get("percentChange"),
                            }
                        )

        process_leg("callExpDateMap", "call")
        process_leg("putExpDateMap", "put")

        df = pd.DataFrame(rows)
        if "underlyingPrice" in raw:
            df.attrs["underlying_price"] = raw.get("underlyingPrice")
        elif "underlying" in raw and isinstance(raw["underlying"], dict):
            underlying_mark = raw["underlying"].get("mark") or raw["underlying"].get("last")
            if underlying_mark is not None:
                df.attrs["underlying_price"] = underlying_mark
        return df


def create_data_provider(
    config: DataProviderConfig,
    *,
    schwab_client: Optional[SchwabClient] = None,
) -> MarketDataProvider:
    backend = config.backend.lower()
    if backend == "mock":
        seed = None
        if config.params:
            seed = config.params.get("seed")
        return MockDataProvider(seed=seed)
    if backend == "schwab":
        if schwab_client is None:
            if config.schwab is None:
                raise ValueError(
                    "Schwab configuration required when backend='schwab' and no client provided"
                )
            try:
                from .schwab import SchwabRESTClient
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ImportError(
                    "Schwab backend requires the optional 'requests' and 'cryptography' packages"
                ) from exc

            schwab_client = SchwabRESTClient(config.schwab)
        return SchwabDataProvider(client=schwab_client)

    raise ValueError(f"Unknown data provider backend: {config.backend}")

