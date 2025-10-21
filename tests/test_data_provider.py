import json
import math
import sys
import types
from pathlib import Path

import pytest

try:  # pragma: no cover - exercised when requests is available
    import requests  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal test envs
    requests_stub = types.ModuleType("requests")

    class HTTPError(Exception):
        def __init__(self, message="", response=None):
            super().__init__(message)
            self.response = response

    class Response:
        def __init__(self):
            self.status_code = 200
            self.headers = {}
            self._content = b""
            self.encoding = "utf-8"

        def json(self):
            if not self._content:
                return None
            return json.loads(self.text)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise HTTPError(f"HTTP {self.status_code}", response=self)

        @property
        def text(self):
            return self._content.decode(self.encoding or "utf-8")

    class Session:
        def request(self, *args, **kwargs):  # pragma: no cover - parity stub
            raise NotImplementedError

        def post(self, *args, **kwargs):  # pragma: no cover - parity stub
            raise NotImplementedError

    requests_stub.HTTPError = HTTPError
    requests_stub.Response = Response
    requests_stub.Session = Session

    sys.modules.setdefault("requests", requests_stub)
    import requests  # type: ignore  # noqa: E402 - import after stubbing

try:  # pragma: no cover - exercised when cryptography is available
    from cryptography.fernet import Fernet  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal test envs
    crypto_stub = types.ModuleType("cryptography")
    fernet_stub = types.ModuleType("cryptography.fernet")

    class _Fernet:  # noqa: D401 - parity stub
        def __init__(self, key):  # pragma: no cover - trivial
            self._key = key

        def encrypt(self, data):
            return data

        def decrypt(self, token):
            return token

    fernet_stub.Fernet = _Fernet
    crypto_stub.fernet = fernet_stub
    sys.modules.setdefault("cryptography", crypto_stub)
    sys.modules.setdefault("cryptography.fernet", fernet_stub)
    from cryptography.fernet import Fernet  # type: ignore  # noqa: E402 - import after stubbing

# Provide lightweight numpy/pandas stand-ins when the real dependencies are
# unavailable (e.g., minimal CI sandboxes). When the packages exist we use them
# directly so tests exercising numpy/pandas behaviour can run.

try:  # pragma: no cover - exercised when numpy/pandas are available
    import numpy  # noqa: F401
    import pandas  # noqa: F401
except Exception:  # pragma: no cover - fallback for constrained environments

    class _DummyRNG:
        def normal(self, loc=0.0, scale=1.0, size=None):
            if size is None:
                return loc
            return [loc for _ in range(size)]

        def standard_normal(self, size=None):  # pragma: no cover - parity stub
            if size is None:
                return 0.0
            return [0.0 for _ in range(size)]

    def _default_rng(seed=None):  # noqa: D401 - simple stub
        return _DummyRNG()

    class _ErrState:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: D401
            return False

    class _Array:
        def __init__(self, data):
            if isinstance(data, _Array):
                self._data = list(data._data)
            elif isinstance(data, list):
                self._data = list(data)
            elif isinstance(data, tuple):
                self._data = list(data)
            else:
                self._data = [data]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, key):
            if isinstance(key, tuple):
                first, second = key
                if isinstance(first, slice):
                    indices = range(*first.indices(len(self._data)))
                    rows = [_ensure_array(self._data[idx]) for idx in indices]
                    return _stack_rows(rows, second)
                row = self._data[first]
                if isinstance(second, slice):
                    return _Array(row[second])
                if isinstance(second, _Array):
                    return _Array([val for val, mask in zip(row, second._data) if mask])
                return row[second]
            if isinstance(key, slice):
                return _Array(self._data[key])
            if isinstance(key, _Array):
                return _Array([val for val, mask in zip(self._data, key._data) if mask])
            return self._data[key]

        def __setitem__(self, key, value):
            if isinstance(key, tuple):
                first, second = key
                if isinstance(first, slice):
                    indices = range(*first.indices(len(self._data)))
                    for idx in indices:
                        self[idx, second] = value
                    return
                row = self._data[first]
                if isinstance(second, slice):
                    indices = range(*second.indices(len(row)))
                    for idx in indices:
                        row[idx] = value
                    return
                row[second] = value
                return
            if isinstance(key, slice):
                indices = range(*key.indices(len(self._data)))
                for idx in indices:
                    self._data[idx] = value
                return
            self._data[key] = value

        def _binary_op(self, other, op):
            if isinstance(other, _Array):
                return _Array([op(a, b) for a, b in zip(self._data, other._data)])
            return _Array([op(a, other) for a in self._data])

        def __add__(self, other):
            return self._binary_op(other, lambda a, b: a + b)

        def __sub__(self, other):
            return self._binary_op(other, lambda a, b: a - b)

        def __mul__(self, other):
            return self._binary_op(other, lambda a, b: a * b)

        def __truediv__(self, other):
            return self._binary_op(other, lambda a, b: a / b if b else float("nan"))

        def __gt__(self, other):
            return self._binary_op(other, lambda a, b: a > b)

        def __ge__(self, other):
            return self._binary_op(other, lambda a, b: a >= b)

        def __lt__(self, other):
            return self._binary_op(other, lambda a, b: a < b)

        def __le__(self, other):
            return self._binary_op(other, lambda a, b: a <= b)

        def mean(self):
            if not self._data:
                return 0.0
            return sum(self._data) / len(self._data)

        def to_list(self):
            return list(self._data)

    def _ensure_array(data):
        if isinstance(data, _Array):
            return data
        if isinstance(data, list):
            return _Array(data)
        if isinstance(data, tuple):
            return _Array(list(data))
        return _Array([data])

    def _value_for_dtype(value, dtype):
        if dtype is bool:
            return bool(value)
        if dtype is int:
            return int(value)
        if dtype is object:
            return value
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return value

    def _create_array(shape, fill_value=0.0, dtype=float):
        if isinstance(shape, tuple):
            outer = []
            for _ in range(shape[0]):
                outer.append([
                    _value_for_dtype(fill_value, dtype)
                    for _ in range(shape[1])
                ])
            return _Array(outer)
        return _Array([
            _value_for_dtype(fill_value, dtype)
            for _ in range(shape)
        ])

    def _array(values, dtype=float):
        if isinstance(values, _Array):
            data = list(values._data)
        elif isinstance(values, (list, tuple)):
            data = list(values)
        else:
            try:
                data = list(values)
            except TypeError:
                return _value_for_dtype(values, dtype)

        converted = []
        for element in data:
            if isinstance(element, _Array):
                converted.append([
                    _value_for_dtype(item, dtype)
                    for item in element._data
                ])
            elif isinstance(element, (list, tuple)):
                converted.append([
                    _value_for_dtype(item, dtype)
                    for item in element
                ])
            else:
                converted.append(_value_for_dtype(element, dtype))
        return _Array(converted)

    def _asarray(values, dtype=float):
        target_dtype = float if dtype is None else dtype
        try:
            result = _array(values, target_dtype)
        except Exception:  # pragma: no cover - fallback for heterogeneous data
            if dtype is None:
                target_dtype = object
                result = _array(values, target_dtype)
            else:
                raise
        if isinstance(result, _Array):
            return result
        return _value_for_dtype(result, target_dtype)

    def _apply_unary(value, func):
        if isinstance(value, _Array):
            return _Array([func(v) for v in value._data])
        if isinstance(value, (list, tuple)):
            return _Array([func(v) for v in value])
        return func(value)

    def _stack_rows(rows, selector):
        if isinstance(selector, slice):
            return _Array([row._data[selector] for row in rows])
        if isinstance(selector, _Array):
            return _Array([
                row._data[idx]
                for row in rows
                for idx, mask in enumerate(selector._data)
                if mask
            ])
        return _Array([row._data[selector] for row in rows])

    def _linspace(start, stop, num=50):
        if num <= 1:
            return [float(start)]
        step = (stop - start) / (num - 1)
        return [float(start + step * i) for i in range(num)]

    def _percentile(array, qs):
        data = sorted(_ensure_array(array)._data)
        if isinstance(qs, (int, float)):
            qs_iter = [qs]
            single = True
        else:
            qs_iter = qs
            single = False
        results = []
        for q in qs_iter:
            if not data:
                results.append(float("nan"))
                continue
            rank = (q / 100.0) * (len(data) - 1)
            lower = int(rank)
            upper = min(lower + 1, len(data) - 1)
            weight = rank - lower
            results.append(data[lower] * (1 - weight) + data[upper] * weight)
        if single:
            return results[0]
        return results

    def _median(array):
        return _percentile(array, [50])[0]

    def _mean(array):
        arr = _ensure_array(array)
        if not arr._data:
            return 0.0
        return sum(arr._data) / len(arr._data)

    def _maximum(a, b):
        arr_a = _ensure_array(a)
        if isinstance(b, _Array):
            return _Array([max(x, y) for x, y in zip(arr_a._data, b._data)])
        result = [max(x, b) for x in arr_a._data]
        if len(result) == 1:
            return result[0]
        return _Array(result)

    def _where(condition, x, y):
        if isinstance(condition, bool):
            return x if condition else y
        cond = _ensure_array(condition)
        if isinstance(x, _Array):
            x_vals = x._data
        else:
            x_vals = [x for _ in cond._data]
        if isinstance(y, _Array):
            y_vals = y._data
        else:
            y_vals = [y for _ in cond._data]
        result = [
            xv if mask else yv
            for mask, xv, yv in zip(cond._data, x_vals, y_vals)
        ]
        if len(result) == 1:
            return result[0]
        return _Array(result)

    def _sqrt(value):
        return _apply_unary(value, math.sqrt)

    def _log(value):
        return _apply_unary(value, math.log)

    def _exp(value):
        return _apply_unary(value, math.exp)

    def _erfc(value):
        return _apply_unary(value, math.erfc)

    def _vectorize(func):
        def _wrapped(values):
            return _apply_unary(values, func)

        return _wrapped

    def _any(array):
        if isinstance(array, _Series):
            data = array._data
        else:
            data = _ensure_array(array)._data
        return any(data)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.random = types.SimpleNamespace(default_rng=_default_rng)
    numpy_stub.linspace = _linspace
    numpy_stub.asarray = _asarray
    numpy_stub.full = lambda shape, fill_value, dtype=float: _create_array(shape, fill_value, dtype)
    numpy_stub.zeros = lambda shape, dtype=float: _create_array(shape, 0.0, dtype)
    numpy_stub.array = lambda values, dtype=float: _array(values, dtype)
    numpy_stub.sqrt = _sqrt
    numpy_stub.log = _log
    numpy_stub.exp = _exp
    numpy_stub.erfc = _erfc
    numpy_stub.vectorize = _vectorize
    numpy_stub.percentile = _percentile
    numpy_stub.median = _median
    numpy_stub.mean = _mean
    numpy_stub.maximum = _maximum
    numpy_stub.where = _where
    numpy_stub.any = _any
    numpy_stub.errstate = lambda **kwargs: _ErrState()
    numpy_stub.nan = float("nan")
    numpy_stub.isscalar = lambda obj: not isinstance(obj, (list, tuple, _Array, dict))
    numpy_stub.bool_ = bool
    sys.modules.setdefault("numpy", numpy_stub)

    class _Series:
        def __init__(self, data):
            self._data = list(data)

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

        def __eq__(self, other):
            return _Series(value == other for value in self._data)

        def _binary_op(self, other, op):
            if isinstance(other, _Series):
                return _Series(op(a, b) for a, b in zip(self._data, other._data))
            return _Series(op(a, other) for a in self._data)

        def __gt__(self, other):
            return self._binary_op(other, lambda a, b: a > b)

        def __ge__(self, other):
            return self._binary_op(other, lambda a, b: a >= b)

        def __lt__(self, other):
            return self._binary_op(other, lambda a, b: a < b)

        def __le__(self, other):
            return self._binary_op(other, lambda a, b: a <= b)

        def to_list(self):
            return list(self._data)

        def to_numpy(self, dtype=None):
            return _asarray(self._data, dtype=dtype)

        def mean(self):
            if not self._data:
                return 0.0
            total = sum(self._data)
            try:
                return total / len(self._data)
            except Exception:  # pragma: no cover - safeguard for non-numeric data
                return 0.0

    class _DataFrame:
        def __init__(self, rows):
            if isinstance(rows, dict):
                keys = list(rows.keys())
                first_value = next(iter(rows.values()), [])
                if isinstance(first_value, _Array):
                    length = len(first_value)
                elif isinstance(first_value, (list, tuple)):
                    length = len(first_value)
                else:
                    length = 1
                self._rows = []
                for idx in range(length):
                    row_dict = {}
                    for key in keys:
                        value = rows[key]
                        if isinstance(value, _Array):
                            row_dict[key] = value._data[idx] if idx < len(value) else None
                        elif isinstance(value, (list, tuple)):
                            row_dict[key] = value[idx]
                        else:
                            row_dict[key] = value
                    self._rows.append(row_dict)
            else:
                self._rows = [dict(row) for row in rows]
            self.attrs = {}
            self.iloc = _ILoc(self)
            self.loc = _Loc(self)

        def sort_values(self, key):
            return _DataFrame(sorted(self._rows, key=lambda row: row.get(key)))

        def reset_index(self, drop=False):
            _ = drop  # parity with pandas signature
            return _DataFrame(self._rows)

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, key):
            if isinstance(key, str):
                return _Series(row.get(key) for row in self._rows)
            if isinstance(key, _Series):
                mask = [bool(value) for value in key]
                return _DataFrame(
                    [row for row, include in zip(self._rows, mask) if include]
                )
            if isinstance(key, (list, tuple)):
                return _DataFrame(
                    [row for row, include in zip(self._rows, key) if include]
                )
            raise TypeError(f"Unsupported key type: {type(key)!r}")

        def iterrows(self):
            for idx, row in enumerate(self._rows):
                yield idx, row

    class _ILoc:
        def __init__(self, df: "_DataFrame") -> None:
            self._df = df

        def __getitem__(self, index):
            return self._df._rows[index]

    class _Loc:
        def __init__(self, df: "_DataFrame") -> None:
            self._df = df

        def __getitem__(self, key):
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError("loc expects (rows, columns)")
            row_sel, col_sel = key
            if isinstance(row_sel, _Series):
                mask = [bool(value) for value in row_sel]
            elif isinstance(row_sel, (list, tuple)):
                mask = [bool(value) for value in row_sel]
            else:
                raise TypeError("Unsupported row selector for loc")

            filtered = [
                row
                for row, include in zip(self._df._rows, mask)
                if include
            ]

            if isinstance(col_sel, str):
                return _Series(row.get(col_sel) for row in filtered)
            if isinstance(col_sel, (list, tuple)):
                return _DataFrame(
                    [
                        {key: row.get(key) for key in col_sel}
                        for row in filtered
                    ]
                )
            return _DataFrame(filtered)

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = _DataFrame
    sys.modules.setdefault("pandas", pandas_stub)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gld_mc.data_provider import SchwabDataProvider
from gld_mc.schwab import SchwabAPIConfig, SchwabAuthManager, TokenBundle, SchwabRESTClient


class _DummySchwabClient:
    def __init__(self, quote=None, chain=None):
        self._quote = quote or {}
        self._chain = chain or {}

    def get_quote(self, symbol):  # noqa: ARG002 - signature parity
        return self._quote

    def get_option_chain(self, symbol, **params):  # noqa: ARG002 - parity
        return self._chain


class _MemoryTokenStore:
    def __init__(self, path, passphrase):  # noqa: D401 - simple stub
        self._bundle = None

    def load(self):
        return self._bundle

    def save(self, bundle):
        self._bundle = bundle

    def clear(self):  # pragma: no cover - parity with real store
        self._bundle = None


def _make_response(status_code, *, method="GET", json_payload=None, text=""):
    response = requests.Response()
    response.status_code = status_code
    response.encoding = "utf-8"
    if json_payload is not None:
        response._content = json.dumps(json_payload).encode("utf-8")
        response.headers["Content-Type"] = "application/json"
    else:
        response._content = text.encode("utf-8")
        if text:
            response.headers["Content-Type"] = "text/plain"
    response.url = "https://example.test/resource"
    return response


def test_parse_option_chain_preserves_call_and_put_rows():
    raw = {
        "symbol": "XYZ",
        "callExpDateMap": {
            "2024-01-19:30": {
                "100.0": [
                    {
                        "putCall": "CALL",
                        "symbol": "XYZ_20240119C00100000",
                        "strikePrice": 100.0,
                        "bidPrice": 1.0,
                        "askPrice": 1.2,
                        "markPrice": 1.1,
                        "lastPrice": 1.05,
                        "delta": 0.55,
                        "gamma": 0.01,
                        "theta": -0.03,
                        "vega": 0.12,
                        "rho": 0.05,
                        "totalVolume": 10,
                        "openInterest": 20,
                        "volatility": 25.0,
                        "quoteTimeInLong": 1_700_000_000_000,
                        "daysToExpiration": 30,
                    }
                ]
            }
        },
        "putExpDateMap": {
            "2024-01-19:30": {
                "100.0": [
                    {
                        "putCall": "PUT",
                        "symbol": "XYZ_20240119P00100000",
                        "strikePrice": 100.0,
                        "bidPrice": 0.9,
                        "askPrice": 1.1,
                        "markPrice": 1.0,
                        "lastPrice": 1.0,
                        "delta": -0.45,
                        "gamma": 0.011,
                        "theta": -0.025,
                        "vega": 0.13,
                        "rho": -0.04,
                        "totalVolume": 15,
                        "openInterest": 25,
                        "volatility": 28.0,
                        "quoteTimeInLong": 1_700_000_500_000,
                        "daysToExpiration": 30,
                    }
                ]
            }
        },
        "underlyingPrice": 102.0,
    }

    df = SchwabDataProvider._parse_option_chain(raw)

    assert len(df) == 2
    assert set(df["option_type"]) == {"call", "put"}

    call_row = df[df["option_type"] == "call"].iloc[0]
    put_row = df[df["option_type"] == "put"].iloc[0]

    assert call_row["expiration"] == "2024-01-19"
    assert put_row["expiration"] == "2024-01-19"
    assert call_row["dte"] == 30
    assert put_row["dte"] == 30

    # strikes remain deduplicated and properly parsed
    assert math.isclose(call_row["strike"], 100.0)
    assert math.isclose(put_row["strike"], 100.0)

    # ensure option type tagging and IV parsing work for both legs
    assert math.isclose(call_row["iv_percent"], 25.0)
    assert math.isclose(put_row["iv_percent"], 28.0)

    # confirm the underlying price metadata propagates
    assert df.attrs["underlying_price"] == 102.0


def test_parse_option_chain_handles_optional_fields():
    raw = {
        "symbol": "ABC",
        "multiplier": 75,
        "underlying": {"symbol": "ABC", "mark": 123.45},
        "callExpDateMap": {
            "2024-05-17:45": {
                "120.0": [
                    {
                        "symbol": "ABC_20240517C00120000",
                        "strikePrice": 120.0,
                        "bidPrice": 5.0,
                        "askPrice": 6.0,
                        "closePrice": 5.5,
                        "totalVolume": 50,
                        "openInterest": 150,
                        "delta": 0.5,
                        "gamma": 0.01,
                        "theta": -0.02,
                        "vega": 0.12,
                        "rho": 0.05,
                        "quoteTimeInLong": 1_700_000_000_000,
                        "daysToExpiration": None,
                        "iv": 0.25,
                        "markPercentChange": 1.23,
                    }
                ]
            }
        },
        "putExpDateMap": {},
    }

    df = SchwabDataProvider._parse_option_chain(raw)
    assert len(df) == 1

    row = df.iloc[0]
    assert row["option_type"] == "call"
    assert math.isclose(row["mark"], 5.5)
    assert math.isclose(row["trade_price"], 5.5)
    assert math.isclose(row["iv"], 0.25)
    assert math.isclose(row["iv_percent"], 25.0)
    assert row["dte"] == 45
    assert row["multiplier"] == 75
    assert df.attrs["underlying_price"] == 123.45


def test_get_spot_prefers_nested_quote_fields():
    quote_payload = {
        "quote": {"lastPrice": 101.25},
        "candles": [{"close": 99.5}],
    }
    provider = SchwabDataProvider(client=_DummySchwabClient(quote=quote_payload))

    spot = provider.get_spot("ABC")
    assert math.isclose(spot, 101.25)


def test_schwab_rest_client_retries_once_on_unauthorized():
    config = SchwabAPIConfig()
    auth_calls = {"refresh": 0}

    class _AuthManager:
        def get_access_token(self):
            return "initial-token"

        def refresh_access_token(self):
            auth_calls["refresh"] += 1
            return TokenBundle(access_token="new-token", refresh_token="r", expires_at=0)

    responses = [
        _make_response(401, json_payload={"error": "expired"}),
        _make_response(200, json_payload={"data": "ok"}),
    ]

    class _Session:
        def __init__(self, queue):
            self._queue = list(queue)
            self.calls = []

        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            response = self._queue.pop(0)
            return response

    session = _Session(responses)
    client = SchwabRESTClient(config, session=session, auth_manager=_AuthManager())

    payload = client._api_request("GET", "https://example.test/resource")

    assert payload == {"data": "ok"}
    assert auth_calls["refresh"] == 1
    assert len(session.calls) == 2


def test_schwab_rest_client_raises_runtime_error_for_server_error():
    config = SchwabAPIConfig()

    class _AuthManager:
        def get_access_token(self):
            return "token"

        def refresh_access_token(self):  # pragma: no cover - should not run
            raise AssertionError("refresh should not be attempted for 5xx responses")

    session = types.SimpleNamespace(
        request=lambda method, url, **kwargs: _make_response(
            503, method=method, text="maintenance window"
        )
    )

    client = SchwabRESTClient(config, session=session, auth_manager=_AuthManager())

    with pytest.raises(RuntimeError) as excinfo:
        client._api_request("GET", "https://example.test/resource")

    message = str(excinfo.value)
    assert "Schwab API request failed (503)" in message
    assert "maintenance window" in message


def test_schwab_auth_manager_exchange_raises_on_missing_tokens(monkeypatch):
    monkeypatch.setenv("SCHWAB_TOKEN_PASSPHRASE", "passphrase")
    monkeypatch.setattr("gld_mc.schwab.EncryptedTokenStore", _MemoryTokenStore)

    config = SchwabAPIConfig()
    config.oauth.client_id = "id"
    config.oauth.client_secret = "secret"
    config.oauth.redirect_uri = "https://example.test/callback"

    session = types.SimpleNamespace(
        post=lambda url, **kwargs: _make_response(
            200, method="POST", json_payload={"access_token": "abc"}
        )
    )

    manager = SchwabAuthManager(config, session=session)

    with pytest.raises(ValueError) as excinfo:
        manager.exchange_authorization_code("dummy-code")

    assert str(excinfo.value) == "Token response missing access or refresh token"


def test_schwab_auth_manager_refresh_raises_on_missing_tokens(monkeypatch):
    monkeypatch.setenv("SCHWAB_TOKEN_PASSPHRASE", "passphrase")
    monkeypatch.setattr("gld_mc.schwab.EncryptedTokenStore", _MemoryTokenStore)

    config = SchwabAPIConfig()
    config.oauth.client_id = "id"
    config.oauth.client_secret = "secret"
    config.oauth.redirect_uri = "https://example.test/callback"

    session = types.SimpleNamespace(
        post=lambda url, **kwargs: _make_response(
            200, method="POST", json_payload={"refresh_token": "def"}
        )
    )

    manager = SchwabAuthManager(config, session=session)
    manager._tokens = TokenBundle(access_token="tok", refresh_token="ref", expires_at=0)

    with pytest.raises(ValueError) as excinfo:
        manager.refresh_access_token()

    assert str(excinfo.value) == "Token response missing access or refresh token"
