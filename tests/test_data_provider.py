import math
import sys
import types
from pathlib import Path

# Provide lightweight numpy/pandas stand-ins so the parser can be imported in
# environments where the heavy dependencies are unavailable (e.g., CI sandboxes).


class _DummyRNG:
    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return loc
        return [loc for _ in range(size)]


def _default_rng(seed=None):  # noqa: D401 - simple stub
    return _DummyRNG()


numpy_stub = types.ModuleType("numpy")
numpy_stub.random = types.SimpleNamespace(default_rng=_default_rng)


def _linspace(start, stop, num=50):
    if num <= 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [float(start + step * i) for i in range(num)]


numpy_stub.linspace = _linspace
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

    def to_list(self):
        return list(self._data)


class _DataFrame:
    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self.attrs = {}
        self.iloc = _ILoc(self)

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


class _ILoc:
    def __init__(self, df: "_DataFrame") -> None:
        self._df = df

    def __getitem__(self, index):
        return self._df._rows[index]


pandas_stub = types.ModuleType("pandas")
pandas_stub.DataFrame = _DataFrame
sys.modules.setdefault("pandas", pandas_stub)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from options_mc.data_provider import SchwabDataProvider


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
