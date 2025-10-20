import importlib

import pytest


def _has_mathstack() -> bool:
    """Return True when the scientific stack is importable and usable."""
    try:
        numpy = importlib.import_module("numpy")
        pandas = importlib.import_module("pandas")
    except ModuleNotFoundError:
        return False

    if not hasattr(pandas, "DataFrame"):
        return False

    default_rng = getattr(getattr(numpy, "random", None), "default_rng", None)
    if default_rng is None:
        return False

    try:
        candidate = default_rng()
    except Exception:  # pragma: no cover - defensive guard for broken RNGs
        return False

    if not hasattr(candidate, "standard_normal"):
        return False

    return True


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "mathstack: requires numpy and pandas; run with `pytest -m mathstack` to include.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _has_mathstack():
        return

    skip_marker = pytest.mark.skip(reason="mathstack tests require numpy and pandas")
    for item in items:
        if "mathstack" in item.keywords:
            item.add_marker(skip_marker)
