"""Smoke tests for submission contract and prediction invariants."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict import load_model, _require_predictable  # noqa: E402
from schema import (  # noqa: E402
    CHANNELS,
    CHANNEL_FILE_PATTERNS,
    HORIZONS_DAYS,
    PREDICTION_COLUMNS,
    UNIFIED_COLUMNS,
)
from verify_submission import check_predictions, check_repo  # noqa: E402


def test_repo_contract() -> None:
    errors = check_repo(ROOT)
    assert not errors, errors


def test_channel_patterns_cover_three_channels() -> None:
    assert set(CHANNEL_FILE_PATTERNS) == set(CHANNELS)
    for ch in CHANNELS:
        assert CHANNEL_FILE_PATTERNS[ch]


def test_unified_schema_has_core_fields() -> None:
    for col in ("date", "channel", "spend", "revenue", "campaign_id"):
        assert col in UNIFIED_COLUMNS


def test_model_pickle_loads() -> None:
    path = ROOT / "pickle" / "model.pkl"
    model = _require_predictable(load_model(path), path)
    assert hasattr(model, "predict")


def test_missing_pickle_fails_loud(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pkl"
    with pytest.raises(FileNotFoundError):
        load_model(missing)


def test_empty_pickle_fails_loud(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        load_model(empty)


def test_corrupt_pickle_fails_loud(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not-a-pickle")
    with pytest.raises(ValueError):
        load_model(bad)


def test_unsupported_artifact_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "dict.pkl"
    with path.open("wb") as fh:
        pickle.dump({"hello": "world"}, fh)
    model = load_model(path)
    with pytest.raises(TypeError):
        _require_predictable(model, path)


def test_predictions_contract_if_present() -> None:
    path = ROOT / "output" / "predictions.csv"
    if not path.exists():
        pytest.skip("output/predictions.csv not generated yet")
    errors = check_predictions(path)
    assert not errors, errors
    df = pd.read_csv(path)
    assert list(df.columns)[: len(PREDICTION_COLUMNS)] or set(PREDICTION_COLUMNS).issubset(df.columns)
    assert set(HORIZONS_DAYS).issubset(set(int(x) for x in df["horizon_days"].unique()))
    assert (df["p10_revenue"] <= df["p50_revenue"] + 1e-6).all()
    assert (df["p50_revenue"] <= df["p90_revenue"] + 1e-6).all()
