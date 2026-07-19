"""
LightGBM quantile revenue forecasting bundle (S3).

Trains separate P10 / P50 / P90 models on hierarchical period targets and
exposes a picklable `.predict(features_df)` used by run.sh / predict.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from schema import (
    FEATURE_VALUE_COLUMNS,
    MAX_INTERVAL_HALF_WIDTH,
    MIN_P10_ROAS_WHEN_HEALTHY,
    PREDICTION_COLUMNS,
    QUANTILES,
)


def _enforce_quantile_order(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.vstack([p10, p50, p90])
    ordered = np.sort(stacked, axis=0)
    return ordered[0], ordered[1], ordered[2]


@dataclass
class RevenueQuantileModel:
    """Picklable model artifact committed under pickle/model.pkl."""

    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_VALUE_COLUMNS))
    quantiles: tuple[float, ...] = QUANTILES
    models: dict[float, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    lgb_params: dict[str, Any] = field(default_factory=dict)
    # Relative conformal half-width: |y-p50|/max(p50,1) quantile on holdout.
    conformal_rel: float = 0.0
    conformal_radius: float = 0.0
    lower_width_scale: float = 1.0
    upper_width_scale: float = 1.0
    log_target: bool = True
    version: str = "s3-lgbm-quantile-v1"

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Features missing columns required by model: {missing}")
        x = frame[self.feature_columns].copy()
        for col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        return x.to_numpy(dtype=float)

    def fit(
        self,
        train: pd.DataFrame,
        *,
        valid: pd.DataFrame | None = None,
        target_col: str = "target_revenue",
        sample_weight: np.ndarray | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> "RevenueQuantileModel":
        y_train = train[target_col].astype(float).clip(lower=0.0).to_numpy()
        if self.log_target:
            y_train = np.log1p(y_train)
        x_train = self._matrix(train)
        w_train = None
        if sample_weight is not None:
            w_train = np.asarray(sample_weight, dtype=float)
            if len(w_train) != len(train):
                raise ValueError("sample_weight length must match train rows")

        base_params: dict[str, Any] = {
            "boosting_type": "gbdt",
            "objective": "quantile",
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 48,
            "min_child_samples": 25,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "seed": random_state,
            "deterministic": True,
            "force_col_wise": True,
        }
        self.lgb_params = dict(base_params)

        x_valid = y_valid = None
        if valid is not None and len(valid):
            x_valid = self._matrix(valid)
            y_valid = valid[target_col].astype(float).clip(lower=0.0).to_numpy()
            if self.log_target:
                y_valid = np.log1p(y_valid)

        self.models = {}
        for q in self.quantiles:
            params = dict(base_params)
            params["alpha"] = q
            dtrain = lgb.Dataset(
                x_train, label=y_train, weight=w_train, feature_name=self.feature_columns
            )
            valid_sets = [dtrain]
            callbacks: list[Any] = []
            if x_valid is not None:
                dvalid = lgb.Dataset(
                    x_valid, label=y_valid, reference=dtrain, feature_name=self.feature_columns
                )
                valid_sets.append(dvalid)
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )
            self.models[q] = booster
        return self

    def _raw_predict_quantiles(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("Model has no fitted boosters")
        x = self._matrix(frame)
        preds = {q: self.models[q].predict(x) for q in self.quantiles}
        if self.log_target:
            preds = {q: np.expm1(preds[q]) for q in preds}
        p10 = np.clip(preds[0.1], a_min=0.0, a_max=None)
        p50 = np.clip(preds[0.5], a_min=0.0, a_max=None)
        p90 = np.clip(preds[0.9], a_min=0.0, a_max=None)
        return _enforce_quantile_order(p10, p50, p90)

    def calibrate_intervals(
        self,
        frame: pd.DataFrame,
        target_col: str = "target_revenue",
        target_coverage: float = 0.80,
        max_rel: float = MAX_INTERVAL_HALF_WIDTH,
    ) -> "RevenueQuantileModel":
        """
        Estimate a mild relative half-width from holdout residuals.

        Used only to *expand* LightGBM quantiles when they are too tight —
        never to replace them with a huge ±85% band (that produced ~12x ranges).
        """
        y = frame[target_col].astype(float).clip(lower=0.0).to_numpy()
        _p10, p50, _p90 = self._raw_predict_quantiles(frame)
        denom = np.maximum(p50, 1.0)
        rel = np.abs(y - p50) / denom
        self.conformal_rel = float(min(np.quantile(rel, target_coverage), max_rel))
        self.conformal_radius = 0.0
        self.lower_width_scale = 1.0
        self.upper_width_scale = 1.0
        return self

    def predict_quantiles(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p10, p50, p90 = self._raw_predict_quantiles(frame)

        # Mild conformal expand: only if model band is thinner than conformal_rel
        if self.conformal_rel > 0:
            expand = float(min(self.conformal_rel, MAX_INTERVAL_HALF_WIDTH))
            p10 = np.minimum(p10, p50 * (1.0 - expand))
            p90 = np.maximum(p90, p50 * (1.0 + expand))

        # Cap excessive width for planning usefulness (~2–3x P90/P10)
        half = MAX_INTERVAL_HALF_WIDTH
        p10 = np.maximum(p10, p50 * (1.0 - half))
        p90 = np.minimum(p90, p50 * (1.0 + half))
        p10 = np.clip(p10, a_min=0.0, a_max=None)

        return _enforce_quantile_order(p10, p50, p90)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return predictions.csv schema rows."""
        p10, p50, p90 = self.predict_quantiles(features)
        assumed = features["planned_spend"].astype(float).fillna(0.0).to_numpy()
        assumed = np.clip(assumed, a_min=0.0, a_max=None)

        def _roas(rev: np.ndarray) -> np.ndarray:
            # No artificial ROAS=100 output clamp (that was ROAS_WINSOR_CAP*2).
            return np.divide(rev, assumed, out=np.zeros_like(rev), where=assumed > 0)

        p50_roas = _roas(p50)
        # If median outlook is healthy, don't advertise P10 "losing money"
        healthy = p50_roas >= 2.0
        p10 = np.where(
            healthy & (assumed > 0),
            np.maximum(p10, MIN_P10_ROAS_WHEN_HEALTHY * assumed),
            p10,
        )
        p10, p50, p90 = _enforce_quantile_order(p10, p50, p90)
        p10_roas, p50_roas, p90_roas = _roas(p10), _roas(p50), _roas(p90)
        p10_roas = np.where(
            healthy, np.maximum(p10_roas, MIN_P10_ROAS_WHEN_HEALTHY), p10_roas
        )

        def _col(name: str) -> np.ndarray:
            if name not in features.columns:
                return np.array([""] * len(features), dtype=object)
            s = features[name]
            return s.fillna("").astype(str).replace({"nan": "", "None": ""}).to_numpy()

        rows = pd.DataFrame(
            {
                "horizon_days": features["horizon_days"].astype(int).to_numpy(),
                "level": _col("level"),
                "channel": _col("channel"),
                "campaign_type": _col("campaign_type"),
                "campaign_id": _col("campaign_id"),
                "campaign_name": _col("campaign_name"),
                "assumed_spend": np.round(assumed, 4),
                "p10_revenue": np.round(p10, 4),
                "p50_revenue": np.round(p50, 4),
                "p90_revenue": np.round(p90, 4),
                "p10_roas": np.round(p10_roas, 6),
                "p50_roas": np.round(p50_roas, 6),
                "p90_roas": np.round(p90_roas, 6),
            }
        )
        return rows[PREDICTION_COLUMNS]
