"""
LightGBM quantile revenue forecasting with split-conformal intervals.

Train three quantile models (α = 0.1 / 0.5 / 0.9) on the train window only,
then calibrate additive conformal adjustments on a held-out calibration window
(per channel). Test data is never used here — only in evaluate.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from schema import (
    CHANNELS,
    CONFORMAL_HOLIDAY_MAX_RATIO,
    CONFORMAL_MIN_STRATUM,
    CONFORMAL_SCORE_TRIM_Q,
    CONFORMAL_TARGET_COVERAGE,
    FEATURE_VALUE_COLUMNS,
    LOW_SPEND_ROAS_CAP,
    LOW_SPEND_THRESHOLD,
    MIN_P10_ROAS_GLOBAL,
    MIN_P10_ROAS_WHEN_HEALTHY,
    PREDICTION_COLUMNS,
    QUANTILES,
    SPLIT_CALIB_FRAC,
    SPLIT_TEST_FRAC,
    SPLIT_TRAIN_FRAC,
)


def _enforce_quantile_order(
    p10: np.ndarray, p50: np.ndarray, p90: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.vstack([p10, p50, p90])
    ordered = np.sort(stacked, axis=0)
    return ordered[0], ordered[1], ordered[2]


def chronological_three_way_split(
    frame: pd.DataFrame,
    *,
    train_frac: float = SPLIT_TRAIN_FRAC,
    calib_frac: float = SPLIT_CALIB_FRAC,
    test_frac: float = SPLIT_TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique as_of_date cutoffs: first 60% train, next 20% calibration,
    final 20% test. Never random.
    """
    if abs(train_frac + calib_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + calib_frac + test_frac must sum to 1")
    frame = frame.sort_values("as_of_date").reset_index(drop=True)
    cutoffs = np.sort(pd.to_datetime(frame["as_of_date"]).unique())
    n = len(cutoffs)
    if n < 3:
        # Degenerate: put what we can into train / calib / test by rows.
        n_rows = len(frame)
        i1 = max(1, int(n_rows * train_frac))
        i2 = max(i1 + 1, int(n_rows * (train_frac + calib_frac)))
        return (
            frame.iloc[:i1].copy(),
            frame.iloc[i1:i2].copy(),
            frame.iloc[i2:].copy(),
        )

    i_train_end = max(1, int(np.floor(n * train_frac)))
    i_calib_end = max(i_train_end + 1, int(np.floor(n * (train_frac + calib_frac))))
    i_calib_end = min(i_calib_end, n - 1)  # leave ≥1 date for test when possible

    train_end = cutoffs[i_train_end - 1]
    calib_end = cutoffs[i_calib_end - 1]

    dates = pd.to_datetime(frame["as_of_date"])
    train = frame[dates <= train_end].copy()
    calib = frame[(dates > train_end) & (dates <= calib_end)].copy()
    test = frame[dates > calib_end].copy()

    if train.empty or calib.empty or test.empty:
        # Fallback row split if date buckets collapsed.
        n_rows = len(frame)
        i1 = max(1, int(n_rows * train_frac))
        i2 = max(i1 + 1, int(n_rows * (train_frac + calib_frac)))
        return (
            frame.iloc[:i1].copy(),
            frame.iloc[i1:i2].copy(),
            frame.iloc[i2:].copy(),
        )
    return train, calib, test


def conformal_quantile_level(n: int, target_coverage: float = CONFORMAL_TARGET_COVERAGE) -> float:
    """Finite-sample split-conformal quantile level: ceil((n+1)*q) / n."""
    if n <= 0:
        return target_coverage
    level = float(np.ceil((n + 1) * target_coverage) / n)
    return float(min(level, 1.0))


def nonconformity_scores(y: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> np.ndarray:
    """score = max(p10 - y, y - p90); >0 means y outside [p10, p90]."""
    y = np.asarray(y, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    return np.maximum(p10 - y, y - p90)


def season_bucket_mask(frame: pd.DataFrame) -> np.ndarray:
    """
    Holiday vs non-holiday season bucket from ``as_of_date``.

    Nov–Dec (peak ecommerce / Q4) vs all other months. Matches ``is_holiday_season``
    in feature generation when that column is present.
    """
    if "is_holiday_season" in frame.columns:
        return frame["is_holiday_season"].astype(int).fillna(0).to_numpy().astype(bool)
    if "as_of_date" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    months = pd.to_datetime(frame["as_of_date"]).dt.month.to_numpy()
    return np.isin(months, (11, 12))


def _season_label(holiday: bool) -> str:
    return "holiday" if holiday else "non_holiday"


def google_brand_bucket(frame: pd.DataFrame) -> np.ndarray:
    """
    Mondrian brand bucket for Google rows: ``brand``, ``non_brand``, or ``""``.

    Trademark / brand Search (``Search_TM_*``, campaign types containing brand)
    vs non-trademark / everything else. Empty string = unknown (channel/aggregate
    rows without a campaign_type) — those skip the brand stratum.
    """
    n = len(frame)
    if n == 0:
        return np.array([], dtype=object)
    if "channel" not in frame.columns:
        return np.array([""] * n, dtype=object)

    ch = frame["channel"].astype(str).str.lower().to_numpy()
    ctype = (
        frame["campaign_type"].fillna("").astype(str).str.lower()
        if "campaign_type" in frame.columns
        else pd.Series([""] * n)
    )
    cname = (
        frame["campaign_name"].fillna("").astype(str).str.lower()
        if "campaign_name" in frame.columns
        else pd.Series([""] * n)
    )
    text = (ctype + "|" + cname).to_numpy()

    out = np.array([""] * n, dtype=object)
    for i in range(n):
        if ch[i] != "google":
            continue
        t = text[i]
        # NTM must win over a naive "_tm_" substring match inside "_ntm_".
        is_ntm = ("_ntm_" in t) or ("ntm_" in t) or ("non_brand" in t) or ("nonbrand" in t)
        is_brand = (not is_ntm) and (
            ("_tm_" in t)
            or ("search_tm" in t)
            or ("brand" in ctype.iloc[i])
            or t.startswith("brand|")
            or ("|brand" in t)
        )
        # Only assign brand/non_brand when we have campaign identity.
        has_identity = bool(ctype.iloc[i].strip()) or bool(cname.iloc[i].strip())
        if not has_identity:
            out[i] = ""
        elif is_brand:
            out[i] = "brand"
        else:
            out[i] = "non_brand"
    return out


@dataclass
class RevenueQuantileModel:
    """Picklable model: three LightGBM quantile boosters + per-channel conformal."""

    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_VALUE_COLUMNS))
    quantiles: tuple[float, ...] = QUANTILES
    models: dict[float, Any] = field(default_factory=dict)
    train_metrics: dict[str, Any] = field(default_factory=dict)
    lgb_params: dict[str, Any] = field(default_factory=dict)
    # Mondrian conformal adjustments (revenue units).
    # Keys: global, {channel}, {channel}_holiday, {channel}_non_holiday,
    # plus google_brand / google_non_brand (+ season) for Google.
    conformal_adjustments: dict[str, float] = field(default_factory=dict)
    conformal_stratum_counts: dict[str, int] = field(default_factory=dict)
    # Legacy aliases kept so older pickles still unpickle; unused by new path.
    conformal_rel: float = 0.0
    conformal_radius: float = 0.0
    log_target: bool = True
    version: str = "s5-mondrian-conformal-v2"
    target_coverage: float = CONFORMAL_TARGET_COVERAGE
    score_trim_q: float = CONFORMAL_SCORE_TRIM_Q
    holiday_max_ratio: float = CONFORMAL_HOLIDAY_MAX_RATIO

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
        """Fit P10 / P50 / P90 LightGBM quantile models on ``train`` only."""
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

    def _raw_predict_quantiles(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw LightGBM quantiles — no conformal adjustment."""
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

    def _scores_on_frame(
        self,
        frame: pd.DataFrame,
        target_col: str = "target_revenue",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (y, p10, p90, scores) from raw quantiles on ``frame``."""
        y = frame[target_col].astype(float).clip(lower=0.0).to_numpy()
        p10, _p50, p90 = self._raw_predict_quantiles(frame)
        scores = nonconformity_scores(y, p10, p90)
        return y, p10, p90, scores

    def _adjustment_from_scores(
        self,
        scores: np.ndarray,
        cov: float,
        *,
        clip_cap: float | None = None,
    ) -> float:
        """
        Trimmed split-conformal adjustment.

        Clip nonconformity scores at the ``score_trim_q`` percentile (default 95th)
        before taking the finite-sample conformal quantile. Optional ``clip_cap``
        (e.g. the non-holiday stratum's 95th) further limits the clip so a single
        outlier holiday period cannot set an inflated local 95th and blow up the band.
        """
        n = int(len(scores))
        if n == 0:
            return 0.0
        scores = np.asarray(scores, dtype=float)
        trim_q = float(self.score_trim_q)
        if n >= 2 and 0.0 < trim_q < 1.0:
            local_cap = float(np.quantile(scores, trim_q))
            cap = local_cap if clip_cap is None else float(min(local_cap, clip_cap))
            scores = np.minimum(scores, cap)
        elif clip_cap is not None:
            scores = np.minimum(scores, float(clip_cap))
        level = conformal_quantile_level(n, cov)
        return float(np.quantile(scores, level))

    @staticmethod
    def _score_trim_cap(scores: np.ndarray, trim_q: float) -> float | None:
        scores = np.asarray(scores, dtype=float)
        if len(scores) < 2 or not (0.0 < trim_q < 1.0):
            return None
        return float(np.quantile(scores, trim_q))

    def calibrate_conformal(
        self,
        calibration: pd.DataFrame,
        target_col: str = "target_revenue",
        target_coverage: float | None = None,
        train: pd.DataFrame | None = None,
        min_stratum: int = CONFORMAL_MIN_STRATUM,
    ) -> "RevenueQuantileModel":
        """
        Mondrian split-conformal calibration on the calibration window.

        Strata: channel × season (holiday Nov–Dec vs non-holiday), plus for
        Google a brand vs non-brand split (TM Search vs everything else), with
        fallbacks brand×season → brand → channel×season → channel → global.

        Score definition:

            score_i = max(p10_i - y_i, y_i - p90_i)
            scores clipped at the 95th percentile (outlier guard)
            q_level = ceil((n+1) * target_coverage) / n
            adjustment = quantile(trimmed_scores, q_level)

        When a holiday stratum has fewer than ``min_stratum`` calibration rows,
        scores are augmented from **train** rows in the same (channel, season
        [, brand]) bucket (model frozen — scores only, not retraining).

        Adjustments are never grid-searched to hit a coverage target.
        """
        if calibration is None or calibration.empty:
            raise ValueError("calibration frame is empty")

        cov = float(target_coverage if target_coverage is not None else self.target_coverage)
        _y, _p10, _p90, cal_scores = self._scores_on_frame(calibration, target_col)
        cal_hol = season_bucket_mask(calibration)
        cal_ch = (
            calibration["channel"].astype(str).str.lower().to_numpy()
            if "channel" in calibration.columns
            else np.array([""] * len(calibration))
        )
        cal_brand = google_brand_bucket(calibration)

        train_scores = train_hol = train_ch = train_brand = None
        if train is not None and not train.empty:
            _yt, _p10t, _p90t, train_scores = self._scores_on_frame(train, target_col)
            train_hol = season_bucket_mask(train)
            train_ch = (
                train["channel"].astype(str).str.lower().to_numpy()
                if "channel" in train.columns
                else np.array([""] * len(train))
            )
            train_brand = google_brand_bucket(train)

        adjustments: dict[str, float] = {}
        counts: dict[str, int] = {}

        def _pool_scores(
            channel: str | None,
            holiday: bool | None,
            brand: str | None = None,
        ) -> np.ndarray:
            mask = np.ones(len(calibration), dtype=bool)
            if channel is not None:
                mask &= cal_ch == channel
            if holiday is not None:
                mask &= cal_hol == holiday
            if brand is not None:
                mask &= cal_brand == brand
            pooled = cal_scores[mask]
            if len(pooled) >= min_stratum or train_scores is None:
                return pooled
            # Sparse holiday (or other) stratum: add train scores for same bucket.
            tmask = np.ones(len(train), dtype=bool)
            if channel is not None:
                tmask &= train_ch == channel
            if holiday is not None:
                tmask &= train_hol == holiday
            if brand is not None:
                tmask &= train_brand == brand
            if tmask.any():
                pooled = np.concatenate([pooled, train_scores[tmask]])
            return pooled

        # Global + per-channel + Mondrian (channel × season)
        adjustments["global"] = self._adjustment_from_scores(cal_scores, cov)
        counts["global"] = int(len(cal_scores))

        for name in CHANNELS:
            ch_scores = _pool_scores(name, None)
            adjustments[name] = self._adjustment_from_scores(ch_scores, cov)
            counts[name] = int(len(ch_scores))
            # Non-holiday first so its 95th can stabilize the holiday clip.
            nh_scores = _pool_scores(name, False)
            counts[f"{name}_non_holiday"] = int(len(nh_scores))
            if len(nh_scores) >= 5:
                adjustments[f"{name}_non_holiday"] = self._adjustment_from_scores(
                    nh_scores, cov
                )
            else:
                adjustments[f"{name}_non_holiday"] = adjustments[name]
            nh_cap = self._score_trim_cap(nh_scores, self.score_trim_q)

            hol_scores = _pool_scores(name, True)
            counts[f"{name}_holiday"] = int(len(hol_scores))
            if len(hol_scores) >= 5:
                # Clip holiday scores using non-holiday 95th so one wild holiday
                # period cannot inflate the local trim cap (~53x blow-up guard).
                hol_adj = self._adjustment_from_scores(
                    hol_scores, cov, clip_cap=nh_cap
                )
            else:
                hol_adj = adjustments[name]
            nh_adj = float(adjustments[f"{name}_non_holiday"])
            # Extra guard: holiday conformal level (≈0.80) sits below the 95th
            # trim, so a heavy holiday tail can still inflate the band — cap vs
            # non-holiday adjustment.
            ratio = float(self.holiday_max_ratio)
            if ratio > 0 and nh_adj > 0:
                hol_adj = min(hol_adj, nh_adj * ratio)
            adjustments[f"{name}_holiday"] = float(hol_adj)

        # Fourth Mondrian cut for Google: brand vs non-brand campaign types.
        for brand in ("brand", "non_brand"):
            brand_key = f"google_{brand}"
            brand_scores = _pool_scores("google", None, brand=brand)
            counts[brand_key] = int(len(brand_scores))
            if len(brand_scores) >= 5:
                adjustments[brand_key] = self._adjustment_from_scores(brand_scores, cov)
            else:
                adjustments[brand_key] = adjustments.get("google", adjustments["global"])

            nh_scores = _pool_scores("google", False, brand=brand)
            nh_key = f"google_{brand}_non_holiday"
            counts[nh_key] = int(len(nh_scores))
            if len(nh_scores) >= 5:
                adjustments[nh_key] = self._adjustment_from_scores(nh_scores, cov)
            else:
                adjustments[nh_key] = adjustments[brand_key]
            nh_cap = self._score_trim_cap(nh_scores, self.score_trim_q)

            hol_key = f"google_{brand}_holiday"
            hol_scores = _pool_scores("google", True, brand=brand)
            counts[hol_key] = int(len(hol_scores))
            if len(hol_scores) >= 5:
                hol_adj = self._adjustment_from_scores(
                    hol_scores, cov, clip_cap=nh_cap
                )
            else:
                hol_adj = adjustments[brand_key]
            nh_adj = float(adjustments[nh_key])
            ratio = float(self.holiday_max_ratio)
            if ratio > 0 and nh_adj > 0:
                hol_adj = min(hol_adj, nh_adj * ratio)
            adjustments[hol_key] = float(hol_adj)

        self.conformal_adjustments = adjustments
        self.conformal_stratum_counts = counts
        self.conformal_radius = float(adjustments["global"])
        self.conformal_rel = 0.0
        return self

    # Back-compat alias
    def calibrate_intervals(self, frame: pd.DataFrame, **kwargs: Any) -> "RevenueQuantileModel":
        return self.calibrate_conformal(frame, **kwargs)

    def _lookup_adjustment(
        self, channel: str, holiday: bool, brand: str = ""
    ) -> float:
        """
        Fallback chain:
          google×brand×season → google×brand → channel×season → channel → global.

        For holiday rows, never apply a narrower adjustment than the matching
        non-holiday stratum (prevents sparse/low-score holiday pools — e.g. Bing —
        from shrinking intervals below off-peak calibration).
        """
        adj = self.conformal_adjustments
        if not adj:
            return 0.0
        global_adj = float(adj.get("global", 0.0))
        ch = channel.lower() if channel else ""
        brand = (brand or "").strip().lower()

        def _floor_holiday(candidate: float, non_hol: float) -> float:
            return max(candidate, non_hol) if holiday else candidate

        # Google brand / non-brand cut (when known).
        if ch == "google" and brand in ("brand", "non_brand"):
            brand_non_hol = float(
                adj.get(f"google_{brand}_non_holiday", adj.get(f"google_{brand}", global_adj))
            )
            if holiday:
                brand_hol = float(
                    adj.get(f"google_{brand}_holiday", brand_non_hol)
                )
                return _floor_holiday(brand_hol, brand_non_hol)
            if f"google_{brand}_non_holiday" in adj or f"google_{brand}" in adj:
                return brand_non_hol

        non_hol_key = f"{ch}_non_holiday"
        non_hol_adj = float(adj.get(non_hol_key, adj.get(ch, global_adj)))
        if holiday:
            season_key = f"{ch}_holiday"
            hol_adj = float(adj.get(season_key, non_hol_adj))
            return max(hol_adj, non_hol_adj)
        if non_hol_key in adj:
            return non_hol_adj
        if ch in adj:
            return float(adj[ch])
        return global_adj

    def _row_adjustments(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-row Mondrian adjustment from channel + season (+ Google brand)."""
        n = len(frame)
        if not self.conformal_adjustments:
            return np.zeros(n, dtype=float)
        hol = season_bucket_mask(frame)
        if "channel" in frame.columns:
            channels = frame["channel"].astype(str).str.lower().to_numpy()
        else:
            channels = np.array([""] * n)
        brands = google_brand_bucket(frame)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            out[i] = self._lookup_adjustment(
                str(channels[i]), bool(hol[i]), brand=str(brands[i])
            )
        return out

    def predict_quantiles(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Raw quantiles + per-channel conformal adjustment.
        p50 is never modified.
        """
        p10, p50, p90 = self._raw_predict_quantiles(frame)
        p50_fixed = p50.copy()
        adj = self._row_adjustments(frame)

        p10_c = p10 - adj
        p90_c = p90 + adj

        # Post-conformal revenue rules (p50 unchanged).
        p10_c = np.maximum(p10_c, 0.0)
        p10_c = np.minimum(p10_c, p50_fixed * 0.95)
        p90_c = np.maximum(p90_c, p50_fixed * 1.05)
        p10_c = np.minimum(p10_c, p50_fixed)
        p90_c = np.maximum(p90_c, p50_fixed)
        return p10_c, p50_fixed, p90_c

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return predictions.csv schema rows (conformalized)."""
        p10, p50, p90 = self.predict_quantiles(features)
        assumed = features["planned_spend"].astype(float).fillna(0.0).to_numpy()
        assumed = np.clip(assumed, a_min=0.0, a_max=None)

        def _roas(rev: np.ndarray) -> np.ndarray:
            return np.divide(rev, assumed, out=np.zeros_like(rev), where=assumed > 0)

        p10_roas, p50_roas, p90_roas = _roas(p10), _roas(p50), _roas(p90)

        # Healthy-median soft floor on p10 ROAS (does not move p50).
        healthy = p50_roas >= 2.0
        p10_roas = np.where(
            healthy, np.maximum(p10_roas, MIN_P10_ROAS_WHEN_HEALTHY), p10_roas
        )
        p10_roas = np.minimum(p10_roas, p50_roas)
        p90_roas = np.maximum(p90_roas, p50_roas)

        # Low-spend ROAS guards (user contract).
        low = assumed < LOW_SPEND_THRESHOLD
        p10_roas = np.where(low, MIN_P10_ROAS_GLOBAL, p10_roas)  # = 0.1
        p90_roas = np.where(low, np.minimum(p90_roas, LOW_SPEND_ROAS_CAP), p90_roas)
        p10_roas = np.minimum(p10_roas, p50_roas)
        p90_roas = np.maximum(p90_roas, p50_roas)

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
