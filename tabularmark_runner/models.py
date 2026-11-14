"""Deterministic downstream-utility models, including XGBoost GPU fallback."""

from __future__ import annotations

import importlib.metadata
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class ModelResult:
    metrics: dict[str, float]
    model: str
    requested_device: str
    used_device: str
    gpu_id: str | None
    fallback_reason: str | None
    missing_value_strategy: str
    input_train_rows: int
    input_test_rows: int
    effective_train_rows: int
    effective_test_rows: int


def split_indices(
    frame: pd.DataFrame,
    target: str,
    *,
    seed: int,
    classification: bool,
    test_size: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(frame))
    stratify = None
    if classification:
        counts = frame[target].value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            stratify = frame[target].to_numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return np.sort(train_idx), np.sort(test_idx)


def _encoded_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    median_impute: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    absent = set(feature_columns).difference(train_frame.columns) | set(
        feature_columns
    ).difference(test_frame.columns)
    if absent:
        raise KeyError(f"model feature columns are missing: {sorted(absent)}")
    train = train_frame.loc[:, list(feature_columns)].copy()
    test = test_frame.loc[:, list(feature_columns)].copy()
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, dummy_na=True, dtype=float)
    combined = combined.apply(pd.to_numeric, errors="coerce")
    combined = combined.replace([np.inf, -np.inf], np.nan)
    train_encoded = combined.iloc[: len(train)].to_numpy(dtype=float)
    test_encoded = combined.iloc[len(train) :].to_numpy(dtype=float)
    if median_impute:
        imputer = SimpleImputer(strategy="median")
        train_encoded = imputer.fit_transform(train_encoded)
        test_encoded = imputer.transform(test_encoded)
    elif not np.isfinite(train_encoded).all() or not np.isfinite(test_encoded).all():
        raise ValueError(
            "complete-case model features still contain non-finite values after encoding"
        )
    return train_encoded, test_encoded


def _label_arrays(
    clean_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    target: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder().fit(clean_frame[target].astype(str))
    known = set(encoder.classes_)
    attacked_classes = set(train_frame.iloc[train_idx][target].astype(str))
    unknown = attacked_classes.difference(known)
    if unknown:
        raise ValueError(
            f"training target {target!r} contains classes absent from clean data: {sorted(unknown)}"
        )
    y_train = encoder.transform(train_frame.iloc[train_idx][target].astype(str))
    y_test = encoder.transform(clean_frame.iloc[test_idx][target].astype(str))
    return y_train, y_test, encoder


def _xgboost_version_major() -> int:
    try:
        return int(importlib.metadata.version("xgboost").split(".", 1)[0])
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return 0


def _new_xgb_classifier(
    *,
    class_count: int,
    seed: int,
    workers: int,
    trees: int,
    max_depth: int,
    device: str,
    gpu_id: str | None,
):
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(
            "Covertype requires xgboost; install requirements.txt before the full run"
        ) from error
    params: dict[str, Any] = {
        "n_estimators": trees,
        "max_depth": max_depth,
        "n_jobs": workers,
        "random_state": seed,
        "seed": seed,
        "tree_method": "hist",
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "eval_metric": "mlogloss" if class_count > 2 else "logloss",
        "verbosity": 0,
    }
    if class_count > 2:
        params.update(objective="multi:softprob", num_class=class_count)
    else:
        params.update(objective="binary:logistic")
    if device == "cuda":
        chosen_gpu = str(gpu_id or "0").split(",", 1)[0].strip()
        if _xgboost_version_major() >= 2:
            params["device"] = f"cuda:{chosen_gpu}"
        else:
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = int(chosen_gpu)
    else:
        params["device"] = "cpu" if _xgboost_version_major() >= 2 else None
        params = {key: value for key, value in params.items() if value is not None}
    return xgb.XGBClassifier(**params)


def _observed_xgb_device(estimator: Any) -> str | None:
    """Return XGBoost's post-fit device, when exposed by XGBoost 2+."""

    if _xgboost_version_major() < 2:
        return None
    try:
        config = json.loads(estimator.get_booster().save_config())
        return str(config["learner"]["generic_param"]["device"]).lower()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot verify the device actually used by XGBoost") from error


def _verify_xgb_device(estimator: Any, requested: str) -> str:
    observed = _observed_xgb_device(estimator)
    if observed is None:
        return requested
    if requested.startswith("cuda"):
        if not observed.startswith("cuda"):
            raise RuntimeError(
                f"XGBoost silently fell back from {requested!r} to {observed!r}"
            )
        if ":" in requested and observed != requested:
            raise RuntimeError(
                f"XGBoost requested {requested!r}, but reports {observed!r}"
            )
    elif observed != "cpu":
        raise RuntimeError(f"XGBoost requested CPU, but reports {observed!r}")
    return observed


def evaluate_classification(
    clean_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    model: str,
    seed: int,
    workers: int,
    rf_trees: int,
    xgb_trees: int,
    xgb_max_depth: int,
    device: str,
    gpu_id: str | None,
) -> ModelResult:
    X_train, X_test = _encoded_features(
        train_frame.iloc[train_idx], clean_frame.iloc[test_idx], feature_columns
    )
    y_train, y_test, encoder = _label_arrays(
        clean_frame, train_frame, target, train_idx, test_idx
    )
    fallback_reason: str | None = None
    if model == "xgboost":
        estimator = _new_xgb_classifier(
            class_count=len(encoder.classes_),
            seed=seed,
            workers=workers,
            trees=xgb_trees,
            max_depth=xgb_max_depth,
            device=device,
            gpu_id=gpu_id,
        )
        try:
            estimator.fit(X_train, y_train)
            requested_device = (
                f"cuda:{str(gpu_id or '0').split(',', 1)[0]}"
                if device == "cuda"
                else "cpu"
            )
            used_device = _verify_xgb_device(estimator, requested_device)
            predictions = estimator.predict(X_test)
        except Exception as gpu_error:
            if device != "cuda":
                raise
            fallback_reason = f"{type(gpu_error).__name__}: {gpu_error}"
            estimator = _new_xgb_classifier(
                class_count=len(encoder.classes_),
                seed=seed,
                workers=workers,
                trees=xgb_trees,
                max_depth=xgb_max_depth,
                device="cpu",
                gpu_id=None,
            )
            estimator.fit(X_train, y_train)
            used_device = _verify_xgb_device(estimator, "cpu")
            predictions = estimator.predict(X_test)
    elif model == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=rf_trees,
            random_state=seed,
            n_jobs=workers,
        )
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        used_device = "cpu"
    elif model == "logistic_regression":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        used_device = "cpu"
    else:
        raise ValueError(f"unknown classifier {model!r}")
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }
    per_class = f1_score(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        average=None,
        zero_division=0,
    )
    for label, value in zip(encoder.classes_, per_class):
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))
        metrics[f"f1_class_{safe_label}"] = float(value)
    return ModelResult(
        metrics=metrics,
        model=model,
        requested_device=device if model == "xgboost" else "cpu",
        used_device=used_device,
        gpu_id=gpu_id if model == "xgboost" and device == "cuda" else None,
        fallback_reason=fallback_reason,
        missing_value_strategy="median_imputation_after_joint_one_hot_encoding",
        input_train_rows=len(train_idx),
        input_test_rows=len(test_idx),
        effective_train_rows=len(train_idx),
        effective_test_rows=len(test_idx),
    )


def evaluate_regression(
    clean_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> ModelResult:
    model_columns = [*feature_columns, target]
    absent = set(model_columns).difference(train_frame.columns) | set(
        model_columns
    ).difference(clean_frame.columns)
    if absent:
        raise KeyError(f"regression model columns are missing: {sorted(absent)}")
    input_train = train_frame.iloc[train_idx]
    input_test = clean_frame.iloc[test_idx]
    complete_train = input_train.dropna(subset=model_columns)
    complete_test = input_test.dropna(subset=model_columns)
    if complete_train.empty or complete_test.empty:
        raise ValueError(
            "regression complete-case filtering removed every training or test row"
        )
    X_train, X_test = _encoded_features(
        complete_train,
        complete_test,
        feature_columns,
        median_impute=False,
    )
    y_train = pd.to_numeric(
        complete_train[target], errors="coerce"
    ).to_numpy(dtype=float)
    y_test = pd.to_numeric(
        complete_test[target], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError(f"regression target {target!r} contains non-finite values")
    estimator = make_pipeline(StandardScaler(), LinearRegression())
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    mse = float(mean_squared_error(y_test, predictions))
    return ModelResult(
        metrics={"mse": mse, "rmse": float(np.sqrt(mse))},
        model="linear_regression",
        requested_device="cpu",
        used_device="cpu",
        gpu_id=None,
        fallback_reason=None,
        missing_value_strategy="complete_case_after_split_on_predictors_plus_target",
        input_train_rows=len(input_train),
        input_test_rows=len(input_test),
        effective_train_rows=len(complete_train),
        effective_test_rows=len(complete_test),
    )


def result_metric_rows(result: ModelResult, base: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric, value in result.metrics.items():
        rows.append(
            {
                **base,
                "model": result.model,
                "metric": metric,
                "value": value,
                "requested_device": result.requested_device,
                "used_device": result.used_device,
                "gpu_id": result.gpu_id,
                "fallback_reason": result.fallback_reason,
                "missing_value_strategy": result.missing_value_strategy,
                "input_train_rows": result.input_train_rows,
                "input_test_rows": result.input_test_rows,
                "effective_train_rows": result.effective_train_rows,
                "effective_test_rows": result.effective_test_rows,
                "dropped_train_rows": (
                    result.input_train_rows - result.effective_train_rows
                ),
                "dropped_test_rows": (
                    result.input_test_rows - result.effective_test_rows
                ),
            }
        )
    return rows
