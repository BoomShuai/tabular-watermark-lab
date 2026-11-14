"""Executable RQ1/RQ2/RQ3/RQ5 experiment stages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .core import (
    DetectionDetails,
    MatchedDetectionDetails,
    WatermarkKey,
    atomic_csv,
    atomic_json,
    atomic_private_json,
    categorical_effective_gamma,
    decision_threshold,
    detect_categorical_details,
    detect_categorical_matched_details,
    detect_numeric,
    embed_categorical,
    embed_numeric,
    make_key,
    match_key_rows,
    sample_sd,
    stable_seed,
)
from .datasets import DatasetBundle, load_many, resolve_column
from .models import (
    ModelResult,
    evaluate_classification,
    evaluate_regression,
    result_metric_rows,
    split_indices,
)


GAMMA_SEMANTICS = {
    "gamma": "legacy alias of nominal_gamma",
    "nominal_gamma": "requested watermark-key gamma",
    "effective_gamma": (
        "categorical: floor(category_count * nominal_gamma) / category_count; "
        "numeric: nominal_gamma"
    ),
    "z_score": "computed with effective_gamma as the null green probability",
}


@dataclass(frozen=True)
class ExperimentContext:
    data_dir: Path
    output_dir: Path
    seed: int
    alpha: float
    allow_downloads: bool
    utility_repeats: int
    attack_repeats: int
    sweep_repeats: int
    roc_neighbors: int
    workers: int
    rf_trees: int
    xgb_trees: int
    xgb_max_depth: int
    device: str
    gpu_ids: tuple[str, ...]
    attack_proportions: tuple[float, ...]

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "input_cache"

    @property
    def artifacts_dir(self) -> Path:
        return self.output_dir / "artifacts"

    def gpu_id_for(self, *parts: object) -> str | None:
        if self.device != "cuda" or not self.gpu_ids:
            return None
        index = stable_seed(self.seed, *parts) % len(self.gpu_ids)
        return self.gpu_ids[index]

    def datasets(self, names: Sequence[str]) -> dict[str, DatasetBundle]:
        return load_many(
            names,
            self.data_dir,
            self.cache_dir,
            seed=self.seed,
            allow_downloads=self.allow_downloads,
        )


def _categories(frame: pd.DataFrame, column: str) -> list[Any]:
    return sorted(pd.unique(frame[column]).tolist(), key=lambda value: str(value))


def _embed_bundle(
    context: ExperimentContext,
    bundle: DatasetBundle,
    *,
    key_label: str = "paper-default",
    n: int | None = None,
    p: float | None = None,
    gamma: float = 0.5,
    persist: bool = True,
) -> tuple[pd.DataFrame, WatermarkKey, list[Any] | None]:
    actual_n = int(bundle.n if n is None else n)
    actual_p = bundle.p if p is None else float(p)
    key = make_key(
        len(bundle.frame),
        actual_n,
        stable_seed(context.seed, "watermark-key", bundle.name, key_label),
        gamma,
    )
    categories: list[Any] | None = None
    if bundle.watermark_kind == "numeric":
        if actual_p is None:
            raise ValueError(f"{bundle.name}: no p specified for numeric watermark")
        watermarked = embed_numeric(
            bundle.frame,
            bundle.watermark_column,
            key,
            p=actual_p,
            k=bundle.k,
        )
    else:
        categories = _categories(bundle.frame, bundle.watermark_column)
        watermarked = embed_categorical(
            bundle.frame, bundle.watermark_column, key, categories
        )
    if persist:
        effective_gamma = (
            key.gamma
            if categories is None
            else categorical_effective_gamma(categories, key.gamma)
        )
        watermarked_path = context.artifacts_dir / "watermarked" / f"{bundle.name}.csv"
        key_path = context.artifacts_dir / "keys" / f"{bundle.name}.json"
        atomic_csv(watermarked_path, watermarked)
        atomic_private_json(
            key_path,
            {
                "dataset": bundle.name,
                "column": bundle.watermark_column,
                "kind": bundle.watermark_kind,
                "p": actual_p,
                "k": bundle.k,
                "categories": categories,
                "nominal_gamma": key.gamma,
                "effective_gamma": effective_gamma,
                "key": key.to_dict(),
                "seed_derivation_label": key_label,
            },
        )
    return watermarked, key, categories


def _detect_bundle(
    bundle: DatasetBundle,
    suspect: pd.DataFrame,
    key: WatermarkKey,
    categories: Sequence[Any] | None,
    *,
    p: float | None = None,
) -> DetectionDetails:
    if bundle.watermark_kind == "numeric":
        actual_p = bundle.p if p is None else p
        if actual_p is None:
            raise ValueError(f"{bundle.name}: numeric detection requires p")
        green_count, z_score = detect_numeric(
            bundle.frame,
            suspect,
            bundle.watermark_column,
            key,
            actual_p,
            bundle.k,
        )
        return DetectionDetails(
            green_count=green_count,
            z_score=z_score,
            nominal_gamma=key.gamma,
            effective_gamma=key.gamma,
        )
    if categories is None:
        raise ValueError(f"{bundle.name}: categorical detection requires categories")
    return detect_categorical_details(
        suspect, bundle.watermark_column, key, categories
    )


def _gamma_fields(
    details: DetectionDetails | MatchedDetectionDetails,
) -> dict[str, float]:
    return {
        "nominal_gamma": details.nominal_gamma,
        "effective_gamma": details.effective_gamma,
    }


def _attack_alteration(
    bundle: DatasetBundle,
    watermarked: pd.DataFrame,
    proportion: float,
    seed: int,
    *,
    numeric_range: float | None = None,
) -> pd.DataFrame:
    if not 0.0 <= proportion <= 1.0:
        raise ValueError(f"attack proportion must be in [0, 1], got {proportion}")
    attacked = watermarked.copy(deep=True)
    count = int(round(proportion * len(attacked)))
    if count == 0:
        return attacked
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(attacked), size=count, replace=False)
    if bundle.watermark_kind == "categorical":
        categories = _categories(bundle.frame, bundle.watermark_column)
        attacked.loc[indices, bundle.watermark_column] = rng.choice(
            np.asarray(categories), size=count, replace=True
        )
    else:
        radius = numeric_range
        if radius is None:
            radius = 2.0 * float(bundle.p or 1.0)
        attacked.loc[indices, bundle.watermark_column] = (
            pd.to_numeric(attacked.loc[indices, bundle.watermark_column]).to_numpy()
            + rng.uniform(-radius, radius, size=count)
        )
    return attacked


def _model_for(name: str) -> str:
    return {
        "covertype": "xgboost",
        "hog": "random_forest",
        "zoo": "random_forest",
        "iris": "random_forest",
        "synthetic": "logistic_regression",
    }[name]


def _evaluate_model(
    context: ExperimentContext,
    bundle: DatasetBundle,
    train_frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    gpu_label: Sequence[object],
) -> ModelResult:
    if bundle.name == "housing":
        return evaluate_regression(
            bundle.frame,
            train_frame,
            bundle.target,
            bundle.feature_columns,
            train_idx,
            test_idx,
        )
    return evaluate_classification(
        bundle.frame,
        train_frame,
        bundle.target,
        bundle.feature_columns,
        train_idx,
        test_idx,
        model=_model_for(bundle.name),
        seed=seed,
        workers=context.workers,
        rf_trees=context.rf_trees,
        xgb_trees=context.xgb_trees,
        xgb_max_depth=context.xgb_max_depth,
        device=context.device,
        gpu_id=context.gpu_id_for(*gpu_label),
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def _write_stage(
    context: ExperimentContext,
    stem: str,
    rows: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[Path]:
    frame = pd.DataFrame(list(rows))
    csv_path = context.output_dir / f"{stem}.csv"
    json_path = context.output_dir / f"{stem}.json"
    atomic_csv(csv_path, frame)
    atomic_json(
        json_path,
        {
            "stage": stem,
            **metadata,
            "row_count": len(frame),
            "records": _records(frame),
        },
    )
    return [csv_path, json_path]


def _detection_row(
    bundle: DatasetBundle,
    suspect_name: str,
    suspect: pd.DataFrame,
    key: WatermarkKey,
    categories: Sequence[Any] | None,
    threshold: float,
) -> dict[str, Any]:
    details = _detect_bundle(bundle, suspect, key, categories)
    return {
        "dataset": bundle.name,
        "suspect": suspect_name,
        "n": key.n,
        "gamma": details.nominal_gamma,
        **_gamma_fields(details),
        "green_count": details.green_count,
        "z_score": details.z_score,
        "threshold": threshold,
        "detected": bool(details.z_score > threshold),
    }


def run_rq1_detectability(context: ExperimentContext) -> list[Path]:
    bundles = context.datasets(("synthetic", "covertype", "hog", "housing"))
    threshold = decision_threshold(context.alpha)
    rows: list[dict[str, Any]] = []
    common_artifacts: list[Path] = []
    for name, bundle in bundles.items():
        watermarked, key, categories = _embed_bundle(context, bundle)
        common_artifacts.extend(
            [
                context.artifacts_dir / "watermarked" / f"{name}.csv",
                context.artifacts_dir / "keys" / f"{name}.json",
            ]
        )
        perturbed = _attack_alteration(
            bundle,
            bundle.frame,
            1.0,
            stable_seed(context.seed, "rq1-perturbed", name),
            numeric_range=float(bundle.p or 1.0),
        )
        for suspect_name, suspect in (
            ("original", bundle.frame),
            ("watermarked", watermarked),
            ("perturbed_original", perturbed),
        ):
            rows.append(
                _detection_row(
                    bundle, suspect_name, suspect, key, categories, threshold
                )
            )
    outputs = _write_stage(
        context,
        "rq1_detectability",
        rows,
        {
            "rq": "RQ1",
            "threshold": threshold,
            "decision": "detected iff z_score > threshold",
            "gamma_semantics": GAMMA_SEMANTICS,
            "datasets": [bundle.metadata() for bundle in bundles.values()],
            "protocol": "paper n/p/k values; corrected accumulating embedding copied from reproduce.py",
        },
    )
    return outputs + common_artifacts


def _numeric_attack(
    frame: pd.DataFrame,
    column: str,
    proportion: float,
    radius: float,
    seed: int,
) -> pd.DataFrame:
    attacked = frame.copy(deep=True)
    count = int(round(proportion * len(attacked)))
    if count:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(attacked), size=count, replace=False)
        attacked.loc[indices, column] = (
            attacked.loc[indices, column].to_numpy(dtype=float)
            + rng.uniform(-radius, radius, size=count)
        )
    return attacked


def _roc_rows_for_synthetic(
    context: ExperimentContext, bundle: DatasetBundle, total: int
) -> list[dict[str, Any]]:
    watermarked, key, categories = _embed_bundle(
        context, bundle, key_label="rq1-roc-synthetic", persist=False
    )
    negative_independent = total // 4
    negative_perturbed = total // 4
    positive = total - negative_independent - negative_perturbed
    rows = []
    attack_levels = (0.2, 0.4, 0.6, 0.8)
    source_values = bundle.frame[bundle.watermark_column].to_numpy(dtype=float)
    source_mean = float(np.mean(source_values))
    source_sd = float(np.std(source_values, ddof=1))
    for index in range(negative_independent):
        rng = np.random.default_rng(
            stable_seed(context.seed, "roc", bundle.name, "independent", index)
        )
        suspect = bundle.frame.copy(deep=True)
        suspect[bundle.watermark_column] = rng.normal(
            source_mean, source_sd, size=len(suspect)
        )
        details = _detect_bundle(bundle, suspect, key, categories)
        rows.append(
            {
                "dataset": bundle.name,
                "neighbor_type": "independently_generated",
                "neighbor_index": index,
                "label": 0,
                "attack_proportion": None,
                **_gamma_fields(details),
                "green_count": details.green_count,
                "z_score": details.z_score,
            }
        )
    for index in range(negative_perturbed):
        proportion = attack_levels[index % 3]
        suspect = _numeric_attack(
            bundle.frame,
            bundle.watermark_column,
            proportion,
            float(bundle.p),
            stable_seed(context.seed, "roc", bundle.name, "original", index),
        )
        details = _detect_bundle(bundle, suspect, key, categories)
        rows.append(
            {
                "dataset": bundle.name,
                "neighbor_type": "perturbed_original",
                "neighbor_index": index,
                "label": 0,
                "attack_proportion": proportion,
                **_gamma_fields(details),
                "green_count": details.green_count,
                "z_score": details.z_score,
            }
        )
    for index in range(positive):
        proportion = attack_levels[index % len(attack_levels)]
        suspect = _numeric_attack(
            watermarked,
            bundle.watermark_column,
            proportion,
            float(bundle.p),
            stable_seed(context.seed, "roc", bundle.name, "watermarked", index),
        )
        details = _detect_bundle(bundle, suspect, key, categories)
        rows.append(
            {
                "dataset": bundle.name,
                "neighbor_type": "perturbed_watermarked",
                "neighbor_index": index,
                "label": 1,
                "attack_proportion": proportion,
                **_gamma_fields(details),
                "green_count": details.green_count,
                "z_score": details.z_score,
            }
        )
    return rows


def _roc_rows_for_housing(
    context: ExperimentContext, bundle: DatasetBundle, total: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(stable_seed(context.seed, "housing-roc-half-split"))
    permutation = rng.permutation(len(bundle.frame))
    half_size = len(bundle.frame) // 2
    a = bundle.frame.iloc[permutation[:half_size]].reset_index(drop=True)
    b = bundle.frame.iloc[permutation[half_size : 2 * half_size]].reset_index(drop=True)
    local = DatasetBundle(
        **{
            **bundle.__dict__,
            "frame": a,
            "n": min(bundle.n, len(a)),
            "source": bundle.source + " [deterministic half A]",
        }
    )
    watermarked, key, categories = _embed_bundle(
        context, local, key_label="rq1-roc-housing-half-a", persist=False
    )
    negative_b = total // 4
    negative_a = total // 4
    positive = total - negative_b - negative_a
    rows = []
    attack_levels = (0.2, 0.4, 0.6, 0.8)
    for index, (neighbor_type, label, count, base) in enumerate(
        (
            ("perturbed_half_b", 0, negative_b, b),
            ("perturbed_half_a", 0, negative_a, a),
            ("perturbed_watermarked_half_a", 1, positive, watermarked),
        )
    ):
        del index
        for neighbor_index in range(count):
            proportion = attack_levels[neighbor_index % len(attack_levels)]
            suspect = _numeric_attack(
                base,
                local.watermark_column,
                proportion,
                float(local.p),
                stable_seed(
                    context.seed, "roc", bundle.name, neighbor_type, neighbor_index
                ),
            )
            details = _detect_bundle(local, suspect, key, categories)
            rows.append(
                {
                    "dataset": bundle.name,
                    "neighbor_type": neighbor_type,
                    "neighbor_index": neighbor_index,
                    "label": label,
                    "attack_proportion": proportion,
                    **_gamma_fields(details),
                    "green_count": details.green_count,
                    "z_score": details.z_score,
                }
            )
    return rows


def run_rq1_roc(context: ExperimentContext) -> list[Path]:
    if context.roc_neighbors < 4:
        raise ValueError("--roc-neighbors must be at least four")
    bundles = context.datasets(("synthetic", "housing"))
    rows = _roc_rows_for_synthetic(
        context, bundles["synthetic"], context.roc_neighbors
    ) + _roc_rows_for_housing(context, bundles["housing"], context.roc_neighbors)
    frame = pd.DataFrame(rows)
    summaries = []
    curve_rows = []
    for name, group in frame.groupby("dataset", sort=False):
        auc = float(roc_auc_score(group["label"], group["z_score"]))
        fpr, tpr, thresholds = roc_curve(group["label"], group["z_score"])
        nominal_gamma = float(group["nominal_gamma"].iloc[0])
        effective_gamma = float(group["effective_gamma"].iloc[0])
        summaries.append(
            {
                "dataset": name,
                "auc": auc,
                "neighbors": len(group),
                "positive": int(group["label"].sum()),
                "negative": int((group["label"] == 0).sum()),
                "nominal_gamma": nominal_gamma,
                "effective_gamma": effective_gamma,
            }
        )
        for false_positive, true_positive, threshold in zip(fpr, tpr, thresholds):
            curve_rows.append(
                {
                    "dataset": name,
                    "false_positive_rate": float(false_positive),
                    "true_positive_rate": float(true_positive),
                    "score_threshold": float(threshold),
                    "auc": auc,
                    "nominal_gamma": nominal_gamma,
                    "effective_gamma": effective_gamma,
                }
            )
    neighbor_csv = context.output_dir / "rq1_roc_neighbors.csv"
    curve_csv = context.output_dir / "rq1_roc_curve.csv"
    summary_json = context.output_dir / "rq1_roc.json"
    atomic_csv(neighbor_csv, frame)
    atomic_csv(curve_csv, pd.DataFrame(curve_rows))
    atomic_json(
        summary_json,
        {
            "stage": "rq1_roc",
            "rq": "RQ1",
            "neighbors_per_dataset": context.roc_neighbors,
            "gamma_semantics": GAMMA_SEMANTICS,
            "summary": summaries,
            "datasets": [bundle.metadata() for bundle in bundles.values()],
            "protocol": {
                "synthetic": "1/4 independent, 1/4 perturbed original, 1/2 perturbed watermarked",
                "housing": "deterministic A/B halves; 1/4 perturbed B, 1/4 perturbed A, 1/2 perturbed watermarked A",
                "score": "TabularMark z-score",
            },
        },
    )
    return [neighbor_csv, curve_csv, summary_json]


def _summarize_metrics(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> list[dict[str, Any]]:
    rows = []
    for keys, group in frame.groupby(list(group_columns), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        values = group["value"].to_numpy(dtype=float)
        row.update(repeats=len(values), mean=float(np.mean(values)), sd=sample_sd(values))
        rows.append(row)
    return rows


def _housing_utility_missing_data(
    frame: pd.DataFrame, bundle: DatasetBundle
) -> dict[str, Any]:
    """Describe Housing complete-case samples once per evaluated model fit."""

    selected = frame.loc[
        (frame["dataset"] == "housing") & frame["missing_value_strategy"].notna()
    ].copy()
    if "result_type" in selected.columns:
        selected = selected.loc[selected["result_type"] == "utility"]
    identity_columns = [
        column
        for column in (
            "variant",
            "repeat",
            "split_seed",
            "attack",
            "attack_proportion",
            "attack_seed",
            "model",
        )
        if column in selected.columns
    ]
    count_columns = [
        "input_train_rows",
        "effective_train_rows",
        "dropped_train_rows",
        "input_test_rows",
        "effective_test_rows",
        "dropped_test_rows",
    ]
    sample_columns = ["missing_value_strategy", *count_columns]
    samples = selected.loc[:, [*identity_columns, *sample_columns]].drop_duplicates()
    for column in count_columns:
        samples[column] = pd.to_numeric(samples[column], errors="raise").astype(int)
    return {
        "strategy": "complete_case_after_split_on_predictors_plus_target",
        "complete_case_columns": [*bundle.feature_columns, bundle.target],
        "imputation": None,
        "effective_samples_by_fit": _records(samples),
    }


def run_rq2_utility(context: ExperimentContext) -> list[Path]:
    bundles = context.datasets(("covertype", "hog", "housing", "zoo"))
    rows: list[dict[str, Any]] = []
    for name, bundle in bundles.items():
        watermarked, _, _ = _embed_bundle(context, bundle)
        for repeat in range(context.utility_repeats):
            split_seed = stable_seed(context.seed, "rq2-split", name, repeat)
            train_idx, test_idx = split_indices(
                bundle.frame,
                bundle.target,
                seed=split_seed,
                classification=name != "housing",
            )
            model_seed = stable_seed(context.seed, "rq2-model", name, repeat)
            for variant, train_frame in (
                ("original", bundle.frame),
                ("watermarked", watermarked),
            ):
                result = _evaluate_model(
                    context,
                    bundle,
                    train_frame,
                    train_idx,
                    test_idx,
                    seed=model_seed,
                    gpu_label=("rq2", name, repeat, variant),
                )
                rows.extend(
                    result_metric_rows(
                        result,
                        {
                            "dataset": name,
                            "variant": variant,
                            "repeat": repeat,
                            "split_seed": split_seed,
                            "model_seed": model_seed,
                        },
                    )
                )
    frame = pd.DataFrame(rows)
    return _write_stage(
        context,
        "rq2_utility",
        rows,
        {
            "rq": "RQ2",
            "datasets": [bundle.metadata() for bundle in bundles.values()],
            "housing_utility_missing_data": _housing_utility_missing_data(
                frame, bundles["housing"]
            ),
            "summary": _summarize_metrics(
                frame, ("dataset", "variant", "model", "metric", "used_device")
            ),
            "protocol": (
                "shared clean 30% holdout per repeat; train original/watermarked "
                "variants; evaluate against clean targets; Housing drops incomplete "
                "predictors+MEDV rows after splitting and performs no imputation"
            ),
        },
    )


def run_rq3_alteration(context: ExperimentContext) -> list[Path]:
    bundles = context.datasets(("covertype", "hog", "housing", "zoo"))
    threshold = decision_threshold(context.alpha)
    rows: list[dict[str, Any]] = []
    for name, bundle in bundles.items():
        watermarked, key, categories = _embed_bundle(context, bundle)
        train_idx, test_idx = split_indices(
            bundle.frame,
            bundle.target,
            seed=42,
            classification=name != "housing",
        )
        for proportion in context.attack_proportions:
            for repeat in range(context.attack_repeats):
                attack_seed = stable_seed(
                    context.seed, "rq3-alteration", name, proportion, repeat
                )
                attacked = _attack_alteration(
                    bundle,
                    watermarked,
                    proportion,
                    attack_seed,
                    numeric_range=40.0 if name == "housing" else None,
                )
                details = _detect_bundle(bundle, attacked, key, categories)
                base = {
                    "dataset": name,
                    "attack": "alteration",
                    "attack_proportion": proportion,
                    "repeat": repeat,
                    "attack_seed": attack_seed,
                    "gamma": details.nominal_gamma,
                    **_gamma_fields(details),
                    "green_count": details.green_count,
                    "n": key.n,
                    "threshold": threshold,
                    "detected": bool(details.z_score > threshold),
                }
                rows.append(
                    {
                        **base,
                        "result_type": "detection",
                        "model": None,
                        "metric": "z_score",
                        "value": details.z_score,
                        "requested_device": None,
                        "used_device": None,
                        "gpu_id": None,
                        "fallback_reason": None,
                    }
                )
                result = _evaluate_model(
                    context,
                    bundle,
                    attacked,
                    train_idx,
                    test_idx,
                    seed=stable_seed(
                        context.seed, "rq3-model", name, proportion, repeat
                    ),
                    gpu_label=("rq3-alteration", name, proportion, repeat),
                )
                utility_rows = result_metric_rows(
                    result, {**base, "result_type": "utility"}
                )
                rows.extend(utility_rows)
    frame = pd.DataFrame(rows)
    return _write_stage(
        context,
        "rq3_alteration",
        rows,
        {
            "rq": "RQ3",
            "threshold": threshold,
            "gamma_semantics": GAMMA_SEMANTICS,
            "datasets": [bundle.metadata() for bundle in bundles.values()],
            "housing_utility_missing_data": _housing_utility_missing_data(
                frame, bundles["housing"]
            ),
            "summary": _summarize_metrics(
                frame,
                (
                    "dataset",
                    "attack_proportion",
                    "nominal_gamma",
                    "effective_gamma",
                    "result_type",
                    "model",
                    "metric",
                ),
            ),
            "protocol": (
                "paper alteration proportions; categorical targets uniformly replaced; "
                "Housing adds U[-40,40] noise and its utility model drops incomplete "
                "predictors+MEDV rows after splitting without imputation"
            ),
        },
    )


def _insert_rows(frame: pd.DataFrame, proportion: float, seed: int) -> pd.DataFrame:
    count = int(round(proportion * len(frame)))
    if count == 0:
        return frame.copy(deep=True)
    rng = np.random.default_rng(seed)
    sampled = frame.iloc[rng.choice(len(frame), size=count, replace=True)].copy()
    expanded = pd.concat([frame, sampled], ignore_index=True)
    return expanded.iloc[rng.permutation(len(expanded))].reset_index(drop=True)


def _delete_rows(frame: pd.DataFrame, proportion: float, seed: int) -> pd.DataFrame:
    count = int(round(proportion * len(frame)))
    if count == 0:
        return frame.copy(deep=True)
    if count >= len(frame):
        return frame.iloc[0:0].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    deleted = set(int(x) for x in rng.choice(len(frame), size=count, replace=False))
    keep = [index for index in range(len(frame)) if index not in deleted]
    return frame.iloc[keep].reset_index(drop=True)


def run_rq3_row_attacks(context: ExperimentContext) -> list[Path]:
    bundle = context.datasets(("covertype",))["covertype"]
    if len(bundle.match_columns) < 2:
        raise ValueError(
            "covertype: insertion/deletion matching needs at least two resolved feature columns; "
            f"resolved={bundle.match_columns}"
        )
    watermarked, key, categories = _embed_bundle(context, bundle)
    assert categories is not None
    threshold = decision_threshold(context.alpha)
    rows = []
    widths = [2] + ([3] if len(bundle.match_columns) >= 3 else [])
    for attack_name, attack_function in (
        ("insertion", _insert_rows),
        ("deletion", _delete_rows),
    ):
        for proportion in context.attack_proportions:
            for repeat in range(context.attack_repeats):
                attack_seed = stable_seed(
                    context.seed, "rq3-row", attack_name, proportion, repeat
                )
                attacked = attack_function(watermarked, proportion, attack_seed)
                for width in widths:
                    columns = bundle.match_columns[:width]
                    matches = match_key_rows(
                        bundle.frame, attacked, key.indices, columns
                    )
                    details = detect_categorical_matched_details(
                        attacked,
                        bundle.watermark_column,
                        key,
                        categories,
                        matches,
                    )
                    rows.append(
                        {
                            "dataset": "covertype",
                            "attack": attack_name,
                            "attack_proportion": proportion,
                            "repeat": repeat,
                            "attack_seed": attack_seed,
                            "matching_width": width,
                            "matching_columns": "|".join(columns),
                            "original_rows": len(bundle.frame),
                            "suspect_rows": len(attacked),
                            "n": key.n,
                            "gamma": details.nominal_gamma,
                            **_gamma_fields(details),
                            "matched_key_rows": details.matched_count,
                            "green_count": details.green_count,
                            "z_score": details.z_score,
                            "threshold": threshold,
                            "detected": bool(details.z_score > threshold),
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = []
    for keys, group in frame.groupby(
        ["attack", "attack_proportion", "matching_width", "matching_columns"],
        sort=False,
    ):
        summary.append(
            {
                "attack": keys[0],
                "attack_proportion": keys[1],
                "matching_width": keys[2],
                "matching_columns": keys[3],
                "nominal_gamma": float(group["nominal_gamma"].iloc[0]),
                "effective_gamma": float(group["effective_gamma"].iloc[0]),
                "z_mean": float(group["z_score"].mean()),
                "z_sd": sample_sd(group["z_score"]),
                "matched_mean": float(group["matched_key_rows"].mean()),
                "detection_rate": float(group["detected"].mean()),
            }
        )
    return _write_stage(
        context,
        "rq3_row_attacks",
        rows,
        {
            "rq": "RQ3",
            "threshold": threshold,
            "gamma_semantics": GAMMA_SEMANTICS,
            "dataset": bundle.metadata(),
            "summary": summary,
            "protocol": "random insertion/deletion followed by O(n) exact composite-key matching; unmatched original key cells count as non-green",
            "paper_alignment_note": "released notebooks compare exact Elevation/Aspect(/third feature) values although the paper describes selected MSBs",
        },
    )


def _resolve_iris_anonymization_columns(bundle: DatasetBundle) -> list[list[str]]:
    alias_groups = [
        ("petal_width", "petal width (cm)"),
        ("petal_length", "petal length (cm)"),
        ("sepal_width", "sepal width (cm)"),
    ]
    resolved = []
    for aliases in alias_groups:
        resolved.append(
            resolve_column(
                bundle.frame,
                aliases,
                dataset="iris",
                purpose="k-anonymization quasi-identifier",
            )
        )
    petal_width, petal_length, sepal_width = resolved
    return [
        [petal_width],
        [petal_length, petal_width],
        [sepal_width, petal_length, petal_width],
    ]


def _microaggregate_k_anonymous(
    frame: pd.DataFrame, columns: Sequence[str], k_anonymity: int
) -> pd.DataFrame:
    if k_anonymity < 2:
        raise ValueError("k_anonymity must be at least two")
    attacked = frame.copy(deep=True)
    order = attacked.sort_values(list(columns), kind="mergesort").index.to_numpy()
    for start in range(0, len(order), k_anonymity):
        group = order[start : start + k_anonymity]
        if len(group) < k_anonymity and start > 0:
            group = order[max(0, len(order) - k_anonymity) :]
        for column in columns:
            attacked.loc[group, column] = float(
                pd.to_numeric(attacked.loc[group, column], errors="raise").mean()
            )
    return attacked


def _class_conditional_gaussian(
    frame: pd.DataFrame,
    target: str,
    feature_columns: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric_features) != len(feature_columns):
        non_numeric = sorted(set(feature_columns).difference(numeric_features))
        raise ValueError(
            "feasible Iris synthesizer currently requires numeric features; "
            f"non-numeric={non_numeric}"
        )
    rng = np.random.default_rng(seed)
    generated_parts = []
    for label, group in frame.groupby(target, sort=True):
        values = group[numeric_features].to_numpy(dtype=float)
        mean = values.mean(axis=0)
        if len(values) > 1:
            covariance = np.atleast_2d(np.cov(values, rowvar=False))
        else:
            covariance = np.eye(len(numeric_features)) * 1e-6
        scale = max(float(np.trace(covariance)), 1.0)
        covariance = covariance + np.eye(len(numeric_features)) * scale * 1e-8
        synthetic = rng.multivariate_normal(mean, covariance, size=len(group))
        part = pd.DataFrame(synthetic, columns=numeric_features)
        part[target] = label
        generated_parts.append(part)
    generated = pd.concat(generated_parts, ignore_index=True)
    return generated.iloc[rng.permutation(len(generated))].reset_index(drop=True)


def run_rq3_iris_attacks(context: ExperimentContext) -> list[Path]:
    bundle = context.datasets(("iris",))["iris"]
    watermarked, key, categories = _embed_bundle(context, bundle)
    assert categories is not None
    threshold = decision_threshold(context.alpha)
    train_idx, test_idx = split_indices(
        bundle.frame, bundle.target, seed=42, classification=True
    )
    rows: list[dict[str, Any]] = []

    def append_model_rows(
        attack: str,
        variant: str,
        train_frame: pd.DataFrame,
        *,
        z_score: float | None,
        green_count: int | None,
        matched_count: int | None,
        nominal_gamma: float,
        effective_gamma: float,
        anonymized_columns: Sequence[str] = (),
        note: str | None = None,
    ) -> None:
        result = _evaluate_model(
            context,
            bundle,
            train_frame,
            train_idx,
            test_idx,
            seed=stable_seed(context.seed, "rq3-iris-model", attack, variant),
            gpu_label=("rq3-iris", attack, variant),
        )
        base = {
            "dataset": "iris",
            "attack": attack,
            "variant": variant,
            "k_anonymity": 3 if attack.startswith("k_anonymity") else None,
            "anonymized_column_count": len(anonymized_columns),
            "anonymized_columns": "|".join(anonymized_columns) or None,
            "matched_key_rows": matched_count,
            "gamma": nominal_gamma,
            "nominal_gamma": nominal_gamma,
            "effective_gamma": effective_gamma,
            "green_count": green_count,
            "z_score": z_score,
            "threshold": threshold,
            "detected": None if z_score is None else bool(z_score > threshold),
            "note": note,
        }
        rows.extend(result_metric_rows(result, base))

    for variant, train_frame in (("original", bundle.frame), ("watermarked", watermarked)):
        details = detect_categorical_details(
            train_frame, bundle.watermark_column, key, categories
        )
        append_model_rows(
            "baseline",
            variant,
            train_frame,
            z_score=details.z_score,
            green_count=details.green_count,
            matched_count=key.n,
            nominal_gamma=details.nominal_gamma,
            effective_gamma=details.effective_gamma,
        )
    for columns in _resolve_iris_anonymization_columns(bundle):
        attacked = _microaggregate_k_anonymous(watermarked, columns, 3)
        remaining = [column for column in bundle.feature_columns if column not in columns]
        if not remaining:
            raise ValueError("Iris k-anonymization left no columns for row matching")
        matches = match_key_rows(bundle.frame, attacked, key.indices, remaining)
        details = detect_categorical_matched_details(
            attacked,
            bundle.watermark_column,
            key,
            categories,
            matches,
        )
        append_model_rows(
            f"k_anonymity_{len(columns)}_columns",
            "watermarked_attacked",
            attacked,
            z_score=details.z_score,
            green_count=details.green_count,
            matched_count=details.matched_count,
            nominal_gamma=details.nominal_gamma,
            effective_gamma=details.effective_gamma,
            anonymized_columns=columns,
            note="k=3 deterministic sorted microaggregation; remaining features used for exact row matching",
        )
    for source_variant, source in (("original", bundle.frame), ("watermarked", watermarked)):
        synthetic = _class_conditional_gaussian(
            source,
            bundle.target,
            bundle.feature_columns,
            stable_seed(context.seed, "rq3-iris-synthetic", source_variant),
        )
        append_model_rows(
            "synthetic_generation",
            f"generated_from_{source_variant}",
            synthetic,
            z_score=None,
            green_count=None,
            matched_count=0,
            nominal_gamma=key.gamma,
            effective_gamma=categorical_effective_gamma(categories, key.gamma),
            note="feasible class-conditional Gaussian synthesizer; no original-to-synthetic row correspondence, so non-blind watermark detection is undefined",
        )
    frame = pd.DataFrame(rows)
    return _write_stage(
        context,
        "rq3_iris_attacks",
        rows,
        {
            "rq": "RQ3",
            "threshold": threshold,
            "gamma_semantics": GAMMA_SEMANTICS,
            "dataset": bundle.metadata(),
            "summary": _summarize_metrics(
                frame,
                (
                    "attack",
                    "variant",
                    "nominal_gamma",
                    "effective_gamma",
                    "metric",
                    "detected",
                ),
            ),
            "protocol": {
                "k_anonymity": "k=3 sorted microaggregation using paper column sets; match with remaining features",
                "synthetic_generation": "auditable feasible class-conditional multivariate Gaussian, not a bit-level recreation of the paper's unspecified generator",
            },
        },
    )


def _sweep_settings(sigma: float) -> dict[str, list[dict[str, Any]]]:
    return {
        "p": [
            {
                "setting": multiplier,
                "setting_label": f"{multiplier:g}sigma",
                "p": multiplier * sigma,
                "n": 300,
                "gamma": 0.5,
                "attack_range": 2.0 * sigma,
            }
            for multiplier in (0.5, 1.0, 1.5, 2.0, 2.5)
        ],
        "n": [
            {
                "setting": n,
                "setting_label": str(n),
                "p": 2.0 * sigma,
                "n": n,
                "gamma": 0.5,
                "attack_range": 2.0 * sigma,
            }
            for n in (100, 200, 300, 400, 500)
        ],
        "gamma": [
            {
                "setting": gamma,
                "setting_label": f"{gamma:.2f}",
                "p": 1.5 * sigma,
                "n": 300,
                "gamma": gamma,
                "attack_range": 1.5 * sigma,
            }
            for gamma in (0.25, 0.33, 0.50, 0.67, 0.75)
        ],
    }


def run_rq5_sweeps(context: ExperimentContext) -> list[Path]:
    bundle = context.datasets(("synthetic",))["synthetic"]
    if bundle.target == bundle.watermark_column:
        raise ValueError(
            "synthetic: RQ5 needs a feature watermark column distinct from the classification target"
        )
    sigma = 20.0
    threshold = decision_threshold(context.alpha)
    train_idx, test_idx = split_indices(
        bundle.frame, bundle.target, seed=42, classification=True
    )
    rows = []
    proportions = (0.0, *context.attack_proportions)
    for sweep, settings in _sweep_settings(sigma).items():
        for parameters in settings:
            for repeat in range(context.sweep_repeats):
                key_label = f"rq5-{sweep}-{parameters['setting_label']}-repeat-{repeat}"
                watermarked, key, categories = _embed_bundle(
                    context,
                    bundle,
                    key_label=key_label,
                    n=int(parameters["n"]),
                    p=float(parameters["p"]),
                    gamma=float(parameters["gamma"]),
                    persist=False,
                )
                for proportion in proportions:
                    attacked = _numeric_attack(
                        watermarked,
                        bundle.watermark_column,
                        proportion,
                        float(parameters["attack_range"]),
                        stable_seed(
                            context.seed,
                            "rq5-attack",
                            sweep,
                            parameters["setting_label"],
                            repeat,
                            proportion,
                        ),
                    )
                    details = _detect_bundle(
                        bundle,
                        attacked,
                        key,
                        categories,
                        p=float(parameters["p"]),
                    )
                    result = _evaluate_model(
                        context,
                        bundle,
                        attacked,
                        train_idx,
                        test_idx,
                        seed=stable_seed(
                            context.seed,
                            "rq5-model",
                            sweep,
                            parameters["setting_label"],
                            repeat,
                            proportion,
                        ),
                        gpu_label=("rq5", sweep, parameters["setting_label"]),
                    )
                    rows.append(
                        {
                            "dataset": "synthetic",
                            "sweep": sweep,
                            "setting": parameters["setting"],
                            "setting_label": parameters["setting_label"],
                            "p": parameters["p"],
                            "n": parameters["n"],
                            "gamma": parameters["gamma"],
                            **_gamma_fields(details),
                            "attack_range": parameters["attack_range"],
                            "attack_proportion": proportion,
                            "repeat": repeat,
                            "green_count": details.green_count,
                            "z_score": details.z_score,
                            "threshold": threshold,
                            "detected": bool(details.z_score > threshold),
                            "accuracy": result.metrics["accuracy"],
                            "macro_f1": result.metrics["macro_f1"],
                            "model": result.model,
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = []
    for keys, group in frame.groupby(
        ["sweep", "setting_label", "attack_proportion"], sort=False
    ):
        summary.append(
            {
                "sweep": keys[0],
                "setting_label": keys[1],
                "attack_proportion": keys[2],
                "nominal_gamma": float(group["nominal_gamma"].iloc[0]),
                "effective_gamma": float(group["effective_gamma"].iloc[0]),
                "z_mean": float(group["z_score"].mean()),
                "z_sd": sample_sd(group["z_score"]),
                "accuracy_mean": float(group["accuracy"].mean()),
                "accuracy_sd": sample_sd(group["accuracy"]),
                "detection_rate": float(group["detected"].mean()),
            }
        )
    return _write_stage(
        context,
        "rq5_sweeps",
        rows,
        {
            "rq": "RQ5",
            "threshold": threshold,
            "gamma_semantics": GAMMA_SEMANTICS,
            "dataset": bundle.metadata(),
            "summary": summary,
            "protocol": {
                "sigma": sigma,
                "p": "[0.5,1,1.5,2,2.5]*sigma; n=300; gamma=.5; attack range=2*sigma",
                "n": "[100,200,300,400,500]; p=2*sigma; gamma=.5; attack range=2*sigma",
                "gamma": "[.25,.33,.5,.67,.75]; n=300; p=1.5*sigma; attack range=1.5*sigma",
                "attack_proportions": list(proportions),
                "utility": "logistic regression trained on attacked/watermarked features and clean labels; shared clean holdout",
            },
        },
    )


STAGES = {
    "rq1_detectability": run_rq1_detectability,
    "rq1_roc": run_rq1_roc,
    "rq2_utility": run_rq2_utility,
    "rq3_alteration": run_rq3_alteration,
    "rq3_row_attacks": run_rq3_row_attacks,
    "rq3_iris_attacks": run_rq3_iris_attacks,
    "rq5_sweeps": run_rq5_sweeps,
}
