"""TabularMark embedding, detection, matching, and deterministic I/O.

Independent reproduction code following the TabularMark paper protocol.
All key-cell edits accumulate in one deep copy. Original upstream notebooks,
datasets and historic experiment artifacts are not bundled; see ATTRIBUTION.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WatermarkKey:
    """Secret row positions and independent domain-partition seeds."""

    indices: tuple[int, ...]
    divide_seeds: tuple[int, ...]
    gamma: float = 0.5

    @property
    def n(self) -> int:
        return len(self.indices)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WatermarkKey":
        return cls(
            indices=tuple(int(x) for x in value["indices"]),
            divide_seeds=tuple(int(x) for x in value["divide_seeds"]),
            gamma=float(value.get("gamma", 0.5)),
        )


@dataclass(frozen=True)
class DetectionDetails:
    """Detection result with the null probability used for the z-score."""

    green_count: int
    z_score: float
    nominal_gamma: float
    effective_gamma: float


@dataclass(frozen=True)
class MatchedDetectionDetails:
    """Matched categorical detection result with explicit gamma semantics."""

    green_count: int
    matched_count: int
    z_score: float
    nominal_gamma: float
    effective_gamma: float


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a process-independent uint32 seed from a base seed and labels."""

    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def make_key(row_count: int, n: int, seed: int, gamma: float = 0.5) -> WatermarkKey:
    if not 0 < n <= row_count:
        raise ValueError(f"n must be in [1, {row_count}], got {n}")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be strictly between 0 and 1, got {gamma}")
    rng = np.random.default_rng(seed)
    divide_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n, dtype=np.uint64)
    indices = rng.choice(row_count, size=n, replace=False)
    return WatermarkKey(
        indices=tuple(int(x) for x in indices),
        divide_seeds=tuple(int(x) for x in divide_seeds),
        gamma=float(gamma),
    )


def z_statistic(green_count: int, n: int, gamma: float = 0.5) -> float:
    if n < 1:
        raise ValueError("n must be positive")
    if not 0 <= green_count <= n:
        raise ValueError(f"green_count must be in [0, {n}], got {green_count}")
    denominator = math.sqrt(n * gamma * (1.0 - gamma))
    if denominator == 0:
        raise ValueError("z statistic is undefined for gamma in {0, 1}")
    return (green_count - n * gamma) / denominator


def decision_threshold(alpha: float = 0.05) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def _green_count_for_k(k: int, gamma: float) -> int:
    count = int(round(k * gamma))
    if not 0 < count < k:
        raise ValueError(f"gamma={gamma} and k={k} create an empty domain")
    return count


def _categorical_green_count_for_k(k: int, gamma: float) -> int:
    # The released categorical notebooks partition with ``len(values) // 2``
    # for gamma=.5.  ``floor`` preserves that protocol for odd cardinalities;
    # the resulting null probability is reported separately as effective_gamma.
    count = int(math.floor(k * gamma))
    if not 0 < count < k:
        raise ValueError(f"gamma={gamma} and k={k} create an empty domain")
    return count


def _numeric_green_intervals(
    divide_seed: int, p: float, k: int, gamma: float
) -> tuple[np.ndarray, np.random.Generator]:
    if p <= 0.0 or k < 2:
        raise ValueError("p must be positive and k must be at least two")
    rng = np.random.default_rng(divide_seed)
    permutation = rng.permutation(k)
    return permutation[: _green_count_for_k(k, gamma)], rng


def _is_in_numeric_green_domain(
    deviation: float, green_interval_ids: np.ndarray, p: float, k: int
) -> bool:
    if not np.isfinite(deviation) or deviation < -p or deviation >= p:
        return False
    width = 2.0 * p / k
    interval_id = int(math.floor((deviation + p) / width))
    interval_id = min(max(interval_id, 0), k - 1)
    return bool(np.any(green_interval_ids == interval_id))


def embed_numeric(
    original: pd.DataFrame,
    column: str,
    key: WatermarkKey,
    p: float,
    k: int,
) -> pd.DataFrame:
    if column not in original.columns:
        raise KeyError(f"missing numerical watermark column: {column}")
    watermarked = original.copy(deep=True)
    width = 2.0 * p / k
    for idx, divide_seed in zip(key.indices, key.divide_seeds):
        green_ids, rng = _numeric_green_intervals(divide_seed, p, k, key.gamma)
        lows = -p + green_ids.astype(float) * width
        highs = lows + width
        candidates = rng.uniform(lows, np.nextafter(highs, lows))
        deviation = float(rng.choice(candidates))
        watermarked.at[idx, column] = float(original.at[idx, column]) + deviation
    return watermarked


def detect_numeric(
    original: pd.DataFrame,
    suspect: pd.DataFrame,
    column: str,
    key: WatermarkKey,
    p: float,
    k: int,
) -> tuple[int, float]:
    if len(original) != len(suspect):
        raise ValueError(
            "original and suspect numerical datasets must align by row; "
            "use matched detection for insertion/deletion attacks"
        )
    green_count = 0
    for idx, divide_seed in zip(key.indices, key.divide_seeds):
        green_ids, _ = _numeric_green_intervals(divide_seed, p, k, key.gamma)
        deviation = float(suspect.at[idx, column]) - float(original.at[idx, column])
        green_count += int(_is_in_numeric_green_domain(deviation, green_ids, p, k))
    return green_count, z_statistic(green_count, key.n, key.gamma)


def _categorical_green_domain(
    categories: np.ndarray, divide_seed: int, gamma: float
) -> tuple[np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(divide_seed)
    shuffled = rng.permutation(categories)
    return shuffled[: _categorical_green_count_for_k(len(categories), gamma)], rng


def categorical_effective_gamma(categories: Sequence[Any], nominal_gamma: float) -> float:
    """Return the attainable green-domain share for categorical values."""

    category_count = len(categories)
    return (
        _categorical_green_count_for_k(category_count, nominal_gamma)
        / category_count
    )


def embed_categorical(
    original: pd.DataFrame,
    column: str,
    key: WatermarkKey,
    categories: Sequence[Any] | None = None,
) -> pd.DataFrame:
    if column not in original.columns:
        raise KeyError(f"missing categorical watermark column: {column}")
    values = np.asarray(
        sorted(pd.unique(original[column]).tolist(), key=lambda value: str(value))
        if categories is None
        else list(categories)
    )
    if len(values) < 2:
        raise ValueError("categorical watermarking requires at least two categories")
    watermarked = original.copy(deep=True)
    for idx, divide_seed in zip(key.indices, key.divide_seeds):
        green, rng = _categorical_green_domain(values, divide_seed, key.gamma)
        watermarked.at[idx, column] = rng.choice(green)
    return watermarked


def detect_categorical(
    suspect: pd.DataFrame,
    column: str,
    key: WatermarkKey,
    categories: Sequence[Any],
) -> tuple[int, float]:
    details = detect_categorical_details(suspect, column, key, categories)
    return details.green_count, details.z_score


def detect_categorical_details(
    suspect: pd.DataFrame,
    column: str,
    key: WatermarkKey,
    categories: Sequence[Any],
) -> DetectionDetails:
    if column not in suspect.columns:
        raise KeyError(f"missing categorical watermark column: {column}")
    values = np.asarray(list(categories))
    effective_gamma = categorical_effective_gamma(values, key.gamma)
    green_count = 0
    for idx, divide_seed in zip(key.indices, key.divide_seeds):
        green, _ = _categorical_green_domain(values, divide_seed, key.gamma)
        green_count += int(np.any(green == suspect.at[idx, column]))
    return DetectionDetails(
        green_count=green_count,
        z_score=z_statistic(green_count, key.n, effective_gamma),
        nominal_gamma=key.gamma,
        effective_gamma=effective_gamma,
    )


def _match_value(value: Any) -> tuple[str, Any]:
    if pd.isna(value):
        return ("nan", None)
    if isinstance(value, (np.integer, int)):
        return ("int", int(value))
    if isinstance(value, (np.floating, float)):
        return ("float", float(value))
    return ("str", str(value))


def row_signature(row: pd.Series, columns: Sequence[str]) -> tuple[tuple[str, Any], ...]:
    return tuple(_match_value(row[column]) for column in columns)


def match_key_rows(
    original: pd.DataFrame,
    suspect: pd.DataFrame,
    original_indices: Sequence[int],
    columns: Sequence[str],
) -> list[int | None]:
    """Match key rows without indexing every suspect row in Python.

    Exact values are the executable protocol used by the released notebooks
    (e.g. Elevation/Aspect).  The paper describes these as selected MSBs; the
    stage metadata records the exact columns and this implementation choice.

    Only hashes of the requested key signatures are materialized as a Python
    set.  Pandas hashes the suspect columns vectorially, after which only rows
    that can match a key enter the stable occurrence-ranked merge.  For a
    duplicated signature, key requests and suspect rows are paired in their
    respective input orders, preserving the former deque-based one-to-one
    semantics while avoiding a Python dictionary with up to millions of rows.
    """

    if not columns:
        raise ValueError("at least one matching column is required")
    if len(set(columns)) != len(columns):
        raise ValueError(f"matching columns contain duplicates: {list(columns)}")
    absent = set(columns).difference(original.columns) | set(columns).difference(
        suspect.columns
    )
    if absent:
        raise KeyError(f"matching columns absent from a dataset: {sorted(absent)}")
    if len(original_indices) == 0:
        return []

    column_list = list(columns)
    key_rows = original.loc[[int(index) for index in original_indices], column_list]
    key_rows = key_rows.reset_index(drop=True)
    key_hashes = pd.util.hash_pandas_object(key_rows, index=False)
    key_hash_set = set(key_hashes.to_numpy(dtype=np.uint64).tolist())

    suspect_values = suspect.loc[:, column_list]
    suspect_hashes = pd.util.hash_pandas_object(suspect_values, index=False)
    candidate_mask = suspect_hashes.isin(key_hash_set).to_numpy(dtype=bool)
    if not bool(candidate_mask.any()):
        return [None] * len(original_indices)

    candidate_positions = np.flatnonzero(candidate_mask)
    candidates = suspect_values.iloc[candidate_positions].reset_index(drop=True).copy()

    reserved = set(original.columns) | set(suspect.columns)

    def unique_name(base: str) -> str:
        name = base
        suffix = 0
        while name in reserved:
            suffix += 1
            name = f"{base}_{suffix}"
        reserved.add(name)
        return name

    occurrence_column = unique_name("__tabularmark_occurrence")
    key_order_column = unique_name("__tabularmark_key_order")
    suspect_index_column = unique_name("__tabularmark_suspect_index")
    present_column = unique_name("__tabularmark_present")

    # ``sort=False`` retains input order within every duplicate-signature group.
    key_rows[key_order_column] = np.arange(len(key_rows), dtype=np.int64)
    key_rows[occurrence_column] = key_rows.groupby(
        column_list, sort=False, dropna=False
    ).cumcount()
    candidates[suspect_index_column] = np.asarray(suspect.index, dtype=object)[
        candidate_positions
    ]
    candidates[present_column] = True
    candidates[occurrence_column] = candidates.groupby(
        column_list, sort=False, dropna=False
    ).cumcount()

    joined = key_rows.merge(
        candidates[
            column_list
            + [occurrence_column, suspect_index_column, present_column]
        ],
        how="left",
        on=column_list + [occurrence_column],
        sort=False,
        validate="one_to_one",
    ).sort_values(key_order_column, kind="stable")
    return [
        int(suspect_index) if present is True else None
        for suspect_index, present in zip(
            joined[suspect_index_column], joined[present_column]
        )
    ]


def detect_categorical_matched(
    suspect: pd.DataFrame,
    column: str,
    key: WatermarkKey,
    categories: Sequence[Any],
    matches: Sequence[int | None],
) -> tuple[int, int, float]:
    details = detect_categorical_matched_details(
        suspect, column, key, categories, matches
    )
    return details.green_count, details.matched_count, details.z_score


def detect_categorical_matched_details(
    suspect: pd.DataFrame,
    column: str,
    key: WatermarkKey,
    categories: Sequence[Any],
    matches: Sequence[int | None],
) -> MatchedDetectionDetails:
    if len(matches) != key.n:
        raise ValueError("matches must contain exactly one entry per key cell")
    values = np.asarray(list(categories))
    effective_gamma = categorical_effective_gamma(values, key.gamma)
    green_count = 0
    matched_count = 0
    for suspect_idx, divide_seed in zip(matches, key.divide_seeds):
        if suspect_idx is None:
            continue
        matched_count += 1
        green, _ = _categorical_green_domain(values, divide_seed, key.gamma)
        green_count += int(np.any(green == suspect.at[suspect_idx, column]))
    return MatchedDetectionDetails(
        green_count=green_count,
        matched_count=matched_count,
        z_score=z_statistic(green_count, key.n, effective_gamma),
        nominal_gamma=key.gamma,
        effective_gamma=effective_gamma,
    )


def exact_binomial_false_positive_rate(n: int, gamma: float, threshold: float) -> float:
    return float(
        sum(
            math.comb(n, green_count)
            * gamma**green_count
            * (1.0 - gamma) ** (n - green_count)
            for green_count in range(n + 1)
            if z_statistic(green_count, n, gamma) > threshold
        )
    )


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_private_json(path: Path, value: Any) -> None:
    """Atomically write secret JSON and enforce owner read/write permissions."""

    atomic_json(path, value)
    path.chmod(0o600)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=json_default, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
