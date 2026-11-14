"""Deterministic, paper-aligned RQ4 comparison primitives.

This module is an independent experimental extension. It reimplements the
three Covertype watermark paths after auditing the public notebooks.  The
implementation is not claimed to be a byte-for-byte replay of their saved
outputs; corrections are documented in the extension README and audit report.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_COLUMN = "Cover_Type"
PRIMARY_KEY_COLUMN = "primary_key"

COVERTYPE_FEATURE_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
] + [f"Wilderness_Area{i}" for i in range(1, 5)] + [f"Soil_Type{i}" for i in range(1, 41)]
COVERTYPE_COLUMNS = COVERTYPE_FEATURE_COLUMNS + [TARGET_COLUMN]


@dataclass
class WatermarkBundle:
    """A watermarked table and the secret state needed for detection."""

    name: str
    data: pd.DataFrame
    state: Dict[str, Any]


def _stable_seed(base_seed: int, namespace: str) -> int:
    payload = f"tabularmark-rq4-v1\x00{int(base_seed)}\x00{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _blake2b_hex(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8")).hexdigest()


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_ready(value: Any) -> Any:
    """Recursively convert NumPy values into strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    return _json_scalar(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_covertype(
    path: Path,
    input_format: str = "auto",
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Load either UCI's 55-column headerless file or a headered CSV.

    ``auto`` treats the file as UCI raw data only when the first non-empty row
    has 55 fully numeric fields.  This avoids silently consuming the first data
    record as a header, a common failure mode for ``covtype.data``.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Covertype input does not exist: {path}")
    if input_format not in {"auto", "uci-raw", "headered"}:
        raise ValueError("input_format must be one of: auto, uci-raw, headered")

    resolved_format = input_format
    if input_format == "auto":
        first_line = ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    first_line = line.strip()
                    break
        fields = [part.strip() for part in first_line.split(",")]
        numeric = len(fields) == len(COVERTYPE_COLUMNS)
        if numeric:
            try:
                [float(part) for part in fields]
            except ValueError:
                numeric = False
        resolved_format = "uci-raw" if numeric else "headered"

    if resolved_format == "uci-raw":
        data = pd.read_csv(path, header=None, names=COVERTYPE_COLUMNS)
    else:
        data = pd.read_csv(path)

    if target_column not in data.columns:
        raise ValueError(
            f"Target column {target_column!r} is absent. Columns start with: "
            f"{list(data.columns[:8])!r}"
        )
    if data.columns.duplicated().any():
        duplicates = sorted(set(data.columns[data.columns.duplicated()].tolist()))
        raise ValueError(f"Duplicate columns are not supported: {duplicates}")
    if data[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values")

    numeric_target = pd.to_numeric(data[target_column], errors="raise")
    if not np.allclose(numeric_target.to_numpy(), np.rint(numeric_target.to_numpy())):
        raise ValueError(f"Target column {target_column!r} must contain integer class labels")
    data = data.copy()
    data[target_column] = np.rint(numeric_target).astype(np.int64)
    classes = sorted(int(value) for value in data[target_column].unique())
    if classes != list(range(1, 8)):
        raise ValueError(f"Expected Covertype classes 1..7, observed {classes}")
    return data


def prepare_balanced_subset(
    data: pd.DataFrame,
    per_class: int = 2000,
    seed: int = 2024,
    target_column: str = TARGET_COLUMN,
    strict: bool = True,
) -> pd.DataFrame:
    """Select a deterministic, globally shuffled, class-balanced subset."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    parts: List[pd.DataFrame] = []
    counts = data[target_column].value_counts().sort_index()
    for label in sorted(int(value) for value in counts.index):
        group = data.loc[data[target_column] == label]
        if len(group) < per_class and strict:
            raise ValueError(
                f"Class {label} has {len(group)} rows, fewer than requested per_class={per_class}. "
                "Use a smaller value only for an explicitly labelled smoke test."
            )
        take = min(per_class, len(group))
        rng = np.random.Generator(np.random.PCG64(_stable_seed(seed, f"subset-class-{label}")))
        positions = np.sort(rng.choice(len(group), size=take, replace=False))
        parts.append(group.iloc[positions])

    subset = pd.concat(parts, axis=0, ignore_index=True)
    shuffle_rng = np.random.Generator(np.random.PCG64(_stable_seed(seed, "subset-global-shuffle")))
    subset = subset.iloc[shuffle_rng.permutation(len(subset))].reset_index(drop=True).copy()
    if PRIMARY_KEY_COLUMN in subset.columns:
        subset = subset.drop(columns=[PRIMARY_KEY_COLUMN])
    subset.insert(0, PRIMARY_KEY_COLUMN, np.arange(len(subset), dtype=np.int64))
    return subset


def _balanced_bits(length: int, seed: int) -> List[int]:
    if length <= 0:
        raise ValueError("watermark length must be positive")
    bits = np.asarray(([0] * ((length + 1) // 2)) + ([1] * (length // 2)), dtype=np.int8)
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(bits)
    return [int(value) for value in bits]


def embed_tabularmark(
    data: pd.DataFrame,
    n_key_cells: int,
    seed: int,
    gamma: float = 0.5,
    target_column: str = TARGET_COLUMN,
) -> WatermarkBundle:
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must lie strictly between 0 and 1")
    if n_key_cells <= 0 or n_key_cells > len(data):
        raise ValueError("n_key_cells must be in 1..len(data)")

    classes = np.asarray(sorted(int(value) for value in data[target_column].unique()), dtype=np.int64)
    green_size = int(math.floor(len(classes) * gamma))
    if green_size <= 0:
        raise ValueError("gamma produces an empty green domain")

    rng = np.random.Generator(np.random.PCG64(_stable_seed(seed, "tabularmark-key-cells")))
    positions = rng.choice(len(data), size=n_key_cells, replace=False)
    cell_seeds = rng.integers(0, 2**32, size=n_key_cells, dtype=np.uint64)
    out = data.copy(deep=True)
    green_domains: List[List[int]] = []
    target_location = out.columns.get_loc(target_column)

    for position, cell_seed in zip(positions, cell_seeds):
        local_rng = np.random.Generator(np.random.PCG64(int(cell_seed)))
        shuffled = local_rng.permutation(classes)
        green = shuffled[:green_size]
        replacement = int(green[int(local_rng.integers(0, len(green)))])
        out.iloc[int(position), target_location] = replacement
        green_domains.append([int(value) for value in green])

    state: Dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "TabularMark",
        "target_column": target_column,
        "classes": [int(value) for value in classes],
        "n_key_cells": int(n_key_cells),
        "nominal_gamma": float(gamma),
        "effective_gamma": float(green_size / len(classes)),
        "key_positions": [int(value) for value in positions],
        "cell_seeds": [int(value) for value in cell_seeds],
        "green_domains": green_domains,
    }
    return WatermarkBundle("TabularMark", out, state)


def detect_tabularmark(data: pd.DataFrame, state: Mapping[str, Any]) -> Dict[str, Any]:
    positions = [int(value) for value in state["key_positions"]]
    green_domains = state["green_domains"]
    target_column = str(state["target_column"])
    if len(data) <= max(positions):
        raise ValueError("Detected table is shorter than the largest stored key position")
    target_location = data.columns.get_loc(target_column)
    green_count = sum(
        int(data.iloc[position, target_location]) in set(int(value) for value in green)
        for position, green in zip(positions, green_domains)
    )
    n = len(positions)
    nominal_gamma = float(state["nominal_gamma"])
    effective_gamma = float(state["effective_gamma"])

    def z_score(probability: float) -> float:
        denominator = math.sqrt(n * probability * (1.0 - probability))
        return (green_count - n * probability) / denominator

    return {
        "mismatch_percentage": float((n - green_count) / n),
        "green_count": int(green_count),
        "n_key_cells": int(n),
        "z_score_nominal": float(z_score(nominal_gamma)),
        "z_score_effective": float(z_score(effective_gamma)),
    }


def _hist_group(secret: str, primary_key: int, groups: int) -> int:
    """Use the intended nested hash, not the notebook's reused-hasher bug."""

    inner = _sha256_hex(f"{secret}{primary_key}")
    outer = _sha256_hex(f"{secret}{inner}")
    return int(outer, 16) % groups


def embed_histmark(
    data: pd.DataFrame,
    watermark_length: int,
    seed: int,
    target_column: str = TARGET_COLUMN,
    primary_key_column: str = PRIMARY_KEY_COLUMN,
) -> WatermarkBundle:
    if primary_key_column not in data.columns:
        raise ValueError(f"HistMark requires {primary_key_column!r}")
    classes = sorted(int(value) for value in data[target_column].unique())
    if len(classes) < 3:
        raise ValueError("HistMark requires at least three target values")

    secret = hashlib.sha256(f"{seed}:histmark:secret".encode("utf-8")).hexdigest()[:16]
    bits = _balanced_bits(watermark_length, _stable_seed(seed, "histmark-watermark-bits"))
    out = data.copy(deep=True)
    out["group_number"] = [
        _hist_group(secret, int(primary_key), watermark_length)
        for primary_key in out[primary_key_column].to_numpy()
    ]
    minimum = int(min(classes))
    maximum = int(max(classes))
    y_hat = (minimum + maximum) / 2.0
    excluded_primary_keys = [
        int(value)
        for value in out.loc[out[target_column].isin([minimum, maximum]), primary_key_column].tolist()
    ]
    peaks: List[Optional[float]] = []

    for group_number, bit in enumerate(bits):
        group_mask = out["group_number"] == group_number
        eligible_mask = group_mask & ~out[target_column].isin([minimum, maximum])
        deviations = np.abs(out.loc[eligible_mask, target_column].to_numpy(dtype=float) - y_hat)
        if deviations.size == 0:
            peaks.append(None)
            continue
        values, counts = np.unique(deviations, return_counts=True)
        peak = float(np.min(values[counts == counts.max()]))
        peaks.append(peak)

        positions = np.flatnonzero(eligible_mask.to_numpy())
        pe = out.loc[eligible_mask, target_column].to_numpy(dtype=float) - y_hat
        transformed = pe.copy()
        positive_peak = np.isclose(pe, peak)
        negative_peak = np.isclose(pe, -peak)
        if bit == 1:
            transformed[positive_peak] = pe[positive_peak] + 1.0
            transformed[negative_peak & ~positive_peak] = pe[negative_peak & ~positive_peak] - 1.0
        high_tail = (pe >= peak + 1.0) & ~positive_peak & ~negative_peak
        low_tail = (pe <= -(peak + 1.0)) & ~positive_peak & ~negative_peak
        transformed[high_tail] = pe[high_tail] + 1.0
        transformed[low_tail] = pe[low_tail] - 1.0
        new_values = transformed + y_hat
        if np.allclose(new_values, np.rint(new_values)):
            new_values = np.rint(new_values).astype(np.int64)
        out.iloc[positions, out.columns.get_loc(target_column)] = new_values

    state = {
        "schema_version": 1,
        "algorithm": "HistMark",
        "target_column": target_column,
        "primary_key_column": primary_key_column,
        "classes": classes,
        "watermark_bits": bits,
        "watermark_length": int(watermark_length),
        "secret": secret,
        "y_hat": float(y_hat),
        "peaks": peaks,
        "excluded_primary_keys": excluded_primary_keys,
        "decoder": "symmetric-absolute-peak",
    }
    return WatermarkBundle("HistMark", out, state)


def detect_histmark(data: pd.DataFrame, state: Mapping[str, Any]) -> Dict[str, Any]:
    target_column = str(state["target_column"])
    primary_key_column = str(state["primary_key_column"])
    bits = [int(value) for value in state["watermark_bits"]]
    excluded = set(int(value) for value in state["excluded_primary_keys"])
    y_hat = float(state["y_hat"])
    detected: List[int] = []
    undecidable = 0
    evidence_count = 0
    excluded_mask = data[primary_key_column].isin(excluded)

    for group_number, peak_value in enumerate(state["peaks"]):
        if peak_value is None:
            detected.append(1)
            undecidable += 1
            continue
        peak = float(peak_value)
        group = data.loc[(data["group_number"] == group_number) & ~excluded_mask]
        absolute_deviation = np.abs(group[target_column].to_numpy(dtype=float) - y_hat)
        zero_votes = int(np.isclose(absolute_deviation, peak).sum())
        one_votes = int(np.isclose(absolute_deviation, peak + 1.0).sum())
        evidence_count += zero_votes + one_votes
        if zero_votes == one_votes:
            undecidable += 1
        detected.append(0 if zero_votes > one_votes else 1)

    mismatches = sum(expected != observed for expected, observed in zip(bits, detected))
    return {
        "mismatch_percentage": float(mismatches / len(bits)),
        "decoded_bits_with_evidence": int(evidence_count),
        "undecidable_bits": int(undecidable),
        "watermark_length": int(len(bits)),
    }


def _semantic_map(value: int) -> int:
    # This is the exact Cover_Type adaptation used by semanticmark.ipynb.
    return int(math.floor((int(value) + 5) / 2))


def embed_semmark(
    data: pd.DataFrame,
    watermark_length: int,
    votes_per_bit: int,
    seed: int,
    target_column: str = TARGET_COLUMN,
    primary_key_column: str = PRIMARY_KEY_COLUMN,
) -> WatermarkBundle:
    if votes_per_bit <= 0:
        raise ValueError("votes_per_bit must be positive")
    if primary_key_column not in data.columns:
        raise ValueError(f"SemMark requires {primary_key_column!r}")

    classes = sorted(int(value) for value in data[target_column].unique())
    bits = _balanced_bits(watermark_length, _stable_seed(seed, "semmark-watermark-bits"))
    secret_master = hashlib.sha256(f"{seed}:semmark:Ks".encode("utf-8")).hexdigest()[:16]
    secret_attribute = hashlib.sha256(f"{seed}:semmark:k".encode("utf-8")).hexdigest()[:16]
    desired_selected = watermark_length * votes_per_bit
    selection_modulus = max(1, int(round(len(data) / desired_selected)))
    out = data.copy(deep=True)
    record_table: Dict[str, str] = {}
    selected_primary_keys: List[int] = []

    for position, row in out.iterrows():
        primary_key = int(row[primary_key_column])
        key_hash = _blake2b_hex(f"{primary_key}{secret_attribute}")
        master_hash = _blake2b_hex(f"{secret_master}{key_hash}")
        if int(master_hash, 16) % selection_modulus != 0:
            continue
        zeta = int(_blake2b_hex(f"{secret_master}{key_hash}{1}"), 16) % watermark_length
        original = int(row[target_column])
        selected_primary_keys.append(primary_key)
        if bits[zeta] == 1:
            replacement = _semantic_map(original)
            out.at[position, target_column] = replacement
            record_table[key_hash] = _blake2b_hex(
                f"{replacement}{secret_attribute}{abs(original - replacement)}"
            )
        else:
            record_table[key_hash] = _blake2b_hex(f"{original}{secret_attribute}")

    state = {
        "schema_version": 1,
        "algorithm": "SemMark",
        "target_column": target_column,
        "primary_key_column": primary_key_column,
        "classes": classes,
        "watermark_bits": bits,
        "watermark_length": int(watermark_length),
        "votes_per_bit_requested": int(votes_per_bit),
        "selection_modulus": int(selection_modulus),
        "secret_master": secret_master,
        "secret_attribute": secret_attribute,
        "selected_primary_keys": selected_primary_keys,
        "record_table": record_table,
    }
    return WatermarkBundle("SemMark", out, state)


def detect_semmark(data: pd.DataFrame, state: Mapping[str, Any]) -> Dict[str, Any]:
    target_column = str(state["target_column"])
    primary_key_column = str(state["primary_key_column"])
    secret_master = str(state["secret_master"])
    secret_attribute = str(state["secret_attribute"])
    record_table = {str(key): str(value) for key, value in state["record_table"].items()}
    selected = set(int(value) for value in state["selected_primary_keys"])
    classes = [int(value) for value in state["classes"]]
    bits = [int(value) for value in state["watermark_bits"]]
    votes = [[0, 0] for _ in bits]

    for _, row in data.loc[data[primary_key_column].isin(selected)].iterrows():
        primary_key = int(row[primary_key_column])
        observed_value = int(row[target_column])
        key_hash = _blake2b_hex(f"{primary_key}{secret_attribute}")
        expected_hash = record_table.get(key_hash)
        if expected_hash is None:
            continue
        zeta = int(_blake2b_hex(f"{secret_master}{key_hash}{1}"), 16) % len(bits)
        if expected_hash == _blake2b_hex(f"{observed_value}{secret_attribute}"):
            votes[zeta][0] += 1
            continue
        for original in classes:
            replacement = _semantic_map(original)
            if replacement != observed_value:
                continue
            candidate = _blake2b_hex(
                f"{observed_value}{secret_attribute}{abs(original - observed_value)}"
            )
            if expected_hash == candidate:
                votes[zeta][1] += 1
                break

    detected: List[int] = []
    undecidable = 0
    evidence_count = 0
    for zero_votes, one_votes in votes:
        evidence_count += zero_votes + one_votes
        if zero_votes == one_votes:
            undecidable += 1
        detected.append(0 if zero_votes > one_votes else 1)
    mismatches = sum(expected != observed for expected, observed in zip(bits, detected))
    return {
        "mismatch_percentage": float(mismatches / len(bits)),
        "decoded_bits_with_evidence": int(evidence_count),
        "undecidable_bits": int(undecidable),
        "watermark_length": int(len(bits)),
    }


def build_rq4_watermarks(
    data: pd.DataFrame,
    seed: int = 2024,
    n_key_cells: int = 400,
    watermark_length: int = 400,
    semmark_votes_per_bit: int = 4,
    target_column: str = TARGET_COLUMN,
) -> Dict[str, WatermarkBundle]:
    return {
        "TabularMark": embed_tabularmark(data, n_key_cells, seed, 0.5, target_column),
        "HistMark": embed_histmark(data, watermark_length, seed, target_column),
        "SemMark": embed_semmark(
            data,
            watermark_length,
            semmark_votes_per_bit,
            seed,
            target_column,
        ),
    }


def build_alteration_plan(
    row_count: int,
    classes: Sequence[int],
    seed: int,
    repeat: int,
) -> Dict[str, np.ndarray]:
    """Create one nested attack plan shared by all watermark schemes."""

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    labels = np.asarray(sorted(int(value) for value in classes), dtype=np.int64)
    rng = np.random.Generator(
        np.random.PCG64(_stable_seed(seed, f"alteration-plan-repeat-{int(repeat)}"))
    )
    return {
        "row_order": rng.permutation(row_count).astype(np.int64),
        "replacement_labels": rng.choice(labels, size=row_count, replace=True).astype(np.int64),
    }


def apply_alteration(
    data: pd.DataFrame,
    fraction: float,
    plan: Mapping[str, np.ndarray],
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must lie in [0, 1]")
    row_order = np.asarray(plan["row_order"], dtype=np.int64)
    replacements = np.asarray(plan["replacement_labels"], dtype=np.int64)
    if len(row_order) != len(data) or len(replacements) != len(data):
        raise ValueError("attack plan length does not match the table")
    count = int(math.floor(len(data) * fraction))
    out = data.copy(deep=True)
    if count:
        positions = row_order[:count]
        target_location = out.columns.get_loc(target_column)
        out.iloc[positions, target_location] = replacements[:count]
    return out


def _detect(bundle: WatermarkBundle, attacked: pd.DataFrame) -> Dict[str, Any]:
    if bundle.name == "TabularMark":
        return detect_tabularmark(attacked, bundle.state)
    if bundle.name == "HistMark":
        return detect_histmark(attacked, bundle.state)
    if bundle.name == "SemMark":
        return detect_semmark(attacked, bundle.state)
    raise ValueError(f"Unknown watermark scheme: {bundle.name}")


def run_alteration_comparison(
    bundles: Mapping[str, WatermarkBundle],
    fractions: Iterable[float],
    repeats: int,
    seed: int,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    ordered_fractions = sorted(set(float(value) for value in fractions))
    if not ordered_fractions or ordered_fractions[0] < 0.0 or ordered_fractions[-1] > 1.0:
        raise ValueError("fractions must be non-empty and lie in [0, 1]")
    first = next(iter(bundles.values()))
    classes = [int(value) for value in first.state["classes"]]
    rows: List[MutableMapping[str, Any]] = []

    for repeat in range(repeats):
        plan = build_alteration_plan(len(first.data), classes, seed, repeat)
        for fraction in ordered_fractions:
            for scheme in sorted(bundles):
                bundle = bundles[scheme]
                attacked = apply_alteration(bundle.data, fraction, plan, target_column)
                metrics: MutableMapping[str, Any] = {
                    "scheme": scheme,
                    "repeat": int(repeat),
                    "alteration_fraction": float(fraction),
                    "alteration_percent": int(round(fraction * 100)),
                }
                metrics.update(_detect(bundle, attacked))
                rows.append(metrics)

    repeats_frame = pd.DataFrame(rows).sort_values(
        ["scheme", "alteration_fraction", "repeat"], ignore_index=True
    )
    metric_columns = [
        column
        for column in [
            "mismatch_percentage",
            "green_count",
            "z_score_nominal",
            "z_score_effective",
            "decoded_bits_with_evidence",
            "undecidable_bits",
        ]
        if column in repeats_frame.columns
    ]
    summary = (
        repeats_frame.groupby(["scheme", "alteration_fraction", "alteration_percent"], sort=True)[
            metric_columns
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary["runs"] = repeats
    std_columns = [column for column in summary.columns if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return repeats_frame, summary


def xgboost_utility(
    original: pd.DataFrame,
    bundles: Mapping[str, WatermarkBundle],
    seed: int,
    device: str,
    n_estimators: int,
    test_size: float,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate all schemes on one shared split and one identical feature set."""

    try:
        import sklearn
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "ML evaluation requested but scikit-learn/xgboost is unavailable. "
            "Install requirements-ml.txt or pass --skip-ml explicitly."
        ) from exc

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must lie strictly between 0 and 1")
    feature_columns = [
        column
        for column in original.columns
        if column not in {target_column, PRIMARY_KEY_COLUMN, "group_number", "y_hat"}
    ]
    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(original[column])]
    if non_numeric:
        raise ValueError(f"XGBoost input features must be numeric; non-numeric columns: {non_numeric}")
    classes = sorted(int(value) for value in original[target_column].unique())
    class_to_zero = {label: index for index, label in enumerate(classes)}
    original_y = original[target_column].map(class_to_zero).to_numpy(dtype=np.int64)
    positions = np.arange(len(original), dtype=np.int64)
    train_positions, test_positions = train_test_split(
        positions,
        test_size=test_size,
        random_state=int(seed),
        stratify=original_y,
    )
    test_x = original.iloc[test_positions][feature_columns].to_numpy(dtype=np.float32)
    test_y = original_y[test_positions]

    major_version = int(str(xgb.__version__).split(".", 1)[0])
    requested_device = device.strip().lower()
    if requested_device == "gpu":
        requested_device = "cuda"
    if requested_device != "cpu" and not requested_device.startswith("cuda"):
        raise ValueError("device must be cpu, cuda, cuda:N, or gpu")

    common_params: Dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": "mlogloss",
        "random_state": int(seed),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "n_jobs": 1,
    }
    if major_version >= 2:
        common_params["device"] = requested_device
    elif requested_device.startswith("cuda"):
        common_params["tree_method"] = "gpu_hist"
        common_params["predictor"] = "gpu_predictor"
        if ":" in requested_device:
            common_params["gpu_id"] = int(requested_device.split(":", 1)[1])

    frames: Dict[str, pd.DataFrame] = {"Original": original}
    frames.update({name: bundle.data for name, bundle in bundles.items()})
    rows: List[Dict[str, Any]] = []
    observed_devices: Dict[str, str] = {}
    for scheme in sorted(frames):
        frame = frames[scheme]
        missing_features = sorted(set(feature_columns) - set(frame.columns))
        if missing_features:
            raise ValueError(f"{scheme} is missing utility features: {missing_features}")
        train_x = frame.iloc[train_positions][feature_columns].to_numpy(dtype=np.float32)
        train_y = frame.iloc[train_positions][target_column].map(class_to_zero)
        if train_y.isna().any():
            raise ValueError(f"{scheme} introduced labels outside the original class domain")
        model = xgb.XGBClassifier(**common_params)
        model.fit(train_x, train_y.to_numpy(dtype=np.int64))
        if major_version >= 2:
            try:
                booster_config = json.loads(model.get_booster().save_config())
                observed_device = str(
                    booster_config["learner"]["generic_param"]["device"]
                ).lower()
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{scheme}: cannot verify the device actually used by XGBoost"
                ) from exc
            observed_devices[scheme] = observed_device
            if requested_device.startswith("cuda"):
                if not observed_device.startswith("cuda"):
                    raise RuntimeError(
                        f"{scheme}: XGBoost silently fell back from {requested_device!r} "
                        f"to {observed_device!r}"
                    )
                if ":" in requested_device and observed_device != requested_device:
                    raise RuntimeError(
                        f"{scheme}: requested {requested_device!r}, but XGBoost reports "
                        f"{observed_device!r}"
                    )
            elif observed_device != "cpu":
                raise RuntimeError(
                    f"{scheme}: requested CPU, but XGBoost reports {observed_device!r}"
                )
        prediction = model.predict(test_x).astype(np.int64)
        report = classification_report(
            test_y,
            prediction,
            labels=list(range(len(classes))),
            target_names=[str(label) for label in classes],
            output_dict=True,
            zero_division=0,
        )
        for label in [str(value) for value in classes] + ["macro avg", "weighted avg"]:
            item = report[label]
            rows.append(
                {
                    "scheme": scheme,
                    "metric_scope": label,
                    "precision": float(item["precision"]),
                    "recall": float(item["recall"]),
                    "f1_score": float(item["f1-score"]),
                    "support": int(item["support"]),
                    "accuracy": float(report["accuracy"]),
                }
            )

    metadata = {
        "status": "COMPLETED",
        "xgboost_version": str(xgb.__version__),
        "scikit_learn_version": str(sklearn.__version__),
        "device_requested": device,
        "device_parameter": requested_device,
        "devices_observed_by_scheme": observed_devices,
        "n_estimators": int(n_estimators),
        "test_size": float(test_size),
        "seed": int(seed),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "train_rows": len(train_positions),
        "test_rows": len(test_positions),
        "determinism_note": (
            "Watermarking, split, and attack plans are deterministic. CPU XGBoost is configured "
            "single-threaded; CUDA reductions can still differ slightly across GPU/software stacks."
        ),
    }
    return pd.DataFrame(rows), metadata
