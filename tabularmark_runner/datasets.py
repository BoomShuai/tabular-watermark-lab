"""Dataset discovery with explicit schema resolution and provenance."""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml, load_digits, load_iris

from .core import atomic_csv, sha256_file, stable_seed


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    target: str
    watermark_kind: str
    watermark_column: str
    n: int
    p: float | None
    k: int
    source: str
    source_sha256: str | None
    feature_columns: list[str]
    match_columns: list[str]
    source_resolution: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "source_resolution": self.source_resolution,
            "shape": list(self.frame.shape),
            "columns": list(self.frame.columns),
            "target": self.target,
            "watermark_kind": self.watermark_kind,
            "watermark_column": self.watermark_column,
            "n": self.n,
            "p": self.p,
            "k": self.k,
            "feature_columns": self.feature_columns,
            "match_columns": self.match_columns,
        }


FILE_NAMES: dict[str, tuple[str, ...]] = {
    "synthetic": ("synthetic_data.csv", "synthetic.csv"),
    "covertype": (
        "cover_type_with_columns.csv",
        "covertype.csv",
        "covtype.csv",
        "covtype.data",
    ),
    "hog": ("digits_hog.csv", "hog.csv", "digits-hog.csv"),
    "housing": ("housingdata.csv", "housing.csv", "boston.csv"),
    "zoo": ("zoo.csv", "zoo.data"),
    "iris": ("iris.csv", "iris.data"),
}

TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "synthetic": ("target", "label", "class", "y"),
    "covertype": ("cover_type", "covertype", "target", "label", "class"),
    "hog": ("target", "category", "digit", "label", "class"),
    "housing": ("medv", "target", "price", "median_value", "median_house_value"),
    "zoo": ("type", "target", "label", "class", "class_type"),
    "iris": ("class", "species", "target", "label", "iris_class"),
}

ZOO_COLUMNS = [
    "animal_name",
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "legs",
    "tail",
    "domestic",
    "catsize",
    "type",
]

IRIS_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class",
]

COVTYPE_COLUMNS = [
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
    *(f"Wilderness_Area{i}" for i in range(1, 5)),
    *(f"Soil_Type{i}" for i in range(1, 41)),
    "Cover_Type",
]


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def resolve_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    dataset: str,
    purpose: str,
) -> str:
    by_normalised: dict[str, list[str]] = {}
    for column in frame.columns:
        by_normalised.setdefault(_normalise(column), []).append(str(column))
    for alias in aliases:
        matches = by_normalised.get(_normalise(alias), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"{dataset}: ambiguous {purpose} alias {alias!r}; matches={matches}"
            )
    raise ValueError(
        f"{dataset}: cannot identify {purpose}. Tried aliases={list(aliases)}; "
        f"available columns={list(frame.columns)}"
    )


def _candidate_files(data_dir: Path, name: str) -> list[Path]:
    wanted = {candidate.casefold() for candidate in FILE_NAMES[name]}
    excluded_parts = {
        "watermarked",
        "alteration",
        "insertion",
        "deletion",
        "perturbed",
        "roc",
        "results",
    }
    candidates = []
    if not data_dir.exists():
        return candidates
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.name.casefold() not in wanted:
            continue
        if any(part.casefold() in excluded_parts for part in path.parts):
            continue
        candidates.append(path)
    rank = {candidate.casefold(): index for index, candidate in enumerate(FILE_NAMES[name])}
    candidates.sort(key=lambda path: (rank.get(path.name.casefold(), 999), len(path.parts), str(path)))
    return candidates


def _select_candidate(
    candidates: Sequence[Path], name: str, *, scope: str
) -> tuple[Path, dict[str, Any]]:
    """Resolve the highest-priority candidate without hiding conflicting copies."""

    if not candidates:
        raise ValueError(f"{name}: no candidates supplied for {scope}")
    rank = {candidate.casefold(): index for index, candidate in enumerate(FILE_NAMES[name])}
    best_priority = min(rank.get(path.name.casefold(), 999) for path in candidates)
    peers = [
        path
        for path in candidates
        if rank.get(path.name.casefold(), 999) == best_priority
    ]
    peers.sort(key=lambda path: (len(path.parts), str(path.resolve())))
    peer_records = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in peers
    ]
    hashes = {record["sha256"] for record in peer_records}
    if len(hashes) != 1:
        details = ", ".join(
            f"{record['path']} (sha256={record['sha256']})"
            for record in peer_records
        )
        priority_name = (
            FILE_NAMES[name][best_priority]
            if best_priority < len(FILE_NAMES[name])
            else "unknown"
        )
        raise ValueError(
            f"{name}: conflicting {scope} candidates at the same filename priority "
            f"{best_priority} ({priority_name!r}): {details}"
        )

    selected = peers[0].resolve()
    ignored = [
        str(path.resolve())
        for path in candidates
        if rank.get(path.name.casefold(), 999) != best_priority
    ]
    return selected, {
        "scope": scope,
        "strategy": (
            "stable_path_among_identical_same_priority_candidates"
            if len(peers) > 1
            else "single_highest_priority_candidate"
        ),
        "filename_priority": best_priority,
        "priority_filename": (
            FILE_NAMES[name][best_priority]
            if best_priority < len(FILE_NAMES[name])
            else None
        ),
        "selected_path": str(selected),
        "selected_sha256": peer_records[0]["sha256"],
        "same_priority_candidates": peer_records,
        "ignored_lower_priority_candidates": sorted(ignored),
    }


def _read_candidate(path: Path, name: str) -> pd.DataFrame:
    if name == "covertype" and path.suffix.casefold() == ".data":
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] != len(COVTYPE_COLUMNS):
            raise ValueError(
                f"Covertype file {path} has {frame.shape[1]} columns; "
                f"expected {len(COVTYPE_COLUMNS)}"
            )
        frame.columns = COVTYPE_COLUMNS
        return frame
    if name == "zoo" and path.suffix.casefold() == ".data":
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] != len(ZOO_COLUMNS):
            raise ValueError(
                f"Zoo file {path} has {frame.shape[1]} columns; expected {len(ZOO_COLUMNS)}"
            )
        frame.columns = ZOO_COLUMNS
        return frame
    if name == "iris" and path.suffix.casefold() == ".data":
        frame = pd.read_csv(path, header=None).dropna(how="all")
        if frame.shape[1] != len(IRIS_COLUMNS):
            raise ValueError(
                f"Iris file {path} has {frame.shape[1]} columns; expected {len(IRIS_COLUMNS)}"
            )
        frame.columns = IRIS_COLUMNS
        return frame
    return pd.read_csv(path)


def _cache_frame(cache_dir: Path, name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    path = cache_dir / f"{name}.csv"
    atomic_csv(path, frame)
    return frame, path


def _generate_synthetic(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(stable_seed(seed, "synthetic-fallback"))
    X = rng.normal(loc=0.0, scale=20.0, size=(2000, 2))
    weight_rng = np.random.default_rng(42)
    weights = weight_rng.uniform(-1.0, 1.0, size=2)
    logits = np.clip(X @ weights, -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    target_rng = np.random.default_rng(42)
    target = target_rng.binomial(1, probabilities)
    return pd.DataFrame(
        {"dimension_0": X[:, 0], "dimension_1": X[:, 1], "target": target}
    )


def _fetch_frame(name: str, cache_dir: Path, seed: int) -> tuple[pd.DataFrame, Path, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if name == "synthetic":
        frame, path = _cache_frame(cache_dir, name, _generate_synthetic(seed))
        return frame, path, "generated:paper-protocol-normal-2000x2"
    if name == "iris":
        bunch = load_iris(as_frame=True)
        frame = bunch.frame.copy()
        frame = frame.rename(
            columns={
                "sepal length (cm)": "sepal_length",
                "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length",
                "petal width (cm)": "petal_width",
                "target": "class",
            }
        )
        frame, path = _cache_frame(cache_dir, name, frame)
        return frame, path, "sklearn.datasets.load_iris"
    if name == "covertype":
        bunch = fetch_covtype(as_frame=True, download_if_missing=True)
        frame = bunch.frame.copy()
        target_name = str(bunch.target.name or "Cover_Type")
        if target_name not in frame.columns:
            frame[target_name] = bunch.target
        frame = frame.rename(columns={target_name: "Cover_Type"})
        frame, path = _cache_frame(cache_dir, name, frame)
        return frame, path, "sklearn.datasets.fetch_covtype"
    if name == "housing":
        bunch = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
        frame = bunch.data.copy()
        frame["MEDV"] = pd.to_numeric(bunch.target, errors="raise")
        frame, path = _cache_frame(cache_dir, name, frame)
        return frame, path, "sklearn.datasets.fetch_openml:boston-v1"
    if name == "zoo":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
        raw_path = cache_dir / "zoo.data"
        urllib.request.urlretrieve(url, raw_path)
        frame = pd.read_csv(raw_path, header=None, names=ZOO_COLUMNS)
        frame, path = _cache_frame(cache_dir, name, frame)
        return frame, path, f"UCI:{url}"
    if name == "hog":
        try:
            from skimage.feature import hog
        except ImportError as error:
            raise FileNotFoundError(
                "HOG CSV was not found and scikit-image is unavailable for the documented fallback"
            ) from error
        bunch = load_digits()
        features = np.vstack(
            [
                hog(
                    image,
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1),
                    feature_vector=True,
                )
                for image in bunch.images
            ]
        )
        frame = pd.DataFrame(features, columns=[str(i) for i in range(features.shape[1])])
        frame["target"] = bunch.target
        frame, path = _cache_frame(cache_dir, name, frame)
        return frame, path, "generated:sklearn-digits+skimage-HOG(9,(4,4),(1,1))"
    raise KeyError(name)


def _numeric_features(frame: pd.DataFrame, target: str) -> list[str]:
    columns = [
        str(column)
        for column in frame.columns
        if str(column) != target and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise ValueError(
            f"no numeric feature columns remain after excluding target {target!r}; "
            f"columns={list(frame.columns)}"
        )
    return columns


def _drop_identifier_features(name: str, frame: pd.DataFrame, target: str) -> list[str]:
    identifier_aliases = {"animalname", "name", "id", "index", "rownumber"}
    features = []
    for column in frame.columns:
        column = str(column)
        if column == target or _normalise(column) in identifier_aliases:
            continue
        features.append(column)
    if not features:
        raise ValueError(f"{name}: no feature columns remain after target/identifier removal")
    return features


def _resolve_watermark_column(
    name: str, frame: pd.DataFrame, target: str, feature_columns: Sequence[str]
) -> tuple[str, str]:
    if name in {"synthetic", "housing"}:
        aliases = (
            ("dimension_0", "dimension0", "dim_0", "x0", "feature_0")
            if name == "synthetic"
            else ("MEDV", "median_value", "median_house_value", "target", "price")
        )
        if name == "housing":
            return target, "numeric"
        try:
            return (
                resolve_column(frame, aliases, dataset=name, purpose="numeric watermark column"),
                "numeric",
            )
        except ValueError as original_error:
            numeric = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
            if len(numeric) == 1:
                return numeric[0], "numeric"
            raise ValueError(
                f"{original_error}. Dynamic fallback found {len(numeric)} numeric features; "
                "refusing to choose ambiguously."
            ) from original_error
    return target, "categorical"


def _match_columns(name: str, frame: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    preferences = {
        "covertype": ("Elevation", "Aspect", "Slope"),
        "iris": ("sepal_length", "sepal_width", "petal_length", "petal_width"),
    }
    selected: list[str] = []
    for alias in preferences.get(name, ()):
        try:
            selected.append(
                resolve_column(frame, (alias,), dataset=name, purpose="matching column")
            )
        except ValueError:
            continue
    if len(selected) >= 3:
        return selected
    numeric = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    for column in numeric:
        if column not in selected:
            selected.append(column)
        if len(selected) == 3:
            break
    if len(selected) < 1:
        raise ValueError(
            f"{name}: cannot identify any stable numeric matching column; "
            f"features={list(feature_columns)}"
        )
    return selected


def load_dataset(
    name: str,
    data_dir: Path,
    cache_dir: Path,
    *,
    seed: int,
    allow_downloads: bool,
) -> DatasetBundle:
    name = name.casefold()
    if name not in FILE_NAMES:
        raise KeyError(f"unknown dataset {name!r}; expected one of {sorted(FILE_NAMES)}")
    candidates = _candidate_files(data_dir, name)
    cached_candidates = _candidate_files(cache_dir, name)
    if candidates:
        source_path, source_resolution = _select_candidate(
            candidates, name, scope="data_dir"
        )
        frame = _read_candidate(source_path, name)
        source = str(source_path)
    elif cached_candidates:
        source_path, source_resolution = _select_candidate(
            cached_candidates, name, scope="cache_dir"
        )
        frame = _read_candidate(source_path, name)
        source = f"cached fallback -> {source_path}"
    elif name in {"synthetic", "iris"} or allow_downloads:
        frame, source_path, source_label = _fetch_frame(name, cache_dir, seed)
        source_path = source_path.resolve()
        source = f"{source_label} -> {source_path}"
        source_resolution = {
            "scope": "generated_or_downloaded_fallback",
            "strategy": "generated_or_downloaded_fallback",
            "selected_path": str(source_path),
            "selected_sha256": sha256_file(source_path),
            "same_priority_candidates": [],
            "ignored_lower_priority_candidates": [],
        }
    else:
        raise FileNotFoundError(
            f"{name}: no input found under {data_dir}. Expected one of "
            f"{list(FILE_NAMES[name])}. Re-run without --no-downloads or provide the CSV."
        )
    frame = frame.dropna(how="all").reset_index(drop=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{name}: duplicate column names after trimming: {duplicates}")
    target = resolve_column(
        frame, TARGET_ALIASES[name], dataset=name, purpose="target column"
    )
    feature_columns = _drop_identifier_features(name, frame, target)
    watermark_column, watermark_kind = _resolve_watermark_column(
        name, frame, target, feature_columns
    )
    parameters = {
        "synthetic": (300, 40.0, 500),
        "covertype": (400, None, 500),
        "hog": (150, None, 500),
        "housing": (50, 25.0, 500),
        "zoo": (15, None, 500),
        "iris": (20, None, 500),
    }
    n, p, k = parameters[name]
    if len(frame) < n:
        raise ValueError(
            f"{name}: dataset has {len(frame)} rows, fewer than paper n={n}; "
            "provide the complete dataset instead of silently reducing watermark strength"
        )
    if watermark_kind == "numeric":
        converted = pd.to_numeric(frame[watermark_column], errors="coerce")
        invalid = int((converted.isna() & frame[watermark_column].notna()).sum())
        if invalid:
            raise ValueError(
                f"{name}: numeric watermark column {watermark_column!r} has "
                f"{invalid} non-numeric values"
            )
        frame[watermark_column] = converted
    if name != "housing" and frame[target].nunique(dropna=True) < 2:
        raise ValueError(f"{name}: target {target!r} has fewer than two classes")
    return DatasetBundle(
        name=name,
        frame=frame,
        target=target,
        watermark_kind=watermark_kind,
        watermark_column=watermark_column,
        n=n,
        p=p,
        k=k,
        source=source,
        source_sha256=sha256_file(source_path),
        feature_columns=feature_columns,
        match_columns=_match_columns(name, frame, feature_columns),
        source_resolution=source_resolution,
    )


def load_many(
    names: Sequence[str],
    data_dir: Path,
    cache_dir: Path,
    *,
    seed: int,
    allow_downloads: bool,
) -> dict[str, DatasetBundle]:
    return {
        name: load_dataset(
            name,
            data_dir,
            cache_dir,
            seed=seed,
            allow_downloads=allow_downloads,
        )
        for name in names
    }
