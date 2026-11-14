"""Command-line orchestration, manifest state, and verified resume support."""

from __future__ import annotations

import argparse
from filelock import FileLock, Timeout
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import sklearn

from .core import atomic_json, sha256_file, sha256_json
from .datasets import FILE_NAMES
from .experiments import ExperimentContext, STAGES


RUNNER_VERSION = "1.1.0"
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_DATA_DIR = Path("private/data")
DEFAULT_OUTPUT_DIR = Path("outputs/run")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gpu_ids(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("--gpu-id must contain at least one CUDA device index")
    for value in values:
        if not value.isdigit():
            raise ValueError(
                f"--gpu-id accepts comma-separated non-negative integers, got {raw!r}"
            )
    return values


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "scikit-image",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
        )
        return {
            "available": True,
            "command": command,
            "rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
        }
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {
            "available": False,
            "command": command,
            "error": f"{type(error).__name__}: {error}",
        }


@contextmanager
def _run_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".runner.lock"
    lock = FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError("another runner owns this output directory") from error
    try:
        yield
    finally:
        lock.release()


def _manifest_template(config: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "created_at_utc": now,
        "updated_at_utc": now,
        "config_fingerprint": fingerprint,
        "config": config,
        "stages": {
            name: {
                "status": "pending",
                "attempts": 0,
                "artifacts": [],
                "error": None,
            }
            for name in STAGES
        },
    }


def _load_manifest(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    fingerprint = sha256_json(config)
    if not path.exists():
        return _manifest_template(config, fingerprint)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(
            f"unsupported manifest schema {manifest.get('schema_version')!r} at {path}"
        )
    existing = manifest.get("config_fingerprint")
    if existing != fingerprint:
        raise RuntimeError(
            "output directory contains a manifest from a different computational "
            f"configuration ({existing} != {fingerprint}). Use a new --output-dir."
        )
    for name in STAGES:
        manifest.setdefault("stages", {}).setdefault(
            name,
            {"status": "pending", "attempts": 0, "artifacts": [], "error": None},
        )
    return manifest


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = _utc_now()
    atomic_json(path, manifest)


def _artifacts_valid(stage_state: dict[str, Any]) -> bool:
    artifacts = stage_state.get("artifacts", [])
    if not artifacts:
        return False
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            return False
    return True


def _computational_config(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    wanted_names = {
        file_name.casefold()
        for candidates in FILE_NAMES.values()
        for file_name in candidates
    }
    excluded_parts = {
        "watermarked",
        "alteration",
        "insertion",
        "deletion",
        "perturbed",
        "roc",
        "results",
    }
    input_files = []
    if data_dir.exists():
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file() or path.name.casefold() not in wanted_names:
                continue
            if any(part.casefold() in excluded_parts for part in path.parts):
                continue
            input_files.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "runner_version": RUNNER_VERSION,
        "data_dir": str(data_dir),
        "input_files": input_files,
        "output_dir": str(Path(args.output_dir).resolve()),
        "seed": args.seed,
        "alpha": args.alpha,
        "allow_downloads": args.allow_downloads,
        "utility_repeats": args.utility_repeats,
        "attack_repeats": args.attack_repeats,
        "sweep_repeats": args.sweep_repeats,
        "roc_neighbors": args.roc_neighbors,
        "workers": args.workers,
        "rf_trees": args.rf_trees,
        "xgb_trees": args.xgb_trees,
        "xgb_max_depth": args.xgb_max_depth,
        "device": args.device,
        "gpu_ids": list(_gpu_ids(args.gpu_id)),
        "attack_proportions": list(args.attack_proportions),
    }


def _context(args: argparse.Namespace) -> ExperimentContext:
    return ExperimentContext(
        data_dir=Path(args.data_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        seed=args.seed,
        alpha=args.alpha,
        allow_downloads=args.allow_downloads,
        utility_repeats=args.utility_repeats,
        attack_repeats=args.attack_repeats,
        sweep_repeats=args.sweep_repeats,
        roc_neighbors=args.roc_neighbors,
        workers=args.workers,
        rf_trees=args.rf_trees,
        xgb_trees=args.xgb_trees,
        xgb_max_depth=args.xgb_max_depth,
        device=args.device,
        gpu_ids=_gpu_ids(args.gpu_id),
        attack_proportions=tuple(args.attack_proportions),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.utility_repeats < 1 or args.attack_repeats < 1 or args.sweep_repeats < 1:
        raise ValueError("all repeat counts must be positive")
    if args.roc_neighbors < 4:
        raise ValueError("--roc-neighbors must be at least 4")
    if args.workers < 1 or args.rf_trees < 1 or args.xgb_trees < 1:
        raise ValueError("worker/tree counts must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie strictly between 0 and 1")
    if any(not 0.0 <= value <= 1.0 for value in args.attack_proportions):
        raise ValueError("all --attack-proportions must lie in [0,1]")
    if len(set(args.attack_proportions)) != len(args.attack_proportions):
        raise ValueError("--attack-proportions must not contain duplicates")
    _gpu_ids(args.gpu_id)


def _apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.utility_repeats = 1
    args.attack_repeats = 1
    args.sweep_repeats = 1
    args.roc_neighbors = 8
    args.rf_trees = min(args.rf_trees, 10)
    args.xgb_trees = min(args.xgb_trees, 5)
    args.xgb_max_depth = min(args.xgb_max_depth, 3)
    args.attack_proportions = (0.2, 1.0)


def run(args: argparse.Namespace) -> int:
    _apply_quick(args)
    _validate_args(args)
    context = _context(args)
    output_dir = context.output_dir
    manifest_path = output_dir / "manifest.json"
    with _run_lock(output_dir):
        config = _computational_config(args)
        manifest = _load_manifest(manifest_path, config)
        _save_manifest(manifest_path, manifest)
        environment_path = output_dir / "environment.json"
        if not environment_path.exists():
            atomic_json(
                environment_path,
                {
                    "generated_at_utc": _utc_now(),
                    "python": sys.version,
                    "python_executable": sys.executable,
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "packages": _package_versions(),
                    "runtime_versions": {
                        "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__,
                    },
                    "nvidia_smi": _nvidia_smi(),
                    "config": config,
                },
            )
        failures = []
        selected = list(args.stages)
        for stage_name in selected:
            state = manifest["stages"][stage_name]
            forced = args.force_all or stage_name in args.force_stage
            if (
                args.resume
                and not forced
                and state.get("status") == "completed"
                and _artifacts_valid(state)
            ):
                print(f"[resume] {stage_name}: completed artifacts verified; skipping")
                continue
            state.update(
                {
                    "status": "running",
                    "started_at_utc": _utc_now(),
                    "completed_at_utc": None,
                    "attempts": int(state.get("attempts", 0)) + 1,
                    "error": None,
                }
            )
            _save_manifest(manifest_path, manifest)
            print(f"[run] {stage_name}", flush=True)
            try:
                artifact_paths = STAGES[stage_name](context)
                missing = [str(path) for path in artifact_paths if not path.is_file()]
                if missing:
                    raise RuntimeError(
                        f"stage returned missing artifact paths: {missing}"
                    )
                state.update(
                    {
                        "status": "completed",
                        "completed_at_utc": _utc_now(),
                        "artifacts": [
                            {
                                "path": str(path.resolve()),
                                "bytes": path.stat().st_size,
                                "sha256": sha256_file(path),
                            }
                            for path in artifact_paths
                        ],
                    }
                )
                print(f"[done] {stage_name}: {len(artifact_paths)} artifacts", flush=True)
            except Exception as error:
                state.update(
                    {
                        "status": "failed",
                        "completed_at_utc": _utc_now(),
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    }
                )
                failures.append(stage_name)
                print(
                    f"[failed] {stage_name}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                _save_manifest(manifest_path, manifest)
                if not args.keep_going:
                    break
            finally:
                _save_manifest(manifest_path, manifest)
        if failures:
            print(
                f"Run finished with failed stages: {', '.join(failures)}. "
                f"Inspect {manifest_path} and rerun the same command to resume.",
                file=sys.stderr,
            )
            return 1
        print(f"All selected stages completed. Manifest: {manifest_path}")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic, restartable TabularMark RQ1/RQ2/RQ3/RQ5 suite."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cpu",
        help="Requested XGBoost device. CUDA failures are recorded and retried on CPU.",
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="One GPU id or comma-separated ids assigned deterministically to XGBoost fits (default: 0).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--xgb-trees", type=int, default=30)
    parser.add_argument("--xgb-max-depth", type=int, default=10)
    parser.add_argument("--utility-repeats", type=int, default=5)
    parser.add_argument("--attack-repeats", type=int, default=3)
    parser.add_argument("--sweep-repeats", type=int, default=3)
    parser.add_argument("--roc-neighbors", type=int, default=400)
    parser.add_argument(
        "--attack-proportions",
        type=float,
        nargs="+",
        default=(0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(STAGES),
        default=tuple(STAGES),
        help="Subset of independently resumable stages to run.",
    )
    parser.add_argument(
        "--allow-downloads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow missing public datasets to be cached under the new output directory.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify and skip completed stages (default: true).",
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        choices=tuple(STAGES),
        default=[],
        help="Recompute one completed stage; may be repeated.",
    )
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue independent stages after a failure; all failures remain in manifest.json.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test settings only: one repeat, 8 ROC neighbors, small models, 20/100%% attacks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
