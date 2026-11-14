#!/usr/bin/env python3
"""Run the corrected, deterministic Covertype comparison for RQ4."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tabularmark_ext.rq4 import (  # noqa: E402
    WatermarkBundle,
    build_rq4_watermarks,
    detect_histmark,
    detect_semmark,
    detect_tabularmark,
    json_ready,
    load_covertype,
    prepare_balanced_subset,
    run_alteration_comparison,
    sha256_file,
    write_json,
    xgboost_utility,
)
from tabularmark_runner.core import atomic_private_json


def parse_percentages(raw: str) -> List[float]:
    """Parse comma-separated integer percentages or START:STOP:STEP."""

    raw = raw.strip()
    values: List[int]
    if ":" in raw:
        parts = [int(value.strip()) for value in raw.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("percentage range must be START:STOP:STEP")
        start, stop, step = parts
        if step <= 0:
            raise argparse.ArgumentTypeError("percentage STEP must be positive")
        values = list(range(start, stop + 1, step))
        if not values or values[-1] != stop:
            raise argparse.ArgumentTypeError("percentage STEP must land exactly on STOP")
    else:
        try:
            values = [int(value.strip()) for value in raw.split(",") if value.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("percentages must be integers") from exc
    if not values or min(values) < 0 or max(values) > 100:
        raise argparse.ArgumentTypeError("percentages must be a non-empty subset of 0..100")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("percentages must not contain duplicates")
    return [value / 100.0 for value in sorted(values)]


def _slug(name: str) -> str:
    return {"TabularMark": "tabularmark", "HistMark": "histmark", "SemMark": "semmark"}[name]


def _config_hash(payload: Dict[str, object]) -> str:
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic RQ4 comparison of TabularMark, HistMark, and SemMark on a "
            "class-balanced Covertype subset. Existing official notebooks are never modified."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Covertype CSV or UCI covtype.data")
    parser.add_argument("--output-dir", required=True, type=Path, help="New result directory")
    parser.add_argument(
        "--input-format",
        choices=["auto", "uci-raw", "headered"],
        default="auto",
        help="Use uci-raw for the original 55-column headerless file",
    )
    parser.add_argument("--target-column", default="Cover_Type")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument(
        "--allow-small-classes",
        action="store_true",
        help="Permit fewer rows than --per-class; intended only for labelled smoke tests",
    )
    parser.add_argument("--n-key-cells", type=int, default=400)
    parser.add_argument("--watermark-length", type=int, default=400)
    parser.add_argument("--semmark-votes-per-bit", type=int, default=4)
    parser.add_argument(
        "--alteration-percentages",
        type=parse_percentages,
        default=parse_percentages("0:100:10"),
        help="Default: 0:100:10. The same nested attack plan is used for all schemes.",
    )
    parser.add_argument("--attack-repeats", type=int, default=5)
    parser.add_argument(
        "--device",
        default="cpu",
        help="XGBoost device: cpu, cuda, or cuda:N. Explicit CUDA failure is not hidden.",
    )
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip XGBoost and record NOT_RUN_BY_REQUEST; never fabricates utility values",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory so runs cannot mix."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    source = load_covertype(input_path, args.input_format, args.target_column)
    balanced = prepare_balanced_subset(
        source,
        per_class=args.per_class,
        seed=args.seed,
        target_column=args.target_column,
        strict=not args.allow_small_classes,
    )
    if args.n_key_cells > len(balanced):
        raise ValueError("--n-key-cells cannot exceed the prepared subset size")

    config: Dict[str, object] = {
        "schema_version": 1,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_format": args.input_format,
        "target_column": args.target_column,
        "seed": args.seed,
        "per_class": args.per_class,
        "allow_small_classes": bool(args.allow_small_classes),
        "n_key_cells": args.n_key_cells,
        "watermark_length": args.watermark_length,
        "semmark_votes_per_bit": args.semmark_votes_per_bit,
        "alteration_percentages": [int(round(value * 100)) for value in args.alteration_percentages],
        "attack_repeats": args.attack_repeats,
        "device": args.device,
        "n_estimators": args.n_estimators,
        "test_size": args.test_size,
        "skip_ml": bool(args.skip_ml),
    }
    config["run_id"] = _config_hash(config)[:16]
    write_json(output_dir / "run_config.json", config)

    balanced_path = output_dir / "covertype_balanced_original.csv"
    balanced.to_csv(balanced_path, index=False)
    bundles = build_rq4_watermarks(
        balanced,
        seed=args.seed,
        n_key_cells=args.n_key_cells,
        watermark_length=args.watermark_length,
        semmark_votes_per_bit=args.semmark_votes_per_bit,
        target_column=args.target_column,
    )

    artifacts: Dict[str, object] = {
        "original": {
            "path": balanced_path.name,
            "sha256": sha256_file(balanced_path),
            "rows": len(balanced),
            "columns": len(balanced.columns),
            "class_counts": {
                str(key): int(value)
                for key, value in balanced[args.target_column].value_counts().sort_index().items()
            },
        },
        "watermarked": {},
    }
    detectors = {
        "TabularMark": detect_tabularmark,
        "HistMark": detect_histmark,
        "SemMark": detect_semmark,
    }
    baseline_diagnostics: Dict[str, object] = {}
    for name in sorted(bundles):
        bundle: WatermarkBundle = bundles[name]
        slug = _slug(name)
        data_path = output_dir / f"covertype_watermarked_{slug}.csv"
        state_path = output_dir / f"watermark_state_{slug}.json"
        bundle.data.to_csv(data_path, index=False)
        atomic_private_json(state_path, json_ready(bundle.state))
        baseline_detection = detectors[name](bundle.data, bundle.state)
        baseline_diagnostics[name] = baseline_detection
        artifacts["watermarked"][name] = {
            "data_path": data_path.name,
            "data_sha256": sha256_file(data_path),
            "state_path": state_path.name,
            "state_sha256": sha256_file(state_path),
            "baseline_detection": baseline_detection,
        }
    write_json(output_dir / "watermark_artifacts.json", artifacts)

    repeats_frame, summary_frame = run_alteration_comparison(
        bundles,
        fractions=args.alteration_percentages,
        repeats=args.attack_repeats,
        seed=args.seed,
        target_column=args.target_column,
    )
    repeats_frame.to_csv(output_dir / "alteration_repeats.csv", index=False, float_format="%.12g")
    summary_frame.to_csv(output_dir / "alteration_summary.csv", index=False, float_format="%.12g")
    write_json(
        output_dir / "alteration_results.json",
        {
            "status": "COMPLETED",
            "metric": "mismatch_percentage",
            "attack_semantics": (
                "For each repeat, percentages are nested prefixes of one deterministic row "
                "permutation. Replacement labels are uniform on classes 1..7 and identical "
                "across schemes; as in the official notebook, a selected cell can receive its "
                "existing value."
            ),
            "repeats": repeats_frame.to_dict(orient="records"),
            "summary": summary_frame.to_dict(orient="records"),
        },
    )

    if args.skip_ml:
        utility_status: Dict[str, object] = {
            "status": "NOT_RUN_BY_REQUEST",
            "reason": "--skip-ml was explicitly supplied; no utility values were generated.",
        }
    else:
        utility_frame, utility_status = xgboost_utility(
            balanced,
            bundles,
            seed=args.seed,
            device=args.device,
            n_estimators=args.n_estimators,
            test_size=args.test_size,
            target_column=args.target_column,
        )
        utility_frame.to_csv(output_dir / "xgboost_utility.csv", index=False, float_format="%.12g")
        write_json(
            output_dir / "xgboost_utility.json",
            {"metadata": utility_status, "metrics": utility_frame.to_dict(orient="records")},
        )
    write_json(output_dir / "xgboost_status.json", utility_status)

    completed = not args.skip_ml
    undecidable_bits = {
        name: int(result.get("undecidable_bits", 0))
        for name, result in baseline_diagnostics.items()
    }
    embedding_status = (
        "COMPLETED_WITH_UNDECIDABLE_BITS"
        if any(undecidable_bits.values())
        else "COMPLETED"
    )
    run_status = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "overall_status": (
            "COMPLETED_WITH_DIAGNOSTICS"
            if completed and embedding_status != "COMPLETED"
            else "COMPLETED"
            if completed
            else "PARTIAL_EXPLICIT_ML_NOT_RUN"
        ),
        "stages": {
            "data_preparation": "COMPLETED",
            "watermark_embedding": embedding_status,
            "alteration_0_to_100": "COMPLETED",
            "xgboost_utility": utility_status["status"],
        },
        "embedding_diagnostics": {
            "undecidable_bits_by_scheme": undecidable_bits,
            "baseline_detection": baseline_diagnostics,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "result_policy": (
            "Only values created by this invocation are in result files. Official notebook "
            "outputs and paper numbers are never copied into measured-result fields."
        ),
    }
    write_json(output_dir / "run_status.json", run_status)
    print(json.dumps(run_status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
