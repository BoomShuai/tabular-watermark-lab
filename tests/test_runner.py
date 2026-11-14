from __future__ import annotations

import importlib.util
import json
import math
import stat
import tempfile
import time
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tabularmark_runner.core import (
    categorical_effective_gamma,
    decision_threshold,
    detect_categorical,
    detect_categorical_details,
    detect_categorical_matched,
    detect_categorical_matched_details,
    detect_numeric,
    embed_categorical,
    embed_numeric,
    make_key,
    match_key_rows,
    z_statistic,
)
from tabularmark_runner.datasets import DatasetBundle, load_dataset
from tabularmark_runner.experiments import (
    ExperimentContext,
    _housing_utility_missing_data,
    run_rq1_roc,
    run_rq2_utility,
    run_rq3_iris_attacks,
    run_rq3_row_attacks,
    run_rq5_sweeps,
)
from tabularmark_runner.models import (
    ModelResult,
    evaluate_classification,
    evaluate_regression,
    result_metric_rows,
)
from tabularmark_runner.runner import main


class CoreTests(unittest.TestCase):
    def test_numeric_round_trip_is_detectable_and_non_mutating(self) -> None:
        original = pd.DataFrame({"x": np.linspace(-2.0, 2.0, 60)})
        before = original.copy(deep=True)
        key = make_key(len(original), 30, 123)
        watermarked = embed_numeric(original, "x", key, p=2.0, k=20)
        green, score = detect_numeric(original, watermarked, "x", key, 2.0, 20)
        pd.testing.assert_frame_equal(original, before)
        self.assertEqual(green, key.n)
        self.assertAlmostEqual(score, math.sqrt(key.n))

    def test_categorical_round_trip_and_ordered_matching(self) -> None:
        original = pd.DataFrame(
            {
                "a": np.arange(60),
                "b": np.arange(60) * 2,
                "target": np.tile(np.arange(6), 10),
            }
        )
        key = make_key(len(original), 30, 456)
        categories = list(range(6))
        watermarked = embed_categorical(original, "target", key, categories)
        shuffled = watermarked.sample(frac=1.0, random_state=1).reset_index(drop=True)
        matches = match_key_rows(original, shuffled, key.indices, ["a", "b"])
        green, matched, score = detect_categorical_matched(
            shuffled, "target", key, categories, matches
        )
        self.assertEqual(matched, key.n)
        self.assertEqual(green, key.n)
        self.assertAlmostEqual(score, math.sqrt(key.n))
        direct_green, _ = detect_categorical(watermarked, "target", key, categories)
        self.assertEqual(direct_green, key.n)

    def test_odd_categorical_domain_uses_floor_and_effective_gamma(self) -> None:
        categories = list(range(7))
        original = pd.DataFrame(
            {
                "row_id": np.arange(70),
                "target": np.tile(categories, 10),
            }
        )
        key = make_key(len(original), 35, 789, gamma=0.5)
        watermarked = embed_categorical(original, "target", key, categories)
        details = detect_categorical_details(
            watermarked, "target", key, categories
        )
        self.assertAlmostEqual(categorical_effective_gamma(categories, 0.5), 3 / 7)
        self.assertEqual(details.nominal_gamma, 0.5)
        self.assertAlmostEqual(details.effective_gamma, 3 / 7)
        self.assertEqual(details.green_count, key.n)
        self.assertAlmostEqual(
            details.z_score, z_statistic(key.n, key.n, 3 / 7)
        )

        shuffled = watermarked.sample(frac=1.0, random_state=3).reset_index(drop=True)
        matches = match_key_rows(original, shuffled, key.indices, ["row_id"])
        matched = detect_categorical_matched_details(
            shuffled, "target", key, categories, matches
        )
        self.assertEqual(matched.matched_count, key.n)
        self.assertEqual(matched.green_count, key.n)
        self.assertAlmostEqual(matched.effective_gamma, 3 / 7)
        self.assertAlmostEqual(
            matched.z_score, z_statistic(key.n, key.n, 3 / 7)
        )

    def test_threshold(self) -> None:
        self.assertAlmostEqual(decision_threshold(0.05), 1.9599639845400536)

    def test_matching_duplicate_signatures_is_stable_and_one_to_one(self) -> None:
        original = pd.DataFrame(
            {
                "a": [1.0, 1.0, 1.0, np.nan],
                "b": ["same", "same", "same", "null"],
            }
        )
        suspect = pd.DataFrame(
            {
                "a": [1.0, 1.0, np.nan],
                "b": ["same", "same", "null"],
            },
            index=[101, 55, 80],
        )
        matches = match_key_rows(original, suspect, [1, 0, 2, 3], ["a", "b"])
        self.assertEqual(matches, [101, 55, None, 80])

    def test_matching_581012_rows_uses_vectorized_candidate_filter(self) -> None:
        row_count = 581_012
        original = pd.DataFrame(
            {
                "Elevation": np.arange(row_count, dtype=np.int32),
                "Aspect": np.arange(row_count, dtype=np.int32) % 360,
            }
        )
        key_indices = [0, 17, 12_345, 400_000, row_count - 1]
        started = time.perf_counter()
        matches = match_key_rows(
            original, original, key_indices, ["Elevation", "Aspect"]
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(matches, key_indices)
        # This is deliberately a loose regression guard, not a stress benchmark.
        self.assertLess(elapsed, 10.0)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "xgboost is optional")
    def test_xgboost_cpu_device_verification(self) -> None:
        rng = np.random.default_rng(99)
        frame = pd.DataFrame(
            {
                "x": rng.normal(size=80),
                "y": rng.normal(size=80),
                "target": np.tile([0, 1], 40),
            }
        )
        result = evaluate_classification(
            frame,
            frame.copy(deep=True),
            "target",
            ["x", "y"],
            np.arange(60),
            np.arange(60, 80),
            model="xgboost",
            seed=99,
            workers=1,
            rf_trees=2,
            xgb_trees=2,
            xgb_max_depth=2,
            device="cpu",
            gpu_id=None,
        )
        self.assertEqual(result.requested_device, "cpu")
        self.assertEqual(result.used_device, "cpu")
        self.assertIsNone(result.fallback_reason)

    def test_housing_regression_uses_predictor_and_target_complete_cases(self) -> None:
        clean = pd.DataFrame(
            {
                "RM": [5.0, np.nan, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, np.nan, 9.5],
                "LSTAT": np.linspace(20.0, 2.0, 10),
                "MEDV": [10.0, 12.0, np.nan, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0],
            }
        )
        result = evaluate_regression(
            clean,
            clean.copy(deep=True),
            "MEDV",
            ["RM", "LSTAT"],
            np.arange(7),
            np.arange(7, 10),
        )
        self.assertEqual(
            result.missing_value_strategy,
            "complete_case_after_split_on_predictors_plus_target",
        )
        self.assertEqual(result.input_train_rows, 7)
        self.assertEqual(result.effective_train_rows, 5)
        self.assertEqual(result.input_test_rows, 3)
        self.assertEqual(result.effective_test_rows, 2)
        rows = result_metric_rows(result, {"dataset": "housing"})
        self.assertTrue(rows)
        self.assertEqual(rows[0]["dropped_train_rows"], 2)
        self.assertEqual(rows[0]["dropped_test_rows"], 1)
        metadata = _housing_utility_missing_data(
            pd.DataFrame(rows),
            DatasetBundle(
                name="housing",
                frame=clean,
                target="MEDV",
                watermark_kind="numeric",
                watermark_column="RM",
                n=30,
                p=None,
                k=1,
                source="HousingData.csv",
                source_sha256="test",
                feature_columns=["RM", "LSTAT"],
                match_columns=["LSTAT", "MEDV"],
            ),
        )
        self.assertEqual(metadata["complete_case_columns"], ["RM", "LSTAT", "MEDV"])
        self.assertIsNone(metadata["imputation"])
        self.assertEqual(len(metadata["effective_samples_by_fit"]), 1)
        self.assertEqual(metadata["effective_samples_by_fit"][0]["effective_train_rows"], 5)


class DatasetResolutionTests(unittest.TestCase):
    @staticmethod
    def _housing_frame(offset: float = 0.0) -> pd.DataFrame:
        count = 60
        return pd.DataFrame(
            {
                "RM": np.linspace(4.0, 9.0, count),
                "LSTAT": np.linspace(25.0, 1.0, count),
                "MEDV": np.linspace(8.0, 45.0, count) + offset,
            }
        )

    def test_identical_same_priority_candidates_are_selected_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            cache_dir = root / "cache"
            first = data_dir / "a" / "HousingData.csv"
            second = data_dir / "b" / "HousingData.csv"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            frame = self._housing_frame()
            frame.to_csv(first, index=False)
            frame.to_csv(second, index=False)

            bundle = load_dataset(
                "housing",
                data_dir,
                cache_dir,
                seed=10_000,
                allow_downloads=False,
            )
            resolution = bundle.metadata()["source_resolution"]
            self.assertEqual(
                resolution["strategy"],
                "stable_path_among_identical_same_priority_candidates",
            )
            self.assertEqual(len(resolution["same_priority_candidates"]), 2)
            self.assertEqual(Path(resolution["selected_path"]), first.resolve())
            self.assertEqual(
                len(
                    {
                        candidate["sha256"]
                        for candidate in resolution["same_priority_candidates"]
                    }
                ),
                1,
            )

    def test_conflicting_same_priority_candidates_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            cache_dir = root / "cache"
            first = data_dir / "a" / "HousingData.csv"
            second = data_dir / "b" / "HousingData.csv"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            self._housing_frame().to_csv(first, index=False)
            self._housing_frame(offset=1.0).to_csv(second, index=False)

            with self.assertRaisesRegex(ValueError, "conflicting data_dir candidates"):
                load_dataset(
                    "housing",
                    data_dir,
                    cache_dir,
                    seed=10_000,
                    allow_downloads=False,
                )


class RunnerResumeTests(unittest.TestCase):
    @staticmethod
    def _write_fixture(data_dir: Path) -> None:
        rng = np.random.default_rng(7)
        count = 500
        synthetic = pd.DataFrame(
            {
                "dimension_0": rng.normal(0, 20, count),
                "dimension_1": rng.normal(0, 20, count),
                "target": np.tile([0, 1], count // 2),
            }
        )
        covertype = pd.DataFrame(
            {
                "Elevation": np.arange(count),
                "Aspect": np.arange(count) % 360,
                "Slope": np.arange(count) % 40,
                "feature": rng.normal(size=count),
                "Cover_Type": np.arange(count) % 7 + 1,
            }
        )
        hog = pd.DataFrame(
            {
                "0": rng.normal(size=count),
                "1": rng.normal(size=count),
                "target": np.arange(count) % 10,
            }
        )
        housing = pd.DataFrame(
            {
                "RM": rng.normal(6, 1, count),
                "LSTAT": rng.normal(12, 4, count),
                "MEDV": rng.normal(25, 8, count),
            }
        )
        synthetic.to_csv(data_dir / "synthetic_data.csv", index=False)
        covertype.to_csv(data_dir / "cover_type_with_columns.csv", index=False)
        hog.to_csv(data_dir / "digits_HOG.csv", index=False)
        housing.to_csv(data_dir / "HousingData.csv", index=False)

    def test_detectability_stage_resumes_after_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            self._write_fixture(data_dir)
            arguments = [
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--stages",
                "rq1_detectability",
                "--quick",
                "--device",
                "cpu",
                "--gpu-id",
                "0",
                "--no-allow-downloads",
            ]
            self.assertEqual(main(arguments), 0)
            self.assertEqual(main(arguments), 0)
            with (output_dir / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            state = manifest["stages"]["rq1_detectability"]
            self.assertEqual(manifest["runner_version"], "1.1.0")
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["attempts"], 1)
            self.assertTrue((output_dir / "rq1_detectability.csv").is_file())
            detection = pd.read_csv(output_dir / "rq1_detectability.csv")
            self.assertTrue(
                {"gamma", "nominal_gamma", "effective_gamma"}.issubset(
                    detection.columns
                )
            )
            covertype_original = detection.loc[
                (detection["dataset"] == "covertype")
                & (detection["suspect"] == "original")
            ].iloc[0]
            self.assertAlmostEqual(float(covertype_original["nominal_gamma"]), 0.5)
            self.assertAlmostEqual(
                float(covertype_original["effective_gamma"]), 3 / 7
            )
            self.assertAlmostEqual(
                float(covertype_original["z_score"]),
                z_statistic(
                    int(covertype_original["green_count"]),
                    int(covertype_original["n"]),
                    3 / 7,
                ),
            )
            self.assertFalse(bool(covertype_original["detected"]))
            with (output_dir / "rq1_detectability.json").open(
                encoding="utf-8"
            ) as handle:
                detection_json = json.load(handle)
            self.assertIn("gamma_semantics", detection_json)
            self.assertTrue(
                all(
                    "nominal_gamma" in record and "effective_gamma" in record
                    for record in detection_json["records"]
                )
            )
            key_paths = sorted((output_dir / "artifacts" / "keys").glob("*.json"))
            self.assertEqual(len(key_paths), 4)
            for key_path in key_paths:
                if os.name != 'nt':
                    self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)

    def test_rq2_clean_and_watermarked_share_model_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame(
                {
                    "x": np.linspace(-1.0, 1.0, 60),
                    "target": np.tile(np.arange(6), 10),
                }
            )
            bundle = DatasetBundle(
                name="hog",
                frame=frame,
                target="target",
                watermark_kind="categorical",
                watermark_column="target",
                n=20,
                p=None,
                k=500,
                source="in-memory fixture",
                source_sha256=None,
                feature_columns=["x"],
                match_columns=["x"],
            )

            class FixtureBundles(dict[str, DatasetBundle]):
                def __missing__(self, key: str) -> DatasetBundle:
                    if key == "housing":
                        return bundle
                    raise KeyError(key)

            bundles = FixtureBundles(hog=bundle)
            context = ExperimentContext(
                data_dir=root / "data",
                output_dir=root / "output",
                seed=10_000,
                alpha=0.05,
                allow_downloads=False,
                utility_repeats=2,
                attack_repeats=1,
                sweep_repeats=1,
                roc_neighbors=8,
                workers=1,
                rf_trees=2,
                xgb_trees=2,
                xgb_max_depth=2,
                device="cpu",
                gpu_ids=("0",),
                attack_proportions=(0.2,),
            )
            model_result = ModelResult(
                metrics={"accuracy": 1.0, "macro_f1": 1.0},
                model="random_forest",
                requested_device="cpu",
                used_device="cpu",
                gpu_id=None,
                fallback_reason=None,
                missing_value_strategy="none",
                input_train_rows=42,
                input_test_rows=18,
                effective_train_rows=42,
                effective_test_rows=18,
            )
            with patch.object(
                ExperimentContext, "datasets", return_value=bundles
            ), patch(
                "tabularmark_runner.experiments._evaluate_model",
                return_value=model_result,
            ) as evaluate, patch(
                "tabularmark_runner.experiments._housing_utility_missing_data",
                return_value={},
            ):
                artifacts = run_rq2_utility(context)

            utility = pd.read_csv(artifacts[0])
            for _, group in utility.groupby(["dataset", "repeat"]):
                self.assertEqual(set(group["variant"]), {"original", "watermarked"})
                self.assertEqual(group["model_seed"].nunique(), 1)
            self.assertEqual(evaluate.call_count, 4)
            for offset in range(0, evaluate.call_count, 2):
                self.assertEqual(
                    evaluate.call_args_list[offset].kwargs["seed"],
                    evaluate.call_args_list[offset + 1].kwargs["seed"],
                )

    def test_non_xgboost_stages_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            self._write_fixture(data_dir)
            context = ExperimentContext(
                data_dir=data_dir,
                output_dir=output_dir,
                seed=10_000,
                alpha=0.05,
                allow_downloads=False,
                utility_repeats=1,
                attack_repeats=1,
                sweep_repeats=1,
                roc_neighbors=8,
                workers=1,
                rf_trees=5,
                xgb_trees=2,
                xgb_max_depth=2,
                device="cpu",
                gpu_ids=("0",),
                attack_proportions=(0.2, 1.0),
            )
            for stage in (
                run_rq1_roc,
                run_rq3_row_attacks,
                run_rq3_iris_attacks,
                run_rq5_sweeps,
            ):
                artifacts = stage(context)
                self.assertTrue(artifacts)
                self.assertTrue(all(path.is_file() for path in artifacts))


if __name__ == "__main__":
    unittest.main()
